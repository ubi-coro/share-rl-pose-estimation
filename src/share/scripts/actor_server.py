#!/usr/bin/env python
import logging
import os
import time
from functools import lru_cache
from queue import Empty
from statistics import mean, quantiles
from typing import Any

import grpc
import torch
from lerobot.policies.sac.configuration_sac import SACConfig
from torch.multiprocessing import Event, Queue

from lerobot.configs import parser
from lerobot.processor import TransitionKey
from lerobot.rl.process import ProcessSignalHandler
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import Transition, move_transition_to_device
from lerobot.utils.utils import get_safe_torch_device, init_logging

from share.configs.rl import MPNetTrainRLServerPipelineConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)
from share.rl.runtime import (
    PrimitiveBudgetCounter,
    apply_parameter_updates_from_queue,
    build_adaptive_registry,
    make_policies_for_registry, sanitize_local_grpc_proxy_env,
)
from share.teleoperators import TeleopEvents


class _CompositeShutdownEvent:
    def __init__(self, *events: Any):
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


@parser.wrap()
def actor_cli(cfg: MPNetTrainRLServerPipelineConfig):
    cfg.validate()
    run_actor(cfg)


def run_actor(cfg: MPNetTrainRLServerPipelineConfig, shutdown_event: Any | None = None) -> dict[str, Any]:
    registry = build_adaptive_registry(cfg.env, cfg.policy)
    is_threaded = _use_threads(registry.actor_learner_policy_cfg)

    if not is_threaded:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # Initialize robot/env first, before any gRPC
    mp_net = ManipulationPrimitiveNet(cfg.env)

    external_shutdown_event = shutdown_event
    if external_shutdown_event is None:
        actor_shutdown_event = ProcessSignalHandler(is_threaded, display_pid=not is_threaded).shutdown_event
        runtime_shutdown_event: Any = actor_shutdown_event
    else:
        if is_threaded:
            from threading import Event as ThreadEvent

            actor_shutdown_event = ThreadEvent()
        else:
            import torch.multiprocessing as mp

            actor_shutdown_event = mp.Event()
        runtime_shutdown_event = _CompositeShutdownEvent(actor_shutdown_event, external_shutdown_event)

    log_dir = os.path.join(str(cfg.output_dir), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")
    init_logging(log_file=log_file, display_pid=not is_threaded)
    logging.info("Actor logging initialized, writing to %s", log_file)

    transport_cfg = registry.actor_learner_policy_cfg.actor_learner_config
    learner_client, grpc_channel = learner_service_client(
        host=transport_cfg.learner_host,
        port=transport_cfg.learner_port,
    )
    if not establish_learner_connection(learner_client, runtime_shutdown_event):
        raise RuntimeError("Failed to establish connection with learner.")

    if not is_threaded:
        grpc_channel.close()
        grpc_channel = None

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    if is_threaded:
        from threading import Thread as ConcurrencyEntity
    else:
        from multiprocessing import Process as ConcurrencyEntity

    receive_policy_worker = ConcurrencyEntity(
        target=receive_policy,
        args=(
            runtime_shutdown_event,
            parameters_queue,
            transport_cfg.learner_host,
            transport_cfg.learner_port,
            grpc_channel,
        ),
        daemon=True,
    )
    transitions_worker = ConcurrencyEntity(
        target=send_transitions,
        args=(
            runtime_shutdown_event,
            transitions_queue,
            transport_cfg.learner_host,
            transport_cfg.learner_port,
            transport_cfg.queue_get_timeout,
            grpc_channel,
        ),
        daemon=True,
    )
    interactions_worker = ConcurrencyEntity(
        target=send_interactions,
        args=(
            runtime_shutdown_event,
            interactions_queue,
            transport_cfg.learner_host,
            transport_cfg.learner_port,
            transport_cfg.queue_get_timeout,
            grpc_channel,
        ),
        daemon=True,
    )

    receive_policy_worker.start()
    transitions_worker.start()
    interactions_worker.start()

    try:
        result = act_with_policy(
            cfg=cfg,
            env=mp_net,
            registry=registry,
            shutdown_event=runtime_shutdown_event,
            parameters_queue=parameters_queue,
            transitions_queue=transitions_queue,
            interactions_queue=interactions_queue,
        )
    finally:
        actor_shutdown_event.set()

        transitions_worker.join()
        interactions_worker.join()
        receive_policy_worker.join()

        transitions_queue.close()
        interactions_queue.close()
        parameters_queue.close()

        transitions_queue.cancel_join_thread()
        interactions_queue.cancel_join_thread()
        parameters_queue.cancel_join_thread()

    return result


def act_with_policy(
    cfg: MPNetTrainRLServerPipelineConfig,
    env: ManipulationPrimitiveNet,
    registry: Any,
    shutdown_event: Any,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
) -> dict[str, Any]:
    set_seed(cfg.seed)

    device = get_safe_torch_device(registry.actor_learner_policy_cfg.device, log=True)

    policies = make_policies_for_registry(env.config, registry, train_mode=False)
    step_counter = PrimitiveBudgetCounter(registry.online_step_budgets)
    applied_parameter_updates = 0

    def reset_segment_state() -> dict[str, Any]:
        return {
            "reward_sum": 0.0,
            "intervention_steps": 0,
            "total_steps": 0,
            "policy_inference_dts": [],
            "pending_transitions": [],
        }

    # Give the streaming worker a short window to fetch initial learner parameters.
    push_frequency = registry.actor_learner_policy_cfg.actor_learner_config.policy_parameters_push_frequency
    warmup_deadline = time.time() + max(5.0, 20.0 * push_frequency)
    while time.time() < warmup_deadline and not shutdown_event.is_set():
        consumed_updates = apply_parameter_updates_from_queue(
            policies=policies,
            parameters_queue=parameters_queue,
            device=device,
        )
        applied_parameter_updates += consumed_updates
        if consumed_updates > 0:
            break
        time.sleep(0.01)

    transition = env.reset()
    segment_state = reset_segment_state()

    try:
        while not step_counter.all_finished and not shutdown_event.is_set():
            step_start_t = time.perf_counter()
            active_primitive = env.active_primitive
            policy = policies.get(active_primitive)
            obs = transition[TransitionKey.OBSERVATION]

            consumed_updates = apply_parameter_updates_from_queue(
                policies=policies,
                parameters_queue=parameters_queue,
                device=str(device),
            )
            applied_parameter_updates += consumed_updates

            if policy is not None:
                inference_t0 = time.perf_counter()

                policy_obs = {key: value for key, value in obs.items() if key in policy.config.input_features}
                action = policy.select_action(batch=policy_obs)

                inference_dt = time.perf_counter() - inference_t0
                segment_state["policy_inference_dts"].append(inference_dt)
                policy_fps = 1.0 / (inference_dt + 1e-9)
                if cfg.env.fps is not None and policy_fps < cfg.env.fps:
                    logging.warning(
                        "[ACTOR] Policy FPS %.1f below required %s at local step %s for primitive '%s'",
                        policy_fps,
                        cfg.env.fps,
                        step_counter[active_primitive],
                        active_primitive,
                    )
            else:
                action = torch.zeros((env.action_dim,), dtype=torch.float32)

            new_transition = env.step(action)
            reward = float(new_transition[TransitionKey.REWARD])
            done = bool(new_transition.get(TransitionKey.DONE, False))
            truncated = bool(new_transition.get(TransitionKey.TRUNCATED, False))
            info = new_transition.get(TransitionKey.INFO, {})

            if policy is not None:
                step_counter.increment(active_primitive)
                segment_state["reward_sum"] += reward
                segment_state["total_steps"] += 1

                is_intervention = _is_intervention(info)
                if is_intervention:
                    segment_state["intervention_steps"] += 1

                next_obs = new_transition[TransitionKey.OBSERVATION]
                policy_next_obs = {key: value for key, value in next_obs.items() if key in policy.config.input_features}
                executed_action = new_transition[TransitionKey.ACTION]

                transition_payload = Transition(
                    state=policy_obs,
                    action=executed_action,
                    reward=reward,
                    next_state=policy_next_obs,
                    done=done,
                    truncated=truncated,
                    complementary_info={
                        "primitive_index": registry.id_to_index[active_primitive],
                        "is_intervention": bool(is_intervention),
                    },
                )
                transition_payload["id"] = active_primitive
                segment_state["pending_transitions"].append(transition_payload)

            transition = new_transition

            if done or truncated:
                if policy is not None:
                    push_transitions_to_transport_queue(
                        transitions=segment_state["pending_transitions"],
                        transitions_queue=transitions_queue,
                    )

                    stats = get_frequency_stats(segment_state["policy_inference_dts"])
                    intervention_rate = 0.0
                    if segment_state["total_steps"] > 0:
                        intervention_rate = segment_state["intervention_steps"] / segment_state["total_steps"]

                    interactions_queue.put(
                        python_object_to_bytes(
                            {
                                "Primitive": active_primitive,
                                "Interaction step": step_counter[active_primitive],
                                "Episodic reward": segment_state["reward_sum"],
                                "Episode intervention": int(segment_state["intervention_steps"] > 0),
                                "Intervention rate": intervention_rate,
                                **stats,
                            }
                        )
                    )

                    step_counter.finish_episode(active_primitive)
                    consumed_updates = apply_parameter_updates_from_queue(
                        policies=policies,
                        parameters_queue=parameters_queue,
                        device=device,
                    )
                    applied_parameter_updates += consumed_updates

                if step_counter.all_finished:
                    break

                transition = env.reset()
                segment_state = reset_segment_state()

            if cfg.env.fps is not None:
                dt = time.perf_counter() - step_start_t
                precise_sleep(max(1 / cfg.env.fps - dt, 0.0))
    finally:
        env.close()

    return {
        "global_step": step_counter.global_step,
        "per_primitive_steps": {primitive_id: step_counter[primitive_id] for primitive_id in registry.adaptive_ids},
        "applied_parameter_updates": applied_parameter_updates,
    }


def _is_intervention(info: dict[str, Any]) -> bool:
    keys = [TeleopEvents.IS_INTERVENTION]
    if hasattr(TeleopEvents.IS_INTERVENTION, "value"):
        keys.append(TeleopEvents.IS_INTERVENTION.value)
    for key in keys:
        if bool(info.get(key, False)):
            return True
    return False


def get_frequency_stats(policy_inference_dts: list[float]) -> dict[str, float]:
    if len(policy_inference_dts) <= 1:
        return {}

    policy_fps = [1.0 / (dt + 1e-9) for dt in policy_inference_dts]
    p90 = quantiles(policy_fps, n=10)[-1]
    return {
        "Policy frequency [Hz]": mean(policy_fps),
        "Policy frequency 90th-p [Hz]": p90,
    }


def establish_learner_connection(
    stub: services_pb2_grpc.LearnerServiceStub,
    shutdown_event: Event,
    attempts: int = 30,
) -> bool:
    for _ in range(attempts):
        if shutdown_event.is_set():
            return False
        try:
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as exc:
            logging.info("[ACTOR] Waiting for learner readiness: %s", exc)
            time.sleep(1.0)
    return False


@lru_cache(maxsize=4)
def learner_service_client(
    host: str,
    port: int,
) -> tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]:
    sanitize_local_grpc_proxy_env(host)
    channel = grpc.insecure_channel(f"{host}:{port}", grpc_channel_options())
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    return stub, channel


def receive_policy(
    shutdown_event: Event,
    parameters_queue: Queue,
    host: str,
    port: int,
    grpc_channel: grpc.Channel | None = None,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
) -> None:
    close_channel = grpc_channel is None
    local_channel = grpc_channel
    if local_channel is None or learner_client is None:
        learner_client, local_channel = learner_service_client(host=host, port=port)

    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )
    except grpc.RpcError as exc:
        logging.error("[ACTOR] receive_policy gRPC error: %s", exc)
    finally:
        if close_channel and local_channel is not None:
            local_channel.close()


def send_transitions(
    shutdown_event: Event,
    transitions_queue: Queue,
    host: str,
    port: int,
    timeout: float,
    grpc_channel: grpc.Channel | None = None,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
) -> None:
    close_channel = grpc_channel is None
    local_channel = grpc_channel
    if local_channel is None or learner_client is None:
        learner_client, local_channel = learner_service_client(host=host, port=port)

    try:
        learner_client.SendTransitions(transitions_stream(shutdown_event, transitions_queue, timeout))
    except grpc.RpcError as exc:
        logging.error("[ACTOR] send_transitions gRPC error: %s", exc)
    finally:
        if close_channel and local_channel is not None:
            local_channel.close()


def send_interactions(
    shutdown_event: Event,
    interactions_queue: Queue,
    host: str,
    port: int,
    timeout: float,
    grpc_channel: grpc.Channel | None = None,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
) -> None:
    close_channel = grpc_channel is None
    local_channel = grpc_channel
    if local_channel is None or learner_client is None:
        learner_client, local_channel = learner_service_client(host=host, port=port)

    try:
        learner_client.SendInteractions(interactions_stream(shutdown_event, interactions_queue, timeout))
    except grpc.RpcError as exc:
        logging.error("[ACTOR] send_interactions gRPC error: %s", exc)
    finally:
        if close_channel and local_channel is not None:
            local_channel.close()


def transitions_stream(shutdown_event: Event, transitions_queue: Queue, timeout: float):
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.Transition,
            log_prefix="[ACTOR] transitions",
        )

    return services_pb2.Empty()


def interactions_stream(shutdown_event: Event, interactions_queue: Queue, timeout: float):
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] interactions",
        )

    return services_pb2.Empty()


def push_transitions_to_transport_queue(transitions: list[dict[str, Any]], transitions_queue: Queue) -> None:
    if not transitions:
        return
    serialized: list[dict[str, Any]] = []
    for transition in transitions:
        tr = move_transition_to_device(transition=transition, device="cpu")
        serialized.append(tr)
    transitions_queue.put(transitions_to_bytes(serialized))


def _use_threads(policy_cfg: SACConfig) -> bool:
    return policy_cfg.concurrency.actor == "threads"


if __name__ == "__main__":
    import experiments
    actor_cli()
