from __future__ import annotations

import copy
import datetime as dt
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.transport.utils import bytes_to_state_dict
from lerobot.utils.transition import move_state_dict_to_device
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)


@dataclass(slots=True)
class AdaptivePrimitiveRegistry:
    adaptive_ids: list[str]
    id_to_index: dict[str, int]
    policy_cfgs: dict[str, SACConfig]
    online_step_budgets: dict[str, int]
    actor_learner_policy_cfg: SACConfig


class PrimitiveBudgetCounter:
    """Track per-primitive interaction counts against online-step budgets."""

    def __init__(self, budgets: dict[str, int]):
        self._budgets = dict(budgets)
        self._counts = {primitive_id: 0 for primitive_id in budgets}
        self._last_finish_counts = {primitive_id: 0 for primitive_id in budgets}

    def __getitem__(self, primitive_id: str) -> int:
        return self._counts[primitive_id]

    def increment(self, primitive_id: str, n: int = 1) -> None:
        if primitive_id in self._counts:
            self._counts[primitive_id] += int(n)

    def is_finished(self, primitive_id: str) -> bool:
        return self._counts.get(primitive_id, 0) >= self._budgets.get(primitive_id, 0)

    @property
    def all_finished(self) -> bool:
        return all(self.is_finished(primitive_id) for primitive_id in self._counts)

    def finish_episode(self, primitive_id: str) -> None:
        if primitive_id in self._counts:
            self._last_finish_counts[primitive_id] = self._counts[primitive_id]

    def episode_length(self, primitive_id: str) -> int:
        return self._counts.get(primitive_id, 0) - self._last_finish_counts.get(primitive_id, 0)

    @property
    def global_step(self) -> int:
        return sum(self._counts.values())


def _policy_for_primitive(
    primitive_policy: Any | None,
    fallback_policy: SACConfig | None,
    primitive_overwrites: dict[str, Any],
) -> SACConfig:
    if primitive_policy is not None:
        if not isinstance(primitive_policy, SACConfig):
            raise TypeError(
                "MP-Net RL server currently requires SAC policies for adaptive primitives. "
                f"Got '{type(primitive_policy).__name__}'."
            )
        policy_cfg = copy.deepcopy(primitive_policy)
    elif fallback_policy is not None:
        policy_cfg = copy.deepcopy(fallback_policy)
    else:
        raise ValueError(
            "Missing SAC policy for adaptive primitive. Configure primitive.policy "
            "or provide a top-level fallback policy."
        )

    for key, value in primitive_overwrites.items():
        setattr(policy_cfg, key, value)
    return policy_cfg


def build_adaptive_registry(
    env_cfg: ManipulationPrimitiveNetConfig,
    fallback_policy: SACConfig | None,
) -> AdaptivePrimitiveRegistry:
    adaptive_ids: list[str] = []
    policy_cfgs: dict[str, SACConfig] = {}
    budgets: dict[str, int] = {}

    for primitive_id, primitive in env_cfg.primitives.items():
        if primitive_id == env_cfg.reset_primitive:
            continue
        if not primitive.is_adaptive:
            continue

        policy_cfg = _policy_for_primitive(
            primitive_policy=primitive.policy,
            fallback_policy=fallback_policy,
            primitive_overwrites=primitive.policy_overwrites,
        )
        adaptive_ids.append(primitive_id)
        policy_cfgs[primitive_id] = policy_cfg
        budgets[primitive_id] = int(policy_cfg.online_steps)

    if not adaptive_ids:
        raise ValueError(
            "No adaptive primitives with SAC policy config found (reset_primitive is excluded by default)."
        )

    id_to_index = {primitive_id: index for index, primitive_id in enumerate(adaptive_ids)}
    actor_learner_policy_cfg = policy_cfgs[adaptive_ids[0]]
    return AdaptivePrimitiveRegistry(
        adaptive_ids=adaptive_ids,
        id_to_index=id_to_index,
        policy_cfgs=policy_cfgs,
        online_step_budgets=budgets,
        actor_learner_policy_cfg=actor_learner_policy_cfg,
    )


def ensure_identity_features_map(primitive: Any) -> None:
    if not getattr(primitive, "features", None):
        return
    if getattr(primitive, "features_map", None) is None:
        primitive.features_map = {}
    for key in primitive.features:
        primitive.features_map.setdefault(key, key)


def make_policies_for_registry(
    env_cfg: ManipulationPrimitiveNetConfig,
    registry: AdaptivePrimitiveRegistry,
    *,
    train_mode: bool,
) -> dict[str, SACPolicy]:
    policies: dict[str, SACPolicy] = {}
    for primitive_id in registry.adaptive_ids:
        primitive = env_cfg.primitives[primitive_id]
        ensure_identity_features_map(primitive)
        policy_cfg = copy.deepcopy(registry.policy_cfgs[primitive_id])
        policy = make_policy(cfg=policy_cfg, env_cfg=primitive)
        if train_mode:
            policy = policy.train()
        else:
            policy = policy.eval()
        policies[primitive_id] = policy
    return policies


def apply_parameter_updates_from_queue(
    *,
    policies: dict[str, SACPolicy],
    parameters_queue: Any,
    device: str,
) -> int:
    latest_payload_per_primitive: dict[str, dict[str, Any]] = {}
    consumed = 0

    while True:
        try:
            payload = parameters_queue.get_nowait()
        except queue.Empty:
            break

        consumed += 1
        decoded = bytes_to_state_dict(payload)
        primitive_id = decoded.get("primitive_id")
        if primitive_id is None or primitive_id not in policies:
            continue
        latest_payload_per_primitive[str(primitive_id)] = decoded

    for primitive_id, payload in latest_payload_per_primitive.items():
        actor_state = payload.get("policy")
        if actor_state is not None:
            actor_state = move_state_dict_to_device(actor_state, device=device)
            policies[primitive_id].actor.load_state_dict(actor_state)

        if "discrete_critic" in payload and hasattr(policies[primitive_id], "discrete_critic"):
            discrete_state = move_state_dict_to_device(payload["discrete_critic"], device=device)
            policies[primitive_id].discrete_critic.load_state_dict(discrete_state)

    return consumed


def sanitize_local_grpc_proxy_env(host: str) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return

    for key in ["http_proxy", "https_proxy", "grpc_proxy", "HTTP_PROXY", "HTTPS_PROXY", "GRPC_PROXY"]:
        os.environ.pop(key, None)

    for key in ["no_proxy", "NO_PROXY", "no_grpc_proxy", "NO_GRPC_PROXY"]:
        current = os.environ.get(key, "")
        items = [x.strip() for x in current.split(",") if x.strip()]
        for needed in ["localhost", "127.0.0.1", "::1"]:
            if needed not in items:
                items.append(needed)
        os.environ[key] = ",".join(items)
