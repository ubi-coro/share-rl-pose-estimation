import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
import sys
from typing import Any

import torch
from lerobot.configs import parser
from lerobot.processor import TransitionKey
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

from share.configs.design_constraints import DesignConstraintsConfig
from share.envs.manipulation_primitive.task_frame import TaskFrame
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.utils.constants import DEFAULT_ROBOT_NAME
from share.utils.transformation_utils import get_robot_pose_from_observation, world_pose_to_task_pose
from share.workspace.mpnet import save_mpnet_config

init_logging()

HOTKEY_HELP = (
    "Hotkeys: "
    "[o] set origin from current pose and reset tracked bounds, "
    "[r] reset tracked bounds at the current pose, "
    "[e] toggle live enforcement of tracked bounds, "
    "[p] print current origin/tracked bounds, "
    "[s] save tracked bounds into the config, "
    "[q] save and quit"
)


def _close_mp_net(mp_net: ManipulationPrimitiveNet) -> None:
    for env in getattr(mp_net, "_envs", {}).values():
        close = getattr(env, "close", None)
        if callable(close):
            close()


class HotkeyController:
    """Small one-shot hotkey listener used by the calibration loop."""

    def __init__(self):
        from pynput import keyboard

        self._counts = {
            name: 0
            for name in ("set_origin", "reset_bounds", "toggle_enforcement", "print_status", "save", "quit")
        }
        self._lock = threading.Lock()
        self._mapping = {
            "o": "set_origin",
            "r": "reset_bounds",
            "e": "toggle_enforcement",
            "p": "print_status",
            "s": "save",
            "q": "quit",
        }

        def on_press(key):
            try:
                if key.char is None:
                    return
                event_name = self._mapping.get(key.char.lower())
                if event_name is None:
                    return
                with self._lock:
                    self._counts[event_name] += 1
            except Exception:
                return

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

    def consume(self, event_name: str) -> bool:
        with self._lock:
            count = self._counts.get(event_name, 0)
            if count <= 0:
                return False
            self._counts[event_name] = count - 1
            return True

    def close(self) -> None:
        self._listener.stop()


def _task_frames_for_primitive(primitive: Any) -> dict[str, TaskFrame]:
    if isinstance(primitive.task_frame, dict):
        return primitive.task_frame
    return {DEFAULT_ROBOT_NAME: primitive.task_frame}


@dataclass(slots=True)
class TrackedFrameBounds:
    """Per-primitive bounds tracked by the calibration tool."""

    origin: list[float] | None = None
    min_pose: list[float] | None = None
    max_pose: list[float] | None = None
    enforce_live: bool = False


class WorkspaceConstraintDesigner:
    """Track and persist per-primitive frame origins and workspace bounds."""

    def __init__(self, mp_net: ManipulationPrimitiveNet, output_path: Path):
        self.mp_net = mp_net
        self.output_path = Path(output_path)
        self._tracked_bounds: dict[str, dict[str, TrackedFrameBounds]] = {}

    def _tracked_record(
        self,
        primitive_name: str,
        robot_name: str,
        *,
        create: bool = False,
    ) -> TrackedFrameBounds | None:
        records = self._tracked_bounds.get(primitive_name)
        if records is None:
            if not create:
                return None
            records = {}
            self._tracked_bounds[primitive_name] = records

        record = records.get(robot_name)
        if record is None and create:
            record = TrackedFrameBounds()
            records[robot_name] = record
        return record

    @staticmethod
    def _free_bounds(width: int) -> tuple[list[float], list[float]]:
        return [float("-inf")] * width, [float("inf")] * width

    def _apply_live_bounds(
        self,
        primitive_name: str,
        *,
        reapply_task_frame: bool = True,
    ) -> None:
        primitive = self.mp_net.config.primitives[primitive_name]
        env = self.mp_net._envs[primitive_name]
        for robot_name, frame in _task_frames_for_primitive(primitive).items():
            env_frame = getattr(env, "task_frame", {}).get(robot_name, frame)
            record = self._tracked_record(primitive_name, robot_name)
            if (
                record is not None
                and record.enforce_live
                and record.min_pose is not None
                and record.max_pose is not None
            ):
                min_pose = [float(value) for value in record.min_pose]
                max_pose = [float(value) for value in record.max_pose]
            else:
                min_pose, max_pose = self._free_bounds(len(frame.target))

            frame.min_pose = list(min_pose)
            frame.max_pose = list(max_pose)
            env_frame.min_pose = list(min_pose)
            env_frame.max_pose = list(max_pose)

        if reapply_task_frame:
            apply_task_frames = getattr(env, "apply_task_frames", None)
            if callable(apply_task_frames):
                apply_task_frames()

    def current_world_pose_by_robot(self, primitive_name: str | None = None) -> dict[str, list[float]]:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        env = self.mp_net._envs[primitive_name]
        primitive = self.mp_net.config.primitives[primitive_name]
        observation = env._get_observation()
        poses: dict[str, list[float]] = {}
        for robot_name in _task_frames_for_primitive(primitive):
            poses[robot_name] = get_robot_pose_from_observation(observation, robot_name)
        return poses

    def current_pose_in_frame_by_robot(self, primitive_name: str | None = None) -> dict[str, list[float]]:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        primitive = self.mp_net.config.primitives[primitive_name]
        current_world = self.current_world_pose_by_robot(primitive_name)
        return {
            robot_name: world_pose_to_task_pose(current_world[robot_name], frame.origin)
            for robot_name, frame in _task_frames_for_primitive(primitive).items()
        }

    def _reset_runtime_state_for_primitive(self, primitive_name: str) -> None:
        env = self.mp_net._envs[primitive_name]
        env_processor = getattr(self.mp_net, "_env_processors", {}).get(primitive_name)
        action_processor = getattr(self.mp_net, "_action_processors", {}).get(primitive_name)
        if env_processor is not None:
            env_processor.reset()
        if action_processor is not None:
            action_processor.reset()
        reset_runtime_state = getattr(env, "reset_runtime_state", None)
        if callable(reset_runtime_state):
            reset_runtime_state()
        apply_task_frames = getattr(env, "apply_task_frames", None)
        if callable(apply_task_frames):
            apply_task_frames()

    def set_origin_from_current_pose(self, primitive_name: str | None = None) -> None:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        primitive = self.mp_net.config.primitives[primitive_name]
        env = self.mp_net._envs[primitive_name]
        if not primitive.is_adaptive:
            logging.info("[%s] Ignoring origin set request because the primitive is not adaptive.", primitive_name)
            return

        current_world = self.current_world_pose_by_robot(primitive_name)
        for robot_name, frame in _task_frames_for_primitive(primitive).items():
            env_frame = getattr(env, "task_frame", {}).get(robot_name, frame)
            origin = [float(v) for v in current_world[robot_name]]
            record = self._tracked_record(primitive_name, robot_name, create=True)
            record.origin = list(origin)
            record.min_pose = [0.0] * len(frame.target)
            record.max_pose = [0.0] * len(frame.target)
            frame.origin = list(origin)
            frame.target = [0.0] * len(frame.target)
            env_frame.origin = list(origin)
            env_frame.target = [0.0] * len(frame.target)

        self._apply_live_bounds(primitive_name, reapply_task_frame=False)
        self._reset_runtime_state_for_primitive(primitive_name)
        logging.info("[%s] Set frame origin from current pose and reset tracked bounds.", primitive_name)
        self.log_status(primitive_name)

    def reset_bounds(self, primitive_name: str | None = None) -> None:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        primitive = self.mp_net.config.primitives[primitive_name]
        reset_any = False
        current_world = self.current_world_pose_by_robot(primitive_name)
        for robot_name, frame in _task_frames_for_primitive(primitive).items():
            record = self._tracked_record(primitive_name, robot_name)
            if record is None or record.origin is None:
                continue
            current_in_frame = world_pose_to_task_pose(current_world[robot_name], record.origin)
            record.min_pose = [float(v) for v in current_in_frame]
            record.max_pose = [float(v) for v in current_in_frame]
            reset_any = True

        if not reset_any:
            logging.info("[%s] Bounds reset ignored because no origin has been set yet.", primitive_name)
            return

        self._apply_live_bounds(primitive_name)
        logging.info("[%s] Reset tracked workspace bounds at the current pose.", primitive_name)
        self.log_status(primitive_name)

    def update_bounds(self, primitive_name: str | None = None) -> None:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive

        primitive = self.mp_net.config.primitives[primitive_name]
        current_world = self.current_world_pose_by_robot(primitive_name)
        for robot_name, frame in _task_frames_for_primitive(primitive).items():
            record = self._tracked_record(primitive_name, robot_name)
            if record is None or record.origin is None:
                continue

            current_in_frame = world_pose_to_task_pose(current_world[robot_name], record.origin)
            if record.min_pose is None or len(record.min_pose) != len(current_in_frame):
                record.min_pose = [float(v) for v in current_in_frame]
            else:
                record.min_pose = [
                    min(float(lower), float(value))
                    for lower, value in zip(record.min_pose, current_in_frame, strict=True)
                ]
            if record.max_pose is None or len(record.max_pose) != len(current_in_frame):
                record.max_pose = [float(v) for v in current_in_frame]
            else:
                record.max_pose = [
                    max(float(upper), float(value))
                    for upper, value in zip(record.max_pose, current_in_frame, strict=True)
                ]

            if record.enforce_live:
                env = self.mp_net._envs[primitive_name]
                env_frame = getattr(env, "task_frame", {}).get(robot_name, frame)
                frame.min_pose = [float(value) for value in record.min_pose]
                frame.max_pose = [float(value) for value in record.max_pose]
                env_frame.min_pose = [float(value) for value in record.min_pose]
                env_frame.max_pose = [float(value) for value in record.max_pose]

    def toggle_live_enforcement(self, primitive_name: str | None = None) -> None:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive

        primitive = self.mp_net.config.primitives[primitive_name]
        tracked_records: list[TrackedFrameBounds] = []
        for robot_name in _task_frames_for_primitive(primitive):
            record = self._tracked_record(primitive_name, robot_name)
            if record is not None and record.origin is not None:
                tracked_records.append(record)

        if not tracked_records:
            logging.info("[%s] Live bound toggle ignored because no origin has been set yet.", primitive_name)
            return

        enable_live = not all(record.enforce_live for record in tracked_records)
        for record in tracked_records:
            record.enforce_live = enable_live

        self._apply_live_bounds(primitive_name)
        mode = "enabled" if enable_live else "disabled"
        logging.info("[%s] Live enforcement of tracked bounds %s.", primitive_name, mode)
        self.log_status(primitive_name)

    def save(self) -> None:
        snapshots: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
        for primitive_name, primitive in self.mp_net.config.primitives.items():
            for robot_name, frame in _task_frames_for_primitive(primitive).items():
                snapshots[(primitive_name, robot_name)] = (list(frame.min_pose), list(frame.max_pose))

        try:
            for primitive_name, records in self._tracked_bounds.items():
                primitive = self.mp_net.config.primitives[primitive_name]
                frames = _task_frames_for_primitive(primitive)
                for robot_name, record in records.items():
                    frame = frames[robot_name]
                    if record.origin is not None:
                        frame.origin = [float(value) for value in record.origin]
                    if record.min_pose is not None:
                        frame.min_pose = [float(value) for value in record.min_pose]
                    if record.max_pose is not None:
                        frame.max_pose = [float(value) for value in record.max_pose]

            save_mpnet_config(self.mp_net.config, self.output_path)
            logging.info("Saved calibrated MP-Net config to %s", self.output_path)
        finally:
            for primitive_name, primitive in self.mp_net.config.primitives.items():
                env = self.mp_net._envs[primitive_name]
                for robot_name, frame in _task_frames_for_primitive(primitive).items():
                    env_frame = getattr(env, "task_frame", {}).get(robot_name, frame)
                    live_min_pose, live_max_pose = snapshots[(primitive_name, robot_name)]
                    frame.min_pose = list(live_min_pose)
                    frame.max_pose = list(live_max_pose)
                    env_frame.min_pose = list(live_min_pose)
                    env_frame.max_pose = list(live_max_pose)

    def status_summary(self, primitive_name: str | None = None) -> dict[str, dict[str, Any]]:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        primitive = self.mp_net.config.primitives[primitive_name]
        summary: dict[str, dict[str, Any]] = {}
        for robot_name, frame in _task_frames_for_primitive(primitive).items():
            record = self._tracked_record(primitive_name, robot_name)
            summary[robot_name] = {
                "origin": None if frame.origin is None else [float(value) for value in frame.origin],
                "tracked_min_pose": None if record is None or record.min_pose is None else [float(value) for value in record.min_pose],
                "tracked_max_pose": None if record is None or record.max_pose is None else [float(value) for value in record.max_pose],
                "live_bounds_enforced": bool(record is not None and record.enforce_live),
            }
        return summary

    def log_status(self, primitive_name: str | None = None) -> None:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        summary = self.status_summary(primitive_name)
        logging.info("[%s] %s", primitive_name, pformat(summary))

    def print_live_pose(self, primitive_name: str | None = None) -> None:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        pose_in_frame = self.current_pose_in_frame_by_robot(primitive_name)
        compact = " | ".join(
            f"{robot}: {[round(float(value), 4) for value in pose]}"
            for robot, pose in pose_in_frame.items()
        )
        sys.stdout.write(f"\r[{primitive_name}] pose_in_frame {compact}   ")
        sys.stdout.flush()


def calibration_loop(
    mp_net: ManipulationPrimitiveNet,
    designer: WorkspaceConstraintDesigner,
    hotkeys: HotkeyController,
    autosave_on_primitive_change: bool = True,
):
    transition = mp_net.reset()
    logging.info(HOTKEY_HELP)
    logging.info("Entered primitive '%s'.", mp_net.active_primitive)

    while True:
        start_loop_t = time.perf_counter()

        if hotkeys.consume("quit"):
            designer.save()
            return transition.get(TransitionKey.INFO, {})

        if hotkeys.consume("save"):
            designer.save()

        if hotkeys.consume("print_status"):
            designer.log_status()

        if hotkeys.consume("set_origin"):
            designer.set_origin_from_current_pose()

        if hotkeys.consume("reset_bounds"):
            designer.reset_bounds()

        if hotkeys.consume("toggle_enforcement"):
            designer.toggle_live_enforcement()

        action = torch.zeros(mp_net.action_dim, dtype=torch.float32)
        previous_primitive = mp_net.active_primitive
        transition = mp_net.step(action)
        designer.update_bounds(previous_primitive)
        designer.print_live_pose(previous_primitive)

        info = transition.get(TransitionKey.INFO, {})
        next_primitive = info.get("transition_to", previous_primitive)
        primitive_changed = next_primitive != previous_primitive

        if primitive_changed:
            designer.log_status(previous_primitive)
            if autosave_on_primitive_change:
                designer.save()

            transition = mp_net.reset()
            logging.info("Entered primitive '%s'.", mp_net.active_primitive)

        elif getattr(mp_net, "_needs_full_reset", False):
            designer.log_status(previous_primitive)
            if autosave_on_primitive_change:
                designer.save()

            transition = mp_net.reset()
            logging.info("Restarted episode at primitive '%s'.", mp_net.active_primitive)

        dt_load = time.perf_counter() - start_loop_t
        precise_sleep(1 / mp_net.config.fps - dt_load)


@parser.wrap()
def design_constraints(cfg: DesignConstraintsConfig):
    logging.info(pformat(asdict(cfg)))
    mp_net = ManipulationPrimitiveNet(cfg.env)
    hotkeys = HotkeyController()
    designer = WorkspaceConstraintDesigner(mp_net=mp_net, output_path=cfg.output_path)
    logging.info(HOTKEY_HELP)

    try:
        log_say("Start calibration", play_sounds=cfg.play_sounds, blocking=False)
        calibration_loop(
            mp_net=mp_net,
            designer=designer,
            hotkeys=hotkeys,
            autosave_on_primitive_change=cfg.autosave_on_primitive_change,
        )
    finally:
        hotkeys.close()
        _close_mp_net(mp_net)
        log_say("Stop calibration", play_sounds=cfg.play_sounds, blocking=True)


if __name__ == "__main__":
    import experiments

    design_constraints()
