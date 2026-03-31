import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
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
    "[o] set origin from current pose and reset bounds, "
    "[r] reset bounds in current frame, "
    "[p] print current origin/bounds, "
    "[s] save config, "
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

        self._counts = {name: 0 for name in ("set_origin", "reset_bounds", "print_status", "save", "quit")}
        self._lock = threading.Lock()
        self._mapping = {
            "o": "set_origin",
            "r": "reset_bounds",
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


class WorkspaceConstraintDesigner:
    """Track and persist per-primitive frame origins and workspace bounds."""

    def __init__(self, mp_net: ManipulationPrimitiveNet, output_path: Path):
        self.mp_net = mp_net
        self.output_path = Path(output_path)
        self._tracked_primitives: set[str] = set()

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

    def set_origin_from_current_pose(self, primitive_name: str | None = None) -> None:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        primitive = self.mp_net.config.primitives[primitive_name]
        if not primitive.is_adaptive:
            logging.info("[%s] Ignoring origin set request because the primitive is not adaptive.", primitive_name)
            return

        env = self.mp_net._envs[primitive_name]
        current_world = self.current_world_pose_by_robot(primitive_name)
        for robot_name, frame in _task_frames_for_primitive(primitive).items():
            origin = [float(v) for v in current_world[robot_name]]
            frame.origin = list(origin)
            frame.min_pose = [0.0] * len(frame.target)
            frame.max_pose = [0.0] * len(frame.target)
            env.task_frame[robot_name].origin = list(origin)
            env.task_frame[robot_name].min_pose = [0.0] * len(frame.target)
            env.task_frame[robot_name].max_pose = [0.0] * len(frame.target)

        env.apply_task_frames()
        self._tracked_primitives.add(primitive_name)
        logging.info("[%s] Set frame origin from current pose and reset bounds.", primitive_name)
        self.log_status(primitive_name)

    def reset_bounds(self, primitive_name: str | None = None) -> None:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        primitive = self.mp_net.config.primitives[primitive_name]
        if primitive_name not in self._tracked_primitives:
            logging.info("[%s] Bounds reset ignored because no origin has been set yet.", primitive_name)
            return

        env = self.mp_net._envs[primitive_name]
        current_world = self.current_world_pose_by_robot(primitive_name)
        for robot_name, frame in _task_frames_for_primitive(primitive).items():
            current_in_frame = world_pose_to_task_pose(current_world[robot_name], frame.origin)
            frame.min_pose = [float(v) for v in current_in_frame]
            frame.max_pose = [float(v) for v in current_in_frame]
            env.task_frame[robot_name].min_pose = list(frame.min_pose)
            env.task_frame[robot_name].max_pose = list(frame.max_pose)

        env.apply_task_frames()
        logging.info("[%s] Reset workspace bounds in the current frame.", primitive_name)
        self.log_status(primitive_name)

    def update_bounds(self, primitive_name: str | None = None) -> None:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        if primitive_name not in self._tracked_primitives:
            return

        primitive = self.mp_net.config.primitives[primitive_name]
        env = self.mp_net._envs[primitive_name]
        current_world = self.current_world_pose_by_robot(primitive_name)
        for robot_name, frame in _task_frames_for_primitive(primitive).items():
            current_in_frame = world_pose_to_task_pose(current_world[robot_name], frame.origin)
            if frame.min_pose is None or len(frame.min_pose) != len(current_in_frame):
                frame.min_pose = [float(v) for v in current_in_frame]
            else:
                frame.min_pose = [
                    min(float(lower), float(value))
                    for lower, value in zip(frame.min_pose, current_in_frame, strict=True)
                ]
            if frame.max_pose is None or len(frame.max_pose) != len(current_in_frame):
                frame.max_pose = [float(v) for v in current_in_frame]
            else:
                frame.max_pose = [
                    max(float(upper), float(value))
                    for upper, value in zip(frame.max_pose, current_in_frame, strict=True)
                ]

            env.task_frame[robot_name].min_pose = list(frame.min_pose)
            env.task_frame[robot_name].max_pose = list(frame.max_pose)

    def save(self) -> None:
        save_mpnet_config(self.mp_net.config, self.output_path)
        logging.info("Saved calibrated MP-Net config to %s", self.output_path)

    def log_status(self, primitive_name: str | None = None) -> None:
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        primitive = self.mp_net.config.primitives[primitive_name]
        summary = {
            robot_name: {
                "origin": frame.origin,
                "min_pose": frame.min_pose,
                "max_pose": frame.max_pose,
            }
            for robot_name, frame in _task_frames_for_primitive(primitive).items()
        }
        logging.info("[%s] %s", primitive_name, pformat(summary))


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

        action = torch.zeros(mp_net.action_dim, dtype=torch.float32)
        previous_primitive = mp_net.active_primitive
        transition = mp_net.step(action)
        designer.update_bounds(previous_primitive)

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
