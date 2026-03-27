"""Shared mock helpers for MP-Net examples."""

from __future__ import annotations

from typing import Any

import numpy as np

from share.envs.manipulation_primitive.task_frame import TASK_FRAME_AXIS_NAMES
from share.utils.mock_utils import MockRobot, MockTeleoperator


ROTATION_ALIASES = {"rx": "wx", "ry": "wy", "rz": "wz"}


class DeterministicTaskFrameRobot(MockRobot):
    """Mock task-frame robot that mirrors commanded EE targets into observations."""

    def __init__(self, name: str, pose: list[float] | None = None):
        super().__init__(name=name, is_task_frame=True)
        self.current_pose = np.array([0.0] * 6 if pose is None else pose, dtype=float)

    @property
    def observation_features(self) -> dict:
        features = {f"joint_{i + 1}.pos": float for i in range(6)}
        for axis_name in ["x", "y", "z", "wx", "wy", "wz"]:
            features[f"{axis_name}.ee_pos"] = float
        return features

    @property
    def action_features(self) -> dict:
        features = super().action_features.copy()
        for axis_name in ["x", "y", "z", "rx", "ry", "rz"]:
            features[f"{axis_name}.ee_pos"] = float
        return features

    def get_observation(self) -> dict[str, Any]:
        obs = {f"joint_{i + 1}.pos": float(self.current_joints[i]) for i in range(6)}
        for axis, axis_name in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
            obs[f"{axis_name}.ee_pos"] = float(self.current_pose[axis])
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        super().send_action(action)
        for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
            for candidate in (axis_name, ROTATION_ALIASES.get(axis_name)):
                if candidate is None:
                    continue
                key = f"{candidate}.ee_pos"
                if key in action:
                    self.current_pose[axis] = float(action[key])
                    break
        return action


def make_mock_connect(
    robot_poses: dict[str, list[float]],
):
    """Create a ``ManipulationPrimitiveNet.connect`` implementation for examples."""

    def _connect():
        robots = {
            name: DeterministicTaskFrameRobot(name=name, pose=pose)
            for name, pose in robot_poses.items()
        }
        teleops = {
            name: MockTeleoperator(name=name, is_delta=True)
            for name in robot_poses
        }
        return robots, teleops, {}

    return _connect
