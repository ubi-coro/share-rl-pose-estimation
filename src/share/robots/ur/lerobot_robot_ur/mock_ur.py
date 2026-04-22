import logging
from dataclasses import asdict
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.processor.hil_processor import GRIPPER_KEY
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, TASK_FRAME_AXIS_NAMES, TaskFrame
from share.robots.ur.lerobot_robot_ur.config_mock_ur import MockURConfig
from share.robots.ur.lerobot_robot_ur.controller import TaskFrameCommand

logger = logging.getLogger(__name__)


class MockUR(Robot):
    """UR-compatible mock robot that never touches hardware."""

    config_class = MockURConfig
    name = "mock_ur"

    joint_names = state_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def __init__(self, config: MockURConfig):
        super().__init__(config)
        self.config = config
        self.task_frame = TaskFrameCommand(controller_overrides=self._default_controller_overrides())
        self._active_control_space: ControlSpace | None = None
        self._is_connected = False
        self._last_action: dict[str, Any] = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        ft = {}
        for axis_name in TASK_FRAME_AXIS_NAMES:
            ft[f"{axis_name}.ee_pos"] = float
            ft[f"{axis_name}.ee_vel"] = float
            ft[f"{axis_name}.ee_wrench"] = float

        for joint_name in self.joint_names:
            ft[f"{joint_name}.pos"] = float
            ft[f"{joint_name}.vel"] = float

        if self.config.use_gripper:
            ft["gripper.pos"] = float

        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            cam_name: (camera.height, camera.width, 3)
            for cam_name, camera in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        ft = {key: float for key in self.task_frame.to_robot_action()}
        if self.config.use_gripper:
            ft["gripper.pos"] = float
        return ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._is_connected = True
        logger.info(f"{self} connected in mock mode.")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {key: 0.0 for key in self._motors_ft}
        for cam_name, shape in self._cameras_ft.items():
            obs_dict[cam_name] = np.zeros(shape, dtype=np.uint8)
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action_space = self._space_from_action(action)
        if action_space is not None:
            self._ensure_control_space(action_space)
            if self.task_frame.space != action_space:
                self.task_frame.space = action_space
                self.task_frame.origin = None if action_space == ControlSpace.JOINT else [0.0] * 6
                if action_space == ControlSpace.JOINT:
                    self.task_frame.control_mode = [ControlMode.POS] * len(self.task_frame.target)

        if self.task_frame.space == ControlSpace.JOINT:
            for index, joint_name in enumerate(self.joint_names):
                canonical_key = f"joint_{index + 1}.pos"
                named_key = f"{joint_name}.pos"
                if canonical_key in action:
                    self.task_frame.target[index] = action[canonical_key]
                    self.task_frame.control_mode[index] = ControlMode.POS
                elif named_key in action:
                    self.task_frame.target[index] = action[named_key]
                    self.task_frame.control_mode[index] = ControlMode.POS
        else:
            for index, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
                if f"{axis_name}.ee_pos" in action:
                    self.task_frame.target[index] = action[f"{axis_name}.ee_pos"]
                    self.task_frame.control_mode[index] = ControlMode.POS
                elif f"{axis_name}.ee_vel" in action:
                    self.task_frame.target[index] = action[f"{axis_name}.ee_vel"]
                    self.task_frame.control_mode[index] = ControlMode.VEL
                elif f"{axis_name}.ee_wrench" in action:
                    self.task_frame.target[index] = action[f"{axis_name}.ee_wrench"]
                    self.task_frame.control_mode[index] = ControlMode.WRENCH

        if self.config.use_gripper and f"{GRIPPER_KEY}.pos" in action:
            self.send_gripper_action(float(action[f"{GRIPPER_KEY}.pos"]))

        self._last_action = dict(action)
        return dict(action)

    def send_gripper_action(self, gripper_action: float) -> None:
        return None

    def set_task_frame(self, new_task_frame: TaskFrameCommand | TaskFrame) -> None:
        new_task_frame = self._task_frame_command_from_frame(new_task_frame)
        self._ensure_control_space(new_task_frame.space)
        self.task_frame = new_task_frame

    def _task_frame_command_from_frame(self, frame: TaskFrameCommand | TaskFrame) -> TaskFrameCommand:
        command = TaskFrameCommand(**asdict(frame))
        command.controller_overrides = self._merged_controller_overrides(command.controller_overrides)
        return command

    def _default_controller_overrides(self) -> dict[str, Any]:
        return {
            "kp": list(self.config.kp),
            "kd": list(self.config.kd),
            "wrench_limits": list(self.config.wrench_limits),
            "compliance_adaptive_limit_enable": list(self.config.compliance_adaptive_limit_enable),
            "compliance_reference_limit_enable": list(self.config.compliance_reference_limit_enable),
            "compliance_desired_wrench": list(self.config.compliance_desired_wrench),
            "compliance_adaptive_limit_min": list(self.config.compliance_adaptive_limit_min),
        }

    def _merged_controller_overrides(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        unknown = set(overrides or {}) - TaskFrameCommand.SUPPORTED_CONTROLLER_OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"Unsupported UR task-frame controller overrides: {', '.join(sorted(unknown))}")
        merged = dict(self.task_frame.controller_overrides or self._default_controller_overrides())
        if not overrides:
            return merged
        merged.update(overrides)
        return merged

    def _ensure_control_space(self, space: ControlSpace | int) -> ControlSpace:
        resolved = ControlSpace(int(space))
        if self._active_control_space is None:
            self._active_control_space = resolved
            return resolved
        if resolved != self._active_control_space:
            raise ValueError("UR robot does not support switching between task-space and joint-space control")
        return resolved

    @staticmethod
    def _space_from_action(action: dict[str, Any]) -> ControlSpace | None:
        has_task_keys = any(
            key.endswith(".ee_pos") or key.endswith(".ee_vel") or key.endswith(".ee_wrench")
            for key in action
        )
        has_joint_keys = any(
            key.endswith(".pos") and ".ee_" not in key and key != f"{GRIPPER_KEY}.pos"
            for key in action
        )

        if has_task_keys and has_joint_keys:
            raise ValueError("UR robot actions cannot mix task-space and joint-space keys")
        if has_task_keys:
            return ControlSpace.TASK
        if has_joint_keys:
            return ControlSpace.JOINT
        return None
