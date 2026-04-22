import logging
from collections import OrderedDict
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.motors import Motor
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from share.motors.dynamixel.dynamixel import TrossenNormMode
from share.robots.viperx.lerobot_robot_viperx.config_mock_viperx import MockViperXConfig

logger = logging.getLogger(__name__)


class MockViperX(Robot):
    """ViperX-compatible mock robot that never touches hardware."""

    config_class = MockViperXConfig
    name = "mock_viperx"

    def __init__(self, config: MockViperXConfig):
        super().__init__(config)
        self.config = config
        self.bus = type(
            "MockViperXBus",
            (),
            {
                "motors": OrderedDict(
                    {
                        "waist": Motor(1, "xm540-w270", TrossenNormMode.RADIANS),
                        "shoulder": Motor(2, "xm540-w270", TrossenNormMode.RADIANS),
                        "shoulder_shadow": Motor(3, "xm540-w270", TrossenNormMode.RADIANS),
                        "elbow": Motor(4, "xm540-w270", TrossenNormMode.RADIANS),
                        "elbow_shadow": Motor(5, "xm540-w270", TrossenNormMode.RADIANS),
                        "forearm_roll": Motor(6, "xm540-w270", TrossenNormMode.RADIANS),
                        "wrist_angle": Motor(7, "xm540-w270", TrossenNormMode.RADIANS),
                        "wrist_rotate": Motor(8, "xm430-w350", TrossenNormMode.RADIANS),
                        "gripper": Motor(9, "xm430-w350", TrossenNormMode.RANGE_0_1),
                    }
                )
            },
        )()
        self._is_connected = False
        self._last_action: dict[str, float] = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors if not motor.endswith("_shadow")}

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            cam_name: (camera.height, camera.width, 3)
            for cam_name, camera in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
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

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._last_action = {
            key: float(value)
            for key, value in action.items()
            if key in self.action_features or key == "finger.pos"
        }
        return dict(self._last_action)
