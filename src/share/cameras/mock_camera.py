from dataclasses import dataclass

import numpy as np
from lerobot.cameras import Camera, CameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


@CameraConfig.register_subclass("mock_camera")
@dataclass
class MockCameraConfig(CameraConfig):
    """Minimal camera config for a black-frame mock camera."""


class MockCamera(Camera):
    """Camera mock that always returns a black RGB image."""

    config_class = MockCameraConfig

    def __init__(self, config: MockCameraConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self._is_connected = True

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._is_connected = False

    @staticmethod
    def find_cameras() -> list[dict[str, str]]:
        return [{"type": "mock_camera", "name": "mock", "id": "mock"}]

    def read(self):
        return self.async_read()

    def async_read(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
