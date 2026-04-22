from __future__ import annotations

from typing import Any

import numpy as np
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from numpy._typing import NDArray

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


class RealSenseDepthCamera(RealSenseCamera):
    """RealSense camera variant that exposes RGB and depth in async reads."""

    def __init__(self, config: RealSenseCameraConfig):
        """
        Initializes the RealSenseCamera instance.

        Args:
            config: The configuration settings for the camera.
        """

        super().__init__(config)
        self.use_depth = True
        self.depth_scale: float | None = None
        self.camera_intrinsics: NDArray[np.float64] | None = None

    def connect(self, warmup: bool = True) -> None:
        super().connect(warmup=warmup)
        self._update_depth_scale()
        self._update_camera_intrinsics()

    def disconnect(self) -> None:
        super().disconnect()
        self.depth_scale = None
        self.camera_intrinsics = None

    def _update_depth_scale(self) -> None:
        if self.rs_profile is None:
            raise RuntimeError(f"{self} is not connected.")

        depth_sensor = self.rs_profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

    def get_depth_scale(self) -> float:
        if self.depth_scale is None:
            raise RuntimeError(f"{self} depth scale is not available. Call connect() first.")
        return self.depth_scale

    def _update_camera_intrinsics(self) -> None:
        if self.rs_profile is None:
            raise RuntimeError(f"{self} is not connected.")
        if rs is None:
            raise RuntimeError("pyrealsense2 is not available.")

        intr = self.rs_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_intrinsics = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def get_camera_intrinsics(self) -> NDArray[np.float64]:
        if self.camera_intrinsics is None:
            raise RuntimeError(f"{self} intrinsics are not available. Call connect() first.")
        return self.camera_intrinsics.copy()

    def depth_to_meters(self, depth_frame: NDArray[Any]) -> NDArray[np.float32]:
        depth_scale = self.get_depth_scale()
        return depth_frame.astype(np.float32) * depth_scale

    def async_read_depth(self, timeout_ms: float = 200, in_meters: bool = False) -> NDArray[Any]:
        if not self.use_depth:
            raise RuntimeError(f"{self} depth stream is not enabled. Set `use_depth=True` in RealSenseCameraConfig.")

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            depth_frame = self.latest_depth_frame
            self.new_frame_event.clear()

        if depth_frame is None:
            raise RuntimeError(f"Internal error: Event set but no depth frame available for {self}.")

        if in_meters:
            depth_frame = self.depth_to_meters(depth_frame)

        return depth_frame

    def read_depth(self, timeout_ms: int = 200, in_meters: bool = False) -> NDArray[Any]:
        depth_frame = super().read_depth(timeout_ms=timeout_ms)
        if in_meters:
            return self.depth_to_meters(depth_frame)
        return depth_frame
