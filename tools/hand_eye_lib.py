from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_control
import rtde_receive

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pose:
    translation: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class CalibrationSample:
    gripper_to_base: Pose
    target_to_camera: Pose


class URRobot:
    def __init__(
        self,
        robot_ip: str,
        velocity: float = 0.10,
        acceleration: float = 0.20,
        blend: float = 0.0,
        enable_control: bool = True,
    ) -> None:
        self.velocity = velocity
        self.acceleration = acceleration
        self.blend = blend
        self.receive = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.control = rtde_control.RTDEControlInterface(robot_ip) if enable_control else None

    def move_to_pose(self, pose: Pose) -> None:
        if self.control is None:
            raise RuntimeError("Robot motion control is not enabled.")
        tcp_pose = np.concatenate([pose.translation, rotation_matrix_to_rotvec(pose.rotation)])
        ok = self.control.moveL(tcp_pose.tolist(), self.velocity, self.acceleration, False)
        if not ok:
            raise RuntimeError("UR moveL failed.")

    def get_tcp_pose(self) -> Pose:
        tcp_pose = np.asarray(self.receive.getActualTCPPose(), dtype=np.float64)
        return Pose(
            translation=tcp_pose[:3],
            rotation=rotvec_to_rotation_matrix(tcp_pose[3:]),
        )

    def stop(self) -> None:
        if self.control is not None:
            self.control.stopScript()


class RealSenseD405Camera:
    def __init__(self, width: int, height: int, fps: int, serial: str | None = None, warmup_frames: int = 30) -> None:
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(self.config)

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        self.camera_matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.asarray(intrinsics.coeffs, dtype=np.float64)

        for _ in range(max(0, warmup_frames)):
            self.pipeline.wait_for_frames()

    def capture_image(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("Failed to capture a color frame from the RealSense D405.")
        return np.asanyarray(color_frame.get_data())

    def stop(self) -> None:
        self.pipeline.stop()


def rotvec_to_rotation_matrix(rotvec: Sequence[float]) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64).reshape(3, 1))
    return rotation_matrix


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    return rotvec.reshape(3)


def make_pose(values: Sequence[float]) -> Pose:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.shape != (6,):
        raise ValueError("Expected a 6D pose [x, y, z, rx, ry, rz].")
    return Pose(
        translation=values_array[:3],
        rotation=rotvec_to_rotation_matrix(values_array[3:]),
    )


def rodrigues_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    return rotvec_to_rotation_matrix((rx, ry, rz))


def compose_pose(reference_pose: Pose, delta_translation: Sequence[float], delta_rotation: np.ndarray) -> Pose:
    delta_translation_array = np.asarray(delta_translation, dtype=np.float64).reshape(3)
    return Pose(
        translation=reference_pose.translation + reference_pose.rotation @ delta_translation_array,
        rotation=reference_pose.rotation @ delta_rotation,
    )


def generate_calibration_poses(reference_pose: Pose, trans_off: float = 0.02, rot_off: float = 0.09) -> list[Pose]:
    translation_offsets = (
        (0.00, 0.00, 0.00),
        (trans_off, 0.00, 0.00),
        (-trans_off, 0.00, 0.00),
        (0.00, trans_off, 0.00),
        (0.00, -trans_off, 0.00),
        (0.00, 0.00, trans_off),
        (0.00, 0.00, -trans_off),
    )
    rotation_offsets = (
        (0.00, 0.00, 0.00),
        (rot_off, 0.00, 0.00),
        (-rot_off, 0.00, 0.00),
        (0.00, rot_off, 0.00),
        (0.00, -rot_off, 0.00),
        (0.00, 0.00, rot_off),
        (0.00, 0.00, -rot_off),
    )

    poses: list[Pose] = []
    for translation_offset in translation_offsets:
        for rotation_offset in rotation_offsets:
            poses.append(
                compose_pose(
                    reference_pose,
                    delta_translation=translation_offset,
                    delta_rotation=rodrigues_rotation(*rotation_offset),
                )
            )
    return poses


def create_checkerboard_points(board_size: tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = board_size
    object_points = np.zeros((cols * rows, 3), dtype=np.float64)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points *= square_size_m
    return object_points


def detect_checkerboard_corners(image: np.ndarray, board_size: tuple[int, int]) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(grayscale, board_size, classic_flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        return cv2.cornerSubPix(grayscale, corners, (11, 11), (-1, -1), criteria)

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE
        found_sb, corners_sb = cv2.findChessboardCornersSB(grayscale, board_size, sb_flags)
        if found_sb:
            return corners_sb

    cols, rows = board_size
    raise ValueError(
        "Checkerboard could not be detected. "
        f"Expected {cols}x{rows} inner corners. "
        "If your board has that many squares instead, rerun with one less in each direction "
        f"(for example, {cols - 1}x{rows - 1})."
    )


def estimate_target_to_camera(
    image: np.ndarray,
    board_size: tuple[int, int],
    square_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Pose:
    refined_corners = detect_checkerboard_corners(image, board_size)

    success, rvec, tvec = cv2.solvePnP(
        create_checkerboard_points(board_size, square_size_m),
        refined_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError("solvePnP failed for the checkerboard observation.")

    return Pose(
        translation=tvec.reshape(3).astype(np.float64),
        rotation=rotvec_to_rotation_matrix(rvec.reshape(3)),
    )


def capture_sample_at_current_pose(
    robot: URRobot,
    camera: RealSenseD405Camera,
    board_size: tuple[int, int],
    square_size_m: float,
    image: np.ndarray | None = None,
) -> CalibrationSample:
    sample_image = camera.capture_image() if image is None else image
    target_to_camera = estimate_target_to_camera(
        image=sample_image,
        board_size=board_size,
        square_size_m=square_size_m,
        camera_matrix=camera.camera_matrix,
        dist_coeffs=camera.dist_coeffs,
    )
    gripper_to_base = robot.get_tcp_pose()
    return CalibrationSample(gripper_to_base=gripper_to_base, target_to_camera=target_to_camera)


def capture_calibration_samples(
    robot: URRobot,
    camera: RealSenseD405Camera,
    poses: Sequence[Pose],
    board_size: tuple[int, int],
    square_size_m: float,
    settle_time_s: float,
) -> list[CalibrationSample]:
    samples: list[CalibrationSample] = []
    for index, pose in enumerate(poses, start=1):
        robot.move_to_pose(pose)
        time.sleep(settle_time_s)
        samples.append(capture_sample_at_current_pose(robot, camera, board_size, square_size_m))
        LOGGER.info("[%02d/%02d] captured sample", index, len(poses))
    return samples


def calibrate_hand_eye(samples: Sequence[CalibrationSample], method: int = cv2.CALIB_HAND_EYE_TSAI) -> Pose:
    if len(samples) < 3:
        raise ValueError("At least three calibration samples are required.")

    rotations_gripper_to_base = [sample.gripper_to_base.rotation for sample in samples]
    translations_gripper_to_base = [sample.gripper_to_base.translation.reshape(3, 1) for sample in samples]
    rotations_target_to_camera = [sample.target_to_camera.rotation for sample in samples]
    translations_target_to_camera = [sample.target_to_camera.translation.reshape(3, 1) for sample in samples]

    rotation_camera_to_gripper, translation_camera_to_gripper = cv2.calibrateHandEye(
        R_gripper2base=rotations_gripper_to_base,
        t_gripper2base=translations_gripper_to_base,
        R_target2cam=rotations_target_to_camera,
        t_target2cam=translations_target_to_camera,
        method=method,
    )
    return Pose(
        translation=np.asarray(translation_camera_to_gripper, dtype=np.float64).reshape(3),
        rotation=np.asarray(rotation_camera_to_gripper, dtype=np.float64).reshape(3, 3),
    )


def pose_to_json(pose: Pose) -> dict[str, list[list[float]] | list[float]]:
    return {
        "translation_m": pose.translation.tolist(),
        "rotation_matrix": pose.rotation.tolist(),
        "rotation_vector_rad": rotation_matrix_to_rotvec(pose.rotation).tolist(),
    }


def pose_from_json(payload: dict[str, object]) -> Pose:
    translation = np.asarray(payload["translation_m"], dtype=np.float64).reshape(3)
    if "rotation_vector_rad" in payload:
        rotation = rotvec_to_rotation_matrix(payload["rotation_vector_rad"])
    else:
        rotation = np.asarray(payload["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    return Pose(translation=translation, rotation=rotation)


def write_poses(path: Path, poses: Sequence[Pose]) -> None:
    payload = {
        "num_poses": len(poses),
        "poses": [pose_to_json(pose) for pose in poses],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_poses(path: Path) -> list[Pose]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses = [pose_from_json(entry) for entry in payload["poses"]]
    if not poses:
        raise ValueError(f"No poses found in {path}.")
    return poses


def build_calibration_result(
    *,
    robot_ip: str,
    camera_serial: str | None,
    board_size: tuple[int, int],
    square_size_m: float,
    num_samples: int,
    reference_pose: Pose,
    camera_to_gripper: Pose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    poses_file: Path | None,
) -> dict[str, object]:
    return {
        "robot_ip": robot_ip,
        "camera_serial": camera_serial,
        "board_size": list(board_size),
        "square_size_m": square_size_m,
        "num_samples": num_samples,
        "poses_file": str(poses_file) if poses_file is not None else None,
        "reference_pose": pose_to_json(reference_pose),
        "camera_to_gripper": pose_to_json(camera_to_gripper),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
