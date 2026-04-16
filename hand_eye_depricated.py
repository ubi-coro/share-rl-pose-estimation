from __future__ import annotations

import argparse
import json
import logging
import sys
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


class UR5eRobot:
    def __init__(self, robot_ip: str, velocity: float, acceleration: float, blend: float) -> None:
        self.velocity = velocity
        self.acceleration = acceleration
        self.blend = blend
        self.control = rtde_control.RTDEControlInterface(robot_ip)
        self.receive = rtde_receive.RTDEReceiveInterface(robot_ip)

    def move_to_pose(self, pose: Pose) -> None:
        tcp_pose = np.concatenate([pose.translation, rotation_matrix_to_rotvec(pose.rotation)])
        ok = self.control.moveL(tcp_pose.tolist(), self.velocity, self.acceleration, False)
        if not ok:
            raise RuntimeError("UR5e moveL failed.")

    def get_tcp_pose(self) -> Pose:
        tcp_pose = np.asarray(self.receive.getActualTCPPose(), dtype=np.float64)
        return Pose(
            translation=tcp_pose[:3],
            rotation=rotvec_to_rotation_matrix(tcp_pose[3:]),
        )

    def stop(self) -> None:
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


def make_pose(values: Sequence[float]) -> Pose:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.shape != (6,):
        raise ValueError("Expected a 6D pose [x, y, z, rx, ry, rz].")
    return Pose(
        translation=values_array[:3],
        rotation=rotvec_to_rotation_matrix(values_array[3:]),
    )


def rotvec_to_rotation_matrix(rotvec: Sequence[float]) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64).reshape(3, 1))
    return rotation_matrix


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    return rotvec.reshape(3)


def rodrigues_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    return rotvec_to_rotation_matrix((rx, ry, rz))


def compose_pose(reference_pose: Pose, delta_translation: Sequence[float], delta_rotation: np.ndarray) -> Pose:
    delta_translation_array = np.asarray(delta_translation, dtype=np.float64).reshape(3)
    return Pose(
        translation=reference_pose.translation + reference_pose.rotation @ delta_translation_array,
        rotation=reference_pose.rotation @ delta_rotation,
    )


def generate_calibration_poses(reference_pose: Pose, trans_off=0.02, rot_off=0.09) -> list[Pose]:
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

    # The sector-based detector is more robust to scale, blur, and lighting variation.
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


def capture_calibration_samples(
    robot: UR5eRobot,
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

        image = camera.capture_image()
        target_to_camera = estimate_target_to_camera(
            image=image,
            board_size=board_size,
            square_size_m=square_size_m,
            camera_matrix=camera.camera_matrix,
            dist_coeffs=camera.dist_coeffs,
        )
        gripper_to_base = robot.get_tcp_pose()
        samples.append(CalibrationSample(gripper_to_base=gripper_to_base, target_to_camera=target_to_camera))
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


def load_poses(path: Path) -> list[Pose]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    poses = [pose_from_json(entry) for entry in payload["poses"]]
    if not poses:
        raise ValueError(f"No poses found in {path}.")
    return poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone UR5e + RealSense D405 hand-eye calibration tool.")
    parser.add_argument("--robot-ip", default="172.22.22.2", help="UR5e robot IP address.")
    parser.add_argument("--camera-serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument(
        "--reference-pose",
        nargs=6,
        type=float,
        #default=None,
        default=[-0.46418443,  0.00693913,  0.65308539, -2.115084951655173, -2.2449716733554577, 0.022559031041897153],
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Optional 6D TCP pose [x, y, z, rx, ry, rz]. If omitted, the current robot pose is used.",
    )
    parser.add_argument(
        "--board-cols",
        type=int,
        default=9,
        help="Checkerboard inner corners along x, not the number of printed squares.",
    )
    parser.add_argument(
        "--board-rows",
        type=int,
        default=6,
        help="Checkerboard inner corners along y, not the number of printed squares.",
    )
    parser.add_argument("--square-size-m", type=float, default=0.022, help="Checkerboard square size in meters.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--velocity", type=float, default=0.10)
    parser.add_argument("--acceleration", type=float, default=0.20)
    parser.add_argument("--blend", type=float, default=0.0)
    parser.add_argument("--settle-time", type=float, default=0.75)
    parser.add_argument("--poses-file", type=Path, default=None, help="Optional JSON file of saved TCP poses to replay.")
    parser.add_argument("--output", type=Path, default=Path("hand_eye_calibration_result.json"))
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    board_size = (args.board_cols, args.board_rows)

    robot = UR5eRobot(
        robot_ip=args.robot_ip,
        velocity=args.velocity,
        acceleration=args.acceleration,
        blend=args.blend,
    )
    camera = RealSenseD405Camera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.camera_serial,
    )

    try:
        current_pose = robot.get_tcp_pose()
        LOGGER.info("Starting TCP Pose: %s", pose_to_json(current_pose))
        if args.poses_file is not None:
            poses = load_poses(args.poses_file)
            reference_pose = poses[0]
            LOGGER.info("Loaded %d poses from %s", len(poses), args.poses_file)
        else:
            reference_pose = make_pose(args.reference_pose) if args.reference_pose is not None else robot.get_tcp_pose()
            poses = generate_calibration_poses(reference_pose)
        samples = capture_calibration_samples(
            robot=robot,
            camera=camera,
            poses=poses,
            board_size=board_size,
            square_size_m=args.square_size_m,
            settle_time_s=args.settle_time,
        )
        camera_to_gripper = calibrate_hand_eye(samples)
    finally:
        camera.stop()
        robot.stop()

    result = {
        "robot_ip": args.robot_ip,
        "camera_serial": args.camera_serial,
        "board_size": list(board_size),
        "square_size_m": args.square_size_m,
        "num_samples": len(samples),
        "poses_file": str(args.poses_file) if args.poses_file is not None else None,
        "reference_pose": pose_to_json(reference_pose),
        "camera_to_gripper": pose_to_json(camera_to_gripper),
        "camera_matrix": camera.camera_matrix.tolist(),
        "dist_coeffs": camera.dist_coeffs.tolist(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    LOGGER.info("%s", json.dumps(result, indent=2))
    LOGGER.info("Saved calibration to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
