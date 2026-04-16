from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from hand_eye_lib import (
    RealSenseD405Camera,
    URRobot,
    build_calibration_result,
    calibrate_hand_eye,
    capture_calibration_samples,
    generate_calibration_poses,
    load_poses,
    make_pose,
    pose_to_json,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated UR hand-eye calibration with generated or saved poses.")
    parser.add_argument("--robot-ip", default="172.22.22.2", help="UR robot IP address.")
    parser.add_argument("--camera-serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument(
        "--reference-pose",
        nargs=6,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Optional 6D TCP pose [x, y, z, rx, ry, rz] used to generate the default 49-pose calibration grid.",
    )
    parser.add_argument("--poses-file", type=Path, default=None, help="Optional JSON file of saved TCP poses to replay.")
    parser.add_argument("--board-cols", type=int, default=9, help="Checkerboard inner corners along x.")
    parser.add_argument("--board-rows", type=int, default=6, help="Checkerboard inner corners along y.")
    parser.add_argument("--square-size-m", type=float, default=0.022, help="Checkerboard square size in meters.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--velocity", type=float, default=0.10)
    parser.add_argument("--acceleration", type=float, default=0.20)
    parser.add_argument("--blend", type=float, default=0.0)
    parser.add_argument("--settle-time", type=float, default=0.75)
    parser.add_argument("--output", type=Path, default=Path("hand_eye_calibration_result.json"))
    return parser.parse_args()


def resolve_calibration_poses(args: argparse.Namespace, robot: URRobot) -> tuple[list, object]:
    if args.poses_file is not None:
        poses = load_poses(args.poses_file)
        LOGGER.info("Loaded %d poses from %s", len(poses), args.poses_file)
        return poses, poses[0]

    reference_pose = make_pose(args.reference_pose) if args.reference_pose is not None else robot.get_tcp_pose()
    poses = generate_calibration_poses(reference_pose)
    LOGGER.info("Generated %d calibration poses from reference pose", len(poses))
    return poses, reference_pose


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    board_size = (args.board_cols, args.board_rows)

    robot = URRobot(
        robot_ip=args.robot_ip,
        velocity=args.velocity,
        acceleration=args.acceleration,
        blend=args.blend,
        enable_control=True,
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
        poses, reference_pose = resolve_calibration_poses(args, robot)
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

    result = build_calibration_result(
        robot_ip=args.robot_ip,
        camera_serial=args.camera_serial,
        board_size=board_size,
        square_size_m=args.square_size_m,
        num_samples=len(samples),
        reference_pose=reference_pose,
        camera_to_gripper=camera_to_gripper,
        camera_matrix=camera.camera_matrix,
        dist_coeffs=camera.dist_coeffs,
        poses_file=args.poses_file,
    )
    write_json(args.output, result)

    LOGGER.info("%s", json.dumps(result, indent=2))
    LOGGER.info("Saved calibration to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
