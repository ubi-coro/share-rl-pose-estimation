from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
from hand_eye_lib import (
    CalibrationSample,
    Pose,
    RealSenseD405Camera,
    URRobot,
    build_calibration_result,
    calibrate_hand_eye,
    capture_sample_at_current_pose,
    capture_calibration_samples,
    generate_calibration_poses,
    load_poses,
    make_pose,
    pose_to_json,
    write_json,
    write_poses,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UR hand-eye calibration with automated or manual sample capture.")
    parser.add_argument("--robot-ip", default="172.22.22.2", help="UR robot IP address.")
    parser.add_argument("--camera-serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument(
        "--mode",
        choices=("auto", "manual"),
        default="auto",
        help="Sample capture mode: auto robot motion or manual guided capture.",
    )
    parser.add_argument(
        "--reference-pose",
        nargs=6,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Optional 6D TCP pose [x, y, z, rx, ry, rz] used to generate the default 49-pose calibration grid.",
    )
    parser.add_argument("--poses-file", type=Path, default=None, help="Optional JSON file of saved TCP poses to replay.")
    parser.add_argument(
        "--save-poses",
        "--poses-output",
        dest="save_poses",
        type=Path,
        default=None,
        help="Optional JSON output path for captured/replayed TCP poses.",
    )
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
    parser.add_argument(
        "--output",
        "--calibration-output",
        dest="output",
        type=Path,
        default=Path("hand_eye_calibration_result.json"),
    )
    return parser.parse_args()


def resolve_calibration_poses(args: argparse.Namespace, robot: URRobot) -> tuple[list[Pose], Pose]:
    if args.poses_file is not None:
        poses = load_poses(args.poses_file)
        LOGGER.info("Loaded %d poses from %s", len(poses), args.poses_file)
        return poses, poses[0]

    reference_pose = make_pose(args.reference_pose) if args.reference_pose is not None else robot.get_tcp_pose()
    poses = generate_calibration_poses(reference_pose)
    LOGGER.info("Generated %d calibration poses from reference pose", len(poses))
    return poses, reference_pose


def capture_samples_manual(
    *,
    robot: URRobot,
    camera: RealSenseD405Camera,
    board_size: tuple[int, int],
    square_size_m: float,
    capture_fps: int,
    poses_output: Path,
) -> tuple[list[CalibrationSample], list[Pose]]:
    import tkinter as tk

    from PIL import Image, ImageTk

    poses: list[Pose] = []
    samples: list[CalibrationSample] = []

    LOGGER.info("Manual mode enabled.")
    LOGGER.info("Move the robot manually, then focus the preview window.")
    LOGGER.info("Press Space to capture checkerboard + current TCP pose.")
    LOGGER.info("Press q, Esc, or close the window to finish and calibrate.")

    root = tk.Tk()
    root.title("Manual Hand-Eye Capture")
    image_label = tk.Label(root)
    image_label.pack()
    info_label = tk.Label(root, text="Space: capture sample    q: quit")
    info_label.pack()

    state: dict[str, object] = {"running": True, "image_tk": None, "latest_image": None}

    def update_frame() -> None:
        if not state["running"]:
            return
        image_bgr = camera.capture_image()
        state["latest_image"] = image_bgr
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        state["image_tk"] = ImageTk.PhotoImage(Image.fromarray(image_rgb))
        image_label.configure(image=state["image_tk"])
        root.after(max(1, int(1000 / max(1, capture_fps))), update_frame)

    def handle_capture(_: tk.Event) -> None:
        latest_image = state["latest_image"]
        if latest_image is None:
            LOGGER.warning("No camera frame available yet.")
            return
        try:
            sample = capture_sample_at_current_pose(
                robot=robot,
                camera=camera,
                board_size=board_size,
                square_size_m=square_size_m,
                image=latest_image,
            )
        except ValueError as exc:
            LOGGER.warning("Capture rejected: %s", exc)
            return

        samples.append(sample)
        poses.append(sample.gripper_to_base)
        write_poses(poses_output, poses)
        LOGGER.info("Saved sample %d to %s", len(samples), poses_output)
        LOGGER.info("Pose: %s", json.dumps(pose_to_json(sample.gripper_to_base), indent=2))

    def handle_quit(_: tk.Event | None = None) -> None:
        state["running"] = False
        root.quit()

    root.bind("<space>", handle_capture)
    root.bind("q", handle_quit)
    root.bind("<Escape>", handle_quit)
    root.protocol("WM_DELETE_WINDOW", handle_quit)

    try:
        update_frame()
        root.mainloop()
    finally:
        root.destroy()

    write_poses(poses_output, poses)
    return samples, poses


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    board_size = (args.board_cols, args.board_rows)

    is_manual = args.mode == "manual"
    poses_output = args.save_poses
    if is_manual and poses_output is None:
        poses_output = Path("robot_poses.json")

    robot = URRobot(
        robot_ip=args.robot_ip,
        velocity=args.velocity,
        acceleration=args.acceleration,
        blend=args.blend,
        enable_control=not is_manual,
    )
    camera = RealSenseD405Camera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.camera_serial,
    )

    try:
        if is_manual:
            try:
                samples, poses = capture_samples_manual(
                    robot=robot,
                    camera=camera,
                    board_size=board_size,
                    square_size_m=args.square_size_m,
                    capture_fps=args.fps,
                    poses_output=poses_output,
                )
            except ImportError as exc:
                LOGGER.error("Manual mode requires tkinter and pillow: %s", exc)
                return 1
            if len(samples) < 3:
                LOGGER.info("Finished with %d samples saved to %s", len(samples), poses_output)
                LOGGER.info("Need at least 3 valid samples to compute hand-eye calibration.")
                return 0
            reference_pose = poses[0]
        else:
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
            if poses_output is not None:
                write_poses(poses_output, poses)
                LOGGER.info("Saved %d replay poses to %s", len(poses), poses_output)
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
        poses_file=poses_output if poses_output is not None else args.poses_file,
    )
    write_json(args.output, result)

    LOGGER.info("%s", json.dumps(result, indent=2))
    if poses_output is not None:
        LOGGER.info("Saved poses to %s", poses_output)
    LOGGER.info("Saved calibration to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
