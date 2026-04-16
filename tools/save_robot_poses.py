from __future__ import annotations

import argparse
import json
import logging
import sys
import tkinter as tk
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from hand_eye_lib import (
    CalibrationSample,
    RealSenseD405Camera,
    URRobot,
    build_calibration_result,
    calibrate_hand_eye,
    capture_sample_at_current_pose,
    pose_to_json,
    write_json,
    write_poses,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual UR hand-eye calibration with live camera preview.")
    parser.add_argument("--robot-ip", required=True, help="UR robot IP address.")
    parser.add_argument("--camera-serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--board-cols", type=int, default=9, help="Checkerboard inner corners along x.")
    parser.add_argument("--board-rows", type=int, default=6, help="Checkerboard inner corners along y.")
    parser.add_argument("--square-size-m", type=float, default=0.022, help="Checkerboard square size in meters.")
    parser.add_argument("--width", type=int, default=640, help="Camera preview width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Camera preview height in pixels.")
    parser.add_argument("--fps", type=int, default=30, help="Camera preview frame rate.")
    parser.add_argument("--poses-output", type=Path, default=Path("robot_poses.json"), help="Output JSON path for captured robot poses.")
    parser.add_argument("--calibration-output", type=Path, default=Path("hand_eye_calibration_result.json"), help="Output JSON path for the hand-eye result.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    board_size = (args.board_cols, args.board_rows)

    robot = URRobot(args.robot_ip, enable_control=False)
    camera = RealSenseD405Camera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.camera_serial,
    )
    poses = []
    samples: list[CalibrationSample] = []

    LOGGER.info("Connected to %s", args.robot_ip)
    LOGGER.info("Move the robot manually, then focus the camera window.")
    LOGGER.info("Press Space to capture a checkerboard sample and save the current TCP pose.")
    LOGGER.info("Press 'q', Esc, or close the window to finish and compute the calibration.")

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
        root.after(max(1, int(1000 / max(1, args.fps))), update_frame)

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
                square_size_m=args.square_size_m,
                image=latest_image,
            )
        except ValueError as exc:
            LOGGER.warning("Capture rejected: %s", exc)
            return

        samples.append(sample)
        poses.append(sample.gripper_to_base)
        write_poses(args.poses_output, poses)
        LOGGER.info("Saved sample %d to %s", len(samples), args.poses_output)
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
        camera.stop()
        root.destroy()
        robot.stop()

    write_poses(args.poses_output, poses)
    if len(samples) < 3:
        LOGGER.info("Finished with %d samples saved to %s", len(samples), args.poses_output)
        LOGGER.info("Need at least 3 valid samples to compute hand-eye calibration.")
        return 0

    result_pose = calibrate_hand_eye(samples)
    result = build_calibration_result(
        robot_ip=args.robot_ip,
        camera_serial=args.camera_serial,
        board_size=board_size,
        square_size_m=args.square_size_m,
        num_samples=len(samples),
        reference_pose=poses[0],
        camera_to_gripper=result_pose,
        camera_matrix=camera.camera_matrix,
        dist_coeffs=camera.dist_coeffs,
        poses_file=args.poses_output,
    )
    write_json(args.calibration_output, result)

    LOGGER.info("%s", json.dumps(result, indent=2))
    LOGGER.info("Saved poses to %s", args.poses_output)
    LOGGER.info("Saved calibration to %s", args.calibration_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
