from __future__ import annotations

import argparse
import json
import logging
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rtde_receive
from PIL import Image, ImageTk

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pose:
    translation: np.ndarray
    rotation: np.ndarray


class UR5eRobotReceiver:
    def __init__(self, robot_ip: str) -> None:
        self.receive = rtde_receive.RTDEReceiveInterface(robot_ip)

    def get_tcp_pose(self) -> Pose:
        tcp_pose = np.asarray(self.receive.getActualTCPPose(), dtype=np.float64)
        return Pose(
            translation=tcp_pose[:3],
            rotation=rotvec_to_rotation_matrix(tcp_pose[3:]),
        )


class RealSenseD405Camera:
    def __init__(self, width: int, height: int, fps: int, serial: str | None = None, warmup_frames: int = 30) -> None:
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(self.config)

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


def rotvec_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64).reshape(3, 1))
    return rotation_matrix


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    return rotvec.reshape(3)


def pose_to_json(pose: Pose) -> dict[str, list[list[float]] | list[float]]:
    return {
        "translation_m": pose.translation.tolist(),
        "rotation_matrix": pose.rotation.tolist(),
        "rotation_vector_rad": rotation_matrix_to_rotvec(pose.rotation).tolist(),
    }


def write_poses(output_path: Path, poses: list[Pose]) -> None:
    payload = {
        "num_poses": len(poses),
        "poses": [pose_to_json(pose) for pose in poses],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_current_pose(robot: UR5eRobotReceiver, poses: list[Pose], output_path: Path) -> None:
    pose = robot.get_tcp_pose()
    poses.append(pose)
    write_poses(output_path, poses)
    LOGGER.info("Saved pose %d to %s", len(poses), output_path)
    LOGGER.info("%s", json.dumps(pose_to_json(pose), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save UR5e TCP poses while manually moving the robot.")
    parser.add_argument("--robot-ip", required=True, help="UR robot IP address.")
    parser.add_argument("--output", type=Path, default=Path("robot_poses.json"), help="Output JSON path.")
    parser.add_argument("--camera-serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--width", type=int, default=640, help="Camera preview width in pixels.")
    parser.add_argument("--height", type=int, default=480, help="Camera preview height in pixels.")
    parser.add_argument("--fps", type=int, default=30, help="Camera preview frame rate.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    robot = UR5eRobotReceiver(args.robot_ip)
    camera = RealSenseD405Camera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.camera_serial,
    )
    poses: list[Pose] = []

    LOGGER.info("Connected to %s", args.robot_ip)
    LOGGER.info("Move the robot manually, then focus the camera window.")
    LOGGER.info("Press Space in the camera window to save the current TCP pose.")
    LOGGER.info("Press 'q' or close the window to finish.")

    root = tk.Tk()
    root.title("RealSense Preview")
    image_label = tk.Label(root)
    image_label.pack()
    info_label = tk.Label(root, text="Space: save pose    q: quit")
    info_label.pack()

    state = {"running": True, "image_tk": None}

    def update_frame() -> None:
        if not state["running"]:
            return
        image_bgr = camera.capture_image()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        state["image_tk"] = ImageTk.PhotoImage(Image.fromarray(image_rgb))
        image_label.configure(image=state["image_tk"])
        root.after(max(1, int(1000 / max(1, args.fps))), update_frame)

    def handle_save(_: tk.Event) -> None:
        save_current_pose(robot, poses, args.output)

    def handle_quit(_: tk.Event | None = None) -> None:
        state["running"] = False
        root.quit()

    root.bind("<space>", handle_save)
    root.bind("q", handle_quit)
    root.bind("<Escape>", handle_quit)
    root.protocol("WM_DELETE_WINDOW", handle_quit)

    try:
        update_frame()
        root.mainloop()
    finally:
        camera.stop()
        root.destroy()

    write_poses(args.output, poses)
    LOGGER.info("Finished with %d poses saved to %s", len(poses), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
