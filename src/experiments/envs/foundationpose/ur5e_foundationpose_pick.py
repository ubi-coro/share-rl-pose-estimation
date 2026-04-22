"""UR5e pick pipeline with explicit FoundationPose and grasp-pose TODO hooks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from PIL import Image

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.cameras import Camera
from lerobot.envs import EnvConfig
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator

from share.cameras import RealSenseDepthCamera
from share.cameras.configuration_realsense_depth import RealSenseDepthCameraConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    ObservationConfig,
)
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import Always, OnTargetPoseReached
from share.pose_estimation.grasp_obj_spec import GraspObjectSpec
from share.pose_estimation.pose_estimator import PoseEstimator
from share.robots.ur import MockURConfig, URConfig
from share.teleoperators.spacemouse import SpaceMouseConfig
from share.utils.constants import DEFAULT_ROBOT_NAME

import logging

logger = logging.getLogger(__name__)

EE_POSE_KEYS = ("x.ee_pos", "y.ee_pos", "z.ee_pos", "rx.ee_pos", "ry.ee_pos", "rz.ee_pos")


def _transform_from_translation_rotation(
    translation_m: list[float] | np.ndarray,
    rotation_matrix: list[list[float]] | np.ndarray,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_m, dtype=np.float64).reshape(3)
    return transform


def load_camera_to_gripper_transform(calibration_path: str | Path) -> np.ndarray:
    payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    camera_to_gripper = payload["camera_to_gripper"]
    return _transform_from_translation_rotation(
        translation_m=camera_to_gripper["translation_m"],
        rotation_matrix=camera_to_gripper["rotation_matrix"],
    )


def object_pose_camera_to_robot_base(
    object_pose_camera: np.ndarray,
    tcp_pose_base_to_gripper: list[float] | np.ndarray,
    camera_to_gripper_transform: np.ndarray,
) -> np.ndarray:
    base_to_gripper = np.eye(4, dtype=np.float64)
    base_to_gripper[:3, :3] = Rotation.from_rotvec(np.asarray(tcp_pose_base_to_gripper[3:], dtype=np.float64)).as_matrix()
    base_to_gripper[:3, 3] = np.asarray(tcp_pose_base_to_gripper[:3], dtype=np.float64)
    camera_to_object = np.asarray(object_pose_camera, dtype=np.float64).reshape(4, 4)
    return base_to_gripper @ camera_to_gripper_transform @ camera_to_object


def object_pose_camera_to_tcp_frame(
    object_pose_camera: np.ndarray,
    camera_to_gripper_transform: np.ndarray,
) -> np.ndarray:
    camera_to_object = np.asarray(object_pose_camera, dtype=np.float64).reshape(4, 4)
    return camera_to_gripper_transform @ camera_to_object


def _shared_processor() -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=30,
        observation=ObservationConfig(
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_ee_pos_to_observation=False,
            add_joint_position_to_observation=False,
        ),
        gripper=GripperConfig(
            enable=False,
            static_pos=0.5,
        ),
    )


class FoundationPosePrimitive(ManipulationPrimitive):
    def __init__(
            self,
            task_frame: dict[str, TaskFrame],
            robot_dict: dict[str, Robot],
            cameras: dict[str, Camera],
            display_cameras: bool = False,
    ):
        super().__init__(task_frame, robot_dict, cameras, display_cameras)
        self.pose_estimator = PoseEstimator()
        self._pose_estimator_initialized = False
        self.camera_to_gripper_transform = load_camera_to_gripper_transform(
            "/home/jzilke/ws/share-rl-pe/hand_eye_calibration_result_ur5e.json"
        )
        self.object_spec = GraspObjectSpec.from_json_file(
            "/home/jzilke/ws/share-rl-pe/hoermann_objects/power_connector/object_spec.json"
        )
        self._pose_estimator_config = {
            "mesh_path": self.object_spec.mesh_path,
            "prompt": self.object_spec.segmentation_prompt,
            "confidence_threshold": self.object_spec.confidence_threshold,
        }
        self.debug_output_dir = Path("tmp")

    def step(self, action: dict[str, dict[str, float]]):
        obs, reward, terminated, truncated, info = super().step(action)
        cam = self.cameras.get(DEFAULT_ROBOT_NAME)
        if not isinstance(cam, RealSenseDepthCamera):
            raise TypeError(f"Expected RealSenseDepthCamera, got {type(cam).__name__}")

        if not self._pose_estimator_initialized:
            self.pose_estimator.configure(
                **self._pose_estimator_config,
                camera_intrinsics=cam.get_camera_intrinsics().tolist(),
            )
            self.pose_estimator.restart_tracking()
            self._pose_estimator_initialized = True
        print(f"configuration", self._pose_estimator_config)
        image = obs.get('observation.images.main')
        depth = cam.read_depth(timeout_ms=False, in_meters=True)

        estimation = self.estimate_pose(image, depth)
        pose = estimation.get('pose')
        print("camera-frame pose:\n", pose)
        tcp_pose = [obs[f"{DEFAULT_ROBOT_NAME}.{key}"] for key in EE_POSE_KEYS]
        print("tcp pose [x, y, z, rx, ry, rz]:", tcp_pose)
        pose_tcp = object_pose_camera_to_tcp_frame(
            object_pose_camera=pose,
            camera_to_gripper_transform=self.camera_to_gripper_transform,
        )
        pose_base = object_pose_camera_to_robot_base(
            object_pose_camera=pose,
            tcp_pose_base_to_gripper=tcp_pose,
            camera_to_gripper_transform=self.camera_to_gripper_transform,
        )
        camera_distance_m = float(np.linalg.norm(np.asarray(pose, dtype=np.float64)[:3, 3]))
        tcp_distance_m = float(np.linalg.norm(pose_tcp[:3, 3]))
        base_distance_m = float(np.linalg.norm(pose_base[:3, 3]))
        print("object pose in tcp frame:\n", pose_tcp)
        print("robot-base pose:\n", pose_base)
        print(f"object translation distance in camera frame [m]: {camera_distance_m:.4f}")
        print(f"object translation distance in tcp frame [m]: {tcp_distance_m:.4f}")
        print(f"object translation distance in base frame [m]: {base_distance_m:.4f}")

        return obs, reward, terminated, truncated, info

    def estimate_pose(self, image, depth):
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        estimation = self.pose_estimator.estimate_pose(image=image, depth=depth)

        if 'mask' in estimation.keys() and estimation.get('mask') is not None:
            Image.fromarray(image).save(self.debug_output_dir / "obs_main.png")
            Image.fromarray(estimation.get('mask')).save(self.debug_output_dir / "mask.png")

        return estimation


class FoundationPosePrimitiveConfig(ManipulationPrimitiveConfig):
    prompt: str | None = None
    mesh_file: str | None = None

    def validate(self, robot_dict, teleop_dict):
        super().validate(robot_dict, teleop_dict)

    def make(
            self,
            robot_dict: dict[str, Robot],
            teleop_dict: dict[str, Teleoperator],
            cameras: dict[str, Camera],
            device: str = "cpu"
    ):
        self.validate(robot_dict, teleop_dict)
        self.infer_features(robot_dict, cameras)  # todo: fix initial_features

        display_cameras = self.processor.image_preprocessing is not None and self.processor.image_preprocessing.display_cameras
        env = FoundationPosePrimitive(task_frame=self.task_frame, robot_dict=robot_dict, cameras=cameras,
                                      display_cameras=display_cameras)

        env_processor = self.make_env_processor(device)
        action_processor = self.make_action_processor(robot_dict, teleop_dict, device)
        return env, env_processor, action_processor


def get_target_prim_cfg():
    home_cfg = ManipulationPrimitiveConfig(
        notes="Move to a known safe start pose.",
        task_frame=TaskFrame(
            target=[-0.35270832570279, -0.2680924732831237, 0.45207352093970204, 1.7958561527916685, -0.04133113810150446, -1.420881296314987],
            #target=[-0.29878662794237504, -0.24038619921648444, 0.47113762731620834, 2.088740637708761, -0.049881005988045235, -1.1461642042972513],
            policy_mode=[None] * 6,
            control_mode=[ControlMode.POS] * 6,
        ),
    )
    return home_cfg


@EnvConfig.register_subclass("ur5e_foundationpose_pick")
@dataclass
class UR5eFoundationPosePickEnvConfig(ManipulationPrimitiveNetConfig):
    """UR5e pipeline: scan pose, FoundationPose, grasp target, close gripper."""

    robot_ip: str = "172.22.22.2"
    fps: int = 30
    offline: bool = False
    start_primitive: str = "estimate_object_pose"
    reset_primitive: str = "estimate_object_pose"
    camera_serial_number: str = "352122271533"

    scan_pose: list[float] = field(default_factory=lambda: [-0.429, 0.126, 0.261, 3.112, 0.068, -2.14])
    target_tolerance: list[float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.10, 0.10, 0.10])
    closed_gripper_position: float = 1.0
    mock_initial_pose: list[float] = field(default_factory=lambda: [0.45, -0.20, 0.35, 3.14, 0.0, 0.0])

    def make(self):
        return super().make()

    def __post_init__(self) -> None:
        move_processor = _shared_processor()

        self.robot = URConfig(
            robot_ip=self.robot_ip,
            frequency=500,
            soft_real_time=True,
            rt_core=3,
            use_gripper=True,
        )
        self.teleop = SpaceMouseConfig(action_scale=[0.25, 0.25, 0.20, 0.50, 0.50, 0.50])
        self.cameras = {
            "main": RealSenseDepthCameraConfig(
                serial_number_or_name=self.camera_serial_number,
                use_depth=True
            ),
        }

        self.primitives = {
            "move_to_scan_pose": get_target_prim_cfg(),
            "estimate_object_pose": FoundationPosePrimitiveConfig(
                # processor=move_processor,
                notes="Move the UR5e to the predefined scan pose before running FoundationPose.",
                task_description="estimate object pose",
            ),
        }
        self.transitions = [
            OnTargetPoseReached(
                source="move_to_scan_pose",
                target="estimate_object_pose",
                tolerance=list(self.target_tolerance),
            ),
            OnTargetPoseReached(
                source="estimate_object_pose",
                target="move_to_scan_pose",
                tolerance=list(self.target_tolerance)
            )
        ]

        super().__post_init__()


__all__ = [
    "UR5eFoundationPosePickEnvConfig",
]
