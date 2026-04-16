"""UR5e pick pipeline with explicit FoundationPose and grasp-pose TODO hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs import EnvConfig
from lerobot.processor.hil_processor import GRIPPER_KEY

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    ObservationConfig,
    PrimitiveEntryContext,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.envs.manipulation_primitive_net.transitions import Always, OnTargetPoseReached
from share.pose_estimation.pose_estimator import PoseEstimator
from share.robots.ur import URConfig
from share.teleoperators.spacemouse import SpaceMouseConfig
from share.utils.constants import DEFAULT_ROBOT_NAME
from share.utils.mock_utils import MockRobot, MockTeleoperator

import logging
logger = logging.getLogger(__name__)



def _single_robot_pose(pose: list[float] | dict[str, list[float]]) -> dict[str, list[float]]:
    if isinstance(pose, dict):
        return {name: [float(v) for v in values] for name, values in pose.items()}
    return {DEFAULT_ROBOT_NAME: [float(v) for v in pose]}


def _shared_processor(*, fps: float, gripper_static_pos: float | None = None) -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=fps,
        observation=ObservationConfig(
            add_ee_pos_to_observation=True,
            add_ee_velocity_to_observation=False,
            add_ee_wrench_to_observation=False,
            add_joint_position_to_observation=True,
        ),
        gripper=GripperConfig(
            enable=False,
            static_pos=gripper_static_pos,
        ),
    )


def _pose_from_entry_observation(
    entry_context: PrimitiveEntryContext | None,
    robot_name: str = DEFAULT_ROBOT_NAME,
) -> list[float]:
    if entry_context is None:
        return [0.0] * 6

    pose = []
    for axis_name in ("x", "y", "z", "rx", "ry", "rz"):
        key = f"{robot_name}.{axis_name}.ee_pos"
        value = entry_context.observation.get(key)
        if value is None:
            return [0.0] * 6
        pose.append(float(value))
    return pose


def _task_frame(target: list[float] | None = None) -> TaskFrame:
    return TaskFrame(
        target=[0.0] * 6 if target is None else [float(v) for v in target],
        space=ControlSpace.TASK,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[None] * 6,
    )

@ManipulationPrimitiveConfig.register_subclass("foundationpose_capture")
@dataclass
class FoundationPoseCapturePrimitiveConfig(ManipulationPrimitiveConfig):
    """Hold pose, run FoundationPose, and stash the resulting object pose."""

    object_pose_store_key: str = "object_pose"
    mesh_path: str | None = None
    prompt: str | None = None
    threshold: float | None = None

    def __post_init__(self):
        super().__post_init__()
        self.pose_estimator = PoseEstimator()

    def capture_object_pose(
        self,
        entry_context: PrimitiveEntryContext | None,
    ) -> list[float] | dict[str, list[float]]:
        logger.warning(
            "FoundationPoseCapturePrimitiveConfig.capture_object_pose() is not implemented. "
            "Using the current end-effector pose as a placeholder object pose."
        )
        return _pose_from_entry_observation(entry_context)

    def on_entry(self, env, entry_context: PrimitiveEntryContext | None) -> None:
        super().on_entry(env, entry_context)
        env.set_runtime_value(
            self.object_pose_store_key,
            _single_robot_pose(self.capture_object_pose(entry_context)),
        )


@ManipulationPrimitiveConfig.register_subclass("grasp_pose_from_object")
@dataclass
class GraspPoseFromObjectPrimitiveConfig(ManipulationPrimitiveConfig):
    """Read a stored object pose and resolve the task-frame grasp target."""

    object_pose_store_key: str = "object_pose"
    grasp_pose_store_key: str | None = "grasp_pose"

    def compute_grasp_pose(
        self,
        object_pose: dict[str, list[float]],
        entry_context: PrimitiveEntryContext | None,
    ) -> list[float] | dict[str, list[float]]:
        logger.warning(
            "GraspPoseFromObjectPrimitiveConfig.compute_grasp_pose() is not implemented. "
            "Returning the stored object pose as a placeholder grasp pose."
        )
        return next(iter(object_pose.values()))

    def on_entry(self, env, entry_context: PrimitiveEntryContext | None) -> None:
        object_pose = env.get_runtime_value(self.object_pose_store_key)
        if object_pose is None:
            raise KeyError(
                f"Missing runtime value '{self.object_pose_store_key}'. "
                "Make sure the FoundationPose primitive runs before the grasp primitive."
            )

        grasp_pose = _single_robot_pose(self.compute_grasp_pose(object_pose, entry_context))
        if self.grasp_pose_store_key is not None:
            env.set_runtime_value(self.grasp_pose_store_key, grasp_pose)
        env.set_target_pose(grasp_pose, info_key=self.target_pose_info_key)


@EnvConfig.register_subclass("ur5e_foundationpose_pick")
@dataclass
class UR5eFoundationPosePickEnvConfig(ManipulationPrimitiveNetConfig):
    """UR5e pipeline: scan pose, FoundationPose, grasp target, close gripper."""

    robot_ip: str = "172.22.22.2"
    fps: int = 30
    offline: bool = False
    start_primitive: str = "move_to_scan_pose"
    reset_primitive: str = "move_to_scan_pose"
    camera_serial_number: str = "352122273250"

    scan_pose: list[float] = field(default_factory=lambda: [-0.429, 0.126, 0.261, 3.112, 0.068, -2.14])
    target_tolerance: list[float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.10, 0.10, 0.10])
    closed_gripper_position: float = 1.0
    mock_initial_pose: list[float] = field(default_factory=lambda: [0.45, -0.20, 0.35, 3.14, 0.0, 0.0])

    def make(self):
        return super().make()

    def __post_init__(self) -> None:
        move_processor = _shared_processor(fps=float(self.fps))

        self.robot = URConfig(
            robot_ip=self.robot_ip,
            frequency=500,
            soft_real_time=True,
            rt_core=3,
            use_gripper=True,
        )
        self.teleop = SpaceMouseConfig(action_scale=[0.25, 0.25, 0.20, 0.50, 0.50, 0.50])
        self.cameras = {
            "main": RealSenseCameraConfig(serial_number_or_name=self.camera_serial_number),
        }

        self.primitives = {
            "move_to_scan_pose": ManipulationPrimitiveConfig(
                task_frame=_task_frame(self.scan_pose),
                processor=move_processor,
                notes="Move the UR5e to the predefined scan pose before running FoundationPose.",
                task_description="move to scan pose",
            ),
            "estimate_object_pose": ManipulationPrimitiveConfig(
                task_frame=_task_frame(self.scan_pose),
                processor=move_processor,
                notes="Move the UR5e to the predefined scan pose before running FoundationPose.",
                task_description="move to scan pose",
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
    "FoundationPoseCapturePrimitiveConfig",
    "GraspPoseFromObjectPrimitiveConfig",
    "UR5eFoundationPosePickEnvConfig",
]
