from typing import Any

import numpy as np
from lerobot.cameras import Camera
from lerobot.robots import Robot

from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.task_frame import TaskFrame


class FoundationPosePrimitive(ManipulationPrimitive):
    """Primitive env variant that executes a scripted chunk per outer step."""

    def __init__(
        self,
        task_frame: dict[str, TaskFrame],
        robot_dict: dict[str, Robot],
        cameras: dict[str, Camera],
        open_loop_config: Any,
        display_cameras: bool = False,
    ):
        """Initialize the scripted open-loop primitive env.

        Args:
            task_frame: Per-robot task-frame command objects.
            robot_dict: Connected robots keyed by primitive robot name.
            cameras: Connected cameras exposed through the env observation dict.
            open_loop_config: Config object supplying trajectory timing params.
            display_cameras: Whether ``render()`` should open live camera views.
        """
        self.open_loop_config = open_loop_config
        self._trajectory_substeps = 0
        super().__init__(
            task_frame=task_frame,
            robot_dict=robot_dict,
            cameras=cameras,
            display_cameras=display_cameras,
        )

    @property
    def uses_autonomous_step(self) -> bool:
        return True

    def reset_runtime_state(self) -> None:
        super().reset_runtime_state()
        self._start_pose: dict[str, list[float]] = {}
        self._duration_substeps = 0
        self._substeps_per_step = 1
        self._trajectory_substeps = 0

    def configure_trajectory(
        self,
        start_pose: dict[str, list[float]],
        target_pose: dict[str, list[float]],
        info_key: str | None,
    ) -> None:
        """Initialize one scripted trajectory from entry-time poses.

        Args:
            start_pose: Per-robot 6D start poses in this primitive's task frame.
            target_pose: Per-robot 6D target poses in this primitive's task
                frame. These are also published as the env target pose.
        """
        self._start_pose = {name: list(pose) for name, pose in start_pose.items()}
        self._duration_substeps, self._substeps_per_step = self.open_loop_config.trajectory_timing(self.robot_dict)
        self.set_target_pose(
            target_pose=target_pose,
            info_key=info_key,
            update_task_frame=False,
        )
        self._set_live_task_frame_pose(start_pose)
        self._primitive_complete = False
        self._trajectory_progress = 0.0
        self._trajectory_substeps = 0

    def step(self, action: dict[str, dict[str, float]]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Run a scripted chunk of the open-loop trajectory.

        Args:
            action: Ignored in v1. The primitive follows its scripted internal
                trajectory instead of consuming an external policy action.

        Returns:
            Standard gym ``(observation, reward, terminated, truncated, info)``
            values after executing up to the configured internal robot
            steps.
        """
        substeps = max(1, int(self._substeps_per_step))
        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = False
        for _ in range(substeps):
            self._trajectory_substeps += 1
            alpha = self._trajectory_substeps / float(self._duration_substeps)
            scripted_pose = self.open_loop_config.target_pose_at(
                alpha=alpha,
                start_pose=self._start_pose,
                goal_pose=self._target_pose,
            )
            self._set_live_task_frame_pose(scripted_pose)
            scripted_action = self._action_from_pose(scripted_pose)
            obs, step_reward, terminated, truncated, _info = super().step(scripted_action)
            reward += step_reward
            self._trajectory_progress = min(1.0, alpha)
            self._primitive_complete = alpha >= 1.0
            if terminated or truncated:
                break

        return obs, reward, terminated, truncated, self._get_info()

    def _set_live_task_frame_pose(self, pose_by_robot: dict[str, list[float]]) -> None:
        """Update the live task-frame setpoint used by the scripted runner."""
        for name, pose in pose_by_robot.items():
            if name not in self.task_frame:
                continue
            for axis in range(min(len(self.task_frame[name].target), len(pose))):
                self.task_frame[name].target[axis] = float(pose[axis])

    def _action_from_pose(self, pose_by_robot: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        """Build one low-level action dict from the current scripted target pose.

        Args:
            pose_by_robot: Per-robot scripted task-space pose for this substep.

        Returns:
            Nested low-level robot commands matching the current scripted pose.
        """
        action: dict[str, dict[str, float]] = {}
        for name, pose in pose_by_robot.items():
            frame = self.task_frame[name]
            action[name] = {
                frame.action_key_for_axis(axis): float(pose[axis])
                for axis in range(len(frame.target))
            }
        return action