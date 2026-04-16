"""Minimal real-robot example: move to a known pose, then lift 10 cm in z."""

from __future__ import annotations

import time

import torch
from lerobot.processor import TransitionKey
from lerobot.utils.robot_utils import precise_sleep

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    ManipulationPrimitiveConfig,
    MoveDeltaPrimitiveConfig,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.envs.manipulation_primitive_net.transitions import OnTargetPoseReached
from share.robots.ur import URConfig
from share.teleoperators.spacemouse import SpaceMouseConfig


# Set this to a safe pose for your robot in world coordinates:
# [x, y, z, rx, ry, rz]
START_POSE = [-0.34, 0.03, 0.53, -2.38, 2.0, -0.03]
TARGET_1 = [-0.34, 0.03, 0.43, 2.2, -1.7, -0.72]
TARGET_2 = [-0.34, 0.05, 0.43, 2.2, -1.7, -0.72]
TARGET_3 = [-0.34, 0.03, 0.43, 2.2, -1.7, -0.72]
TARGET_4 = [-0.24, 0.05, 0.43, 2.2, -2.0, -0.0]


home_cfg = ManipulationPrimitiveConfig(
    notes="Move to a known safe start pose.",
    task_frame=TaskFrame(
        target=START_POSE,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    ),
)

home_cfg_1 = ManipulationPrimitiveConfig(
    notes="Resolve the target at entry time as start pose + 0.10 m in world z.",
    task_frame=TaskFrame(
        target=TARGET_1,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    )
)

home_cfg_2 = ManipulationPrimitiveConfig(
    notes="Resolve the target at entry time as start pose + 0.10 m in world z.",
    task_frame=TaskFrame(
        target=TARGET_2,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    )
)

home_cfg_3 = ManipulationPrimitiveConfig(
    notes="Resolve the target at entry time as start pose + 0.10 m in world z.",
    task_frame=TaskFrame(
        target=TARGET_3,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    )
)

home_cfg_4 = ManipulationPrimitiveConfig(
    notes="Resolve the target at entry time as start pose + 0.10 m in world z.",
    task_frame=TaskFrame(
        target=TARGET_4,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    )
)
home_cfg_5 = ManipulationPrimitiveConfig(
    notes="Resolve the target at entry time as start pose + 0.10 m in world z.",
    task_frame=TaskFrame(
        target=START_POSE,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    )
)

done_cfg = ManipulationPrimitiveConfig(
    notes="Terminal hold after the 10 cm lift completes.",
    task_frame=TaskFrame(
        target=START_POSE,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    ),
    is_terminal=True,
)


net_cfg = ManipulationPrimitiveNetConfig(
    fps=30,
    start_primitive="home",
    reset_primitive="home",
    primitives={
        "home": home_cfg,
        "home_1": home_cfg_1,
        "home_2": home_cfg_2,
        "home_3": home_cfg_3,
        "home_4": home_cfg_4,
        "home_5": home_cfg_5,
        "done": done_cfg,
    },
    transitions=[
        OnTargetPoseReached(source="home", target="home_1", axes=["x", "y", "z"], tolerance=0.01),
        OnTargetPoseReached(source="home_1", target="home_2", axes=["x", "y", "z"], tolerance=0.01),
        OnTargetPoseReached(source="home_2", target="home_3", axes=["x", "y", "z"], tolerance=0.01),
        OnTargetPoseReached(source="home_3", target="home_4", axes=["x", "y", "z"], tolerance=0.01),
        OnTargetPoseReached(source="home_4", target="home_5", axes=["x", "y", "z"], tolerance=0.01),
        OnTargetPoseReached(source="home_5", target="done", axes=["x", "y", "z"], tolerance=0.01),
    ],
    robot=URConfig(
        robot_ip="172.22.22.2",
        frequency=500,
        soft_real_time=True,
        rt_core=3,
        use_gripper=True,
    ),
    # This example does not use teleop input, but the current runtime expects a teleop config.
    teleop=SpaceMouseConfig(action_scale=[0.25, 0.25, 0.20, 0.50, 0.50, 0.50]),
)


def run_demo(max_steps: int = 2_000) -> None:
    """Run until the robot reaches START_POSE + [0, 0, 0.10, 0, 0, 0]."""
    net = ManipulationPrimitiveNet(net_cfg)
    transition = net.reset()

    print(f"start -> {net.active_primitive}")

    for _step in range(max_steps):
        loop_t0 = time.perf_counter()
        action = torch.zeros(net.action_dim, dtype=torch.float32)
        transition = net.step(action)
        info = transition[TransitionKey.INFO]
        print(transition[TransitionKey.OBSERVATION])
        # print(
        #     f"[{net.active_primitive}] "
        #     f"primitive_step={info.get('primitive_step', 0):04d} "
        #     f"reason={info.get('transition_reason')} "
        #     f"target={info.get('primitive_target_pose')}"
        # )

        # if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
        #     print("Reached the final primitive. Call net.reset() to run the sequence again.")
        #     break

        dt = time.perf_counter() - loop_t0
        precise_sleep(1 / net.config.fps - dt)


if __name__ == "__main__":
    run_demo()
