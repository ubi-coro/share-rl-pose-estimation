from __future__ import annotations

from types import SimpleNamespace

import pytest

from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from share.scripts.design_constraints import WorkspaceConstraintDesigner


class _FakeEnv:
    def __init__(self, observation: dict[str, float], frame: TaskFrame):
        self._observation = observation
        self.task_frame = {"arm": frame}
        self.apply_calls = 0

    def _get_observation(self) -> dict[str, float]:
        return dict(self._observation)

    def apply_task_frames(self) -> None:
        self.apply_calls += 1


class _FakeMPNet:
    def __init__(self, primitive: ManipulationPrimitiveConfig, env: _FakeEnv):
        self.active_primitive = "pick"
        self.config = SimpleNamespace(primitives={"pick": primitive})
        self._envs = {"pick": env}


def _adaptive_frame() -> TaskFrame:
    return TaskFrame(
        target=[0.0] * 6,
        space=ControlSpace.TASK,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.RELATIVE] * 6,
        origin=[0.0] * 6,
    )


def _observation(x: float, y: float, z: float) -> dict[str, float]:
    return {
        "arm.x.ee_pos": x,
        "arm.y.ee_pos": y,
        "arm.z.ee_pos": z,
        "arm.rx.ee_pos": 0.0,
        "arm.ry.ee_pos": 0.0,
        "arm.rz.ee_pos": 0.0,
    }


def test_set_origin_from_current_pose_resets_tracked_bounds_without_enforcing_them(tmp_path):
    primitive_frame = _adaptive_frame()
    env_frame = _adaptive_frame()
    primitive = ManipulationPrimitiveConfig(task_frame={"arm": primitive_frame})
    env = _FakeEnv(_observation(0.5, -0.2, 0.3), env_frame)
    designer = WorkspaceConstraintDesigner(_FakeMPNet(primitive, env), tmp_path / "mpnet.json")

    designer.set_origin_from_current_pose()
    status = designer.status_summary("pick")

    assert primitive.task_frame["arm"].origin == pytest.approx([0.5, -0.2, 0.3, 0.0, 0.0, 0.0])
    assert primitive.task_frame["arm"].min_pose == [-float("inf")] * 6
    assert primitive.task_frame["arm"].max_pose == [float("inf")] * 6
    assert status["arm"]["tracked_min_pose"] == pytest.approx([0.0] * 6)
    assert status["arm"]["tracked_max_pose"] == pytest.approx([0.0] * 6)
    assert status["arm"]["live_bounds_enforced"] is False
    assert env.apply_calls == 1


def test_update_bounds_tracks_pose_extrema_without_live_enforcement(tmp_path):
    primitive = ManipulationPrimitiveConfig(task_frame={"arm": _adaptive_frame()})
    env = _FakeEnv(_observation(0.0, 0.0, 0.0), _adaptive_frame())
    designer = WorkspaceConstraintDesigner(_FakeMPNet(primitive, env), tmp_path / "mpnet.json")

    designer.set_origin_from_current_pose()

    env._observation = _observation(0.1, -0.2, 0.3)
    designer.update_bounds()
    env._observation = _observation(-0.4, 0.5, 0.1)
    designer.update_bounds()
    status = designer.status_summary("pick")

    assert status["arm"]["tracked_min_pose"] == pytest.approx([-0.4, -0.2, 0.0, 0.0, 0.0, 0.0])
    assert status["arm"]["tracked_max_pose"] == pytest.approx([0.1, 0.5, 0.3, 0.0, 0.0, 0.0])
    assert primitive.task_frame["arm"].min_pose == [-float("inf")] * 6
    assert primitive.task_frame["arm"].max_pose == [float("inf")] * 6


def test_toggle_live_enforcement_applies_tracked_bounds_to_live_frame(tmp_path):
    primitive = ManipulationPrimitiveConfig(task_frame={"arm": _adaptive_frame()})
    env = _FakeEnv(_observation(0.0, 0.0, 0.0), _adaptive_frame())
    designer = WorkspaceConstraintDesigner(_FakeMPNet(primitive, env), tmp_path / "mpnet.json")

    designer.set_origin_from_current_pose()
    env._observation = _observation(0.2, -0.1, 0.3)
    designer.update_bounds()

    designer.toggle_live_enforcement()
    status = designer.status_summary("pick")

    assert status["arm"]["live_bounds_enforced"] is True
    assert primitive.task_frame["arm"].min_pose == pytest.approx([0.0, -0.1, 0.0, 0.0, 0.0, 0.0])
    assert primitive.task_frame["arm"].max_pose == pytest.approx([0.2, 0.0, 0.3, 0.0, 0.0, 0.0])
    assert env.apply_calls == 2
