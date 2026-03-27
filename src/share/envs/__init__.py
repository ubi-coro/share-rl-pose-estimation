"""Environment package for MP-Net components.

Keep this module import-light so low-level utilities like ``TaskFrame`` remain
usable during upstream ``lerobot`` initialization without circular imports.
"""

from __future__ import annotations

__all__ = [
    "ManipulationPrimitiveConfig",
    "ManipulationPrimitiveProcessorConfig",
    "ManipulationPrimitiveConfig",
    "MoveDeltaPrimitiveConfig",
    "OpenLoopTrajectoryPrimitiveConfig",
    "ManipulationPrimitive",
    "TaskFrame",
    "TASK_FRAME_AXIS_NAMES",
    "ManipulationPrimitiveNetConfig",
    "ManipulationPrimitiveNet",
]


def __getattr__(name: str):
    if name in {
        "ManipulationPrimitiveConfig",
        "ManipulationPrimitiveProcessorConfig",
        "ManipulationPrimitiveConfig",
        "MoveDeltaPrimitiveConfig",
        "OpenLoopTrajectoryPrimitiveConfig",
        "ManipulationPrimitive",
        "TaskFrame",
        "TASK_FRAME_AXIS_NAMES",
    }:
        from .manipulation_primitive import __dict__ as primitive_exports

        return primitive_exports[name]

    if name in {"ManipulationPrimitiveNetConfig", "ManipulationPrimitiveNet"}:
        from .manipulation_primitive_net import __dict__ as net_exports

        return net_exports[name]

    raise AttributeError(name)
