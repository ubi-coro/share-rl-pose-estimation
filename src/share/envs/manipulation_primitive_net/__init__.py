"""Lazy exports for manipulation-primitive-net modules."""

from __future__ import annotations

__all__ = ["ManipulationPrimitiveNetConfig", "ManipulationPrimitiveNet"]


def __getattr__(name: str):
    if name == "ManipulationPrimitiveNetConfig":
        from .config_manipulation_primitive_net import ManipulationPrimitiveNetConfig

        return ManipulationPrimitiveNetConfig

    if name == "ManipulationPrimitiveNet":
        from .env_manipulation_primitive_net import ManipulationPrimitiveNet

        return ManipulationPrimitiveNet

    raise AttributeError(name)
