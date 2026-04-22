from dataclasses import dataclass

from lerobot.robots import RobotConfig

from share.robots.viperx.lerobot_robot_viperx.config_viperx import ViperXConfig


@RobotConfig.register_subclass("mock_viperx")
@dataclass
class MockViperXConfig(ViperXConfig):
    """Headless ViperX config for development without hardware."""

    port: str = "mock"
