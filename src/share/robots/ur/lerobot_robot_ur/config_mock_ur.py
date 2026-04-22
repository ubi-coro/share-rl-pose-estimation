from dataclasses import dataclass

from lerobot.robots import RobotConfig

from share.robots.ur.lerobot_robot_ur.config_ur import URConfig


@RobotConfig.register_subclass("mock_ur")
@dataclass
class MockURConfig(URConfig):
    """Headless UR config for development without hardware."""

    robot_ip: str = "mock"
