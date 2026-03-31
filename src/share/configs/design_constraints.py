from dataclasses import dataclass
from pathlib import Path

from lerobot.configs import parser

from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig


@dataclass
class DesignConstraintsConfig:
    """Live MP-Net workspace/origin calibration config."""

    env: ManipulationPrimitiveNetConfig
    output_path: Path | None = None
    autosave_on_primitive_change: bool = True
    play_sounds: bool = True

    def __post_init__(self):
        env_path = parser.get_path_arg("env")
        if self.output_path is None and env_path:
            self.output_path = Path(env_path)

        if self.output_path is None:
            raise ValueError(
                "DesignConstraintsConfig requires either --env.path=<mpnet.json> "
                "or an explicit output_path."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["env"]
