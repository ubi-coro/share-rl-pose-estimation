import types
from typing import get_origin, Union, get_args

from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.processor.hil_processor import GRIPPER_KEY
from lerobot.utils.constants import REWARD, DONE


DELTA_TRANSLATION_ACTION_NAMES = {
    "delta_x",
    "delta_y",
    "delta_z",
    "x.vel",
    "y.vel",
    "z.vel",
}
DELTA_ROTATION_ACTION_NAMES = {
    "delta_rx",
    "delta_ry",
    "delta_rz",
    "rx.vel",
    "ry.vel",
    "rz.vel",
}
DELTA_AUXILIARY_ACTION_NAMES = {
    GRIPPER_KEY,
    f"{GRIPPER_KEY}.pos",
}
DELTA_ACTION_NAMES = (
    DELTA_TRANSLATION_ACTION_NAMES |
    DELTA_ROTATION_ACTION_NAMES |
    DELTA_AUXILIARY_ACTION_NAMES
)


def check_task_frame_robot(robot_dict: dict[str, "Robot"]):
    is_task_frame_robot = {}
    for name, r in robot_dict.items():
        is_task_frame_robot[name] = hasattr(r, "set_task_frame")

    return is_task_frame_robot


def get_teleoperator_action_names(teleoperator: "Teleoperator") -> set[str]:
    action_features = getattr(teleoperator, "action_features", {})
    if not isinstance(action_features, dict):
        return set()

    feature_names = action_features.get("names")
    if isinstance(feature_names, dict):
        return {str(name) for name in feature_names}

    return {str(name) for name in action_features}


def check_delta_teleoperator(teleop_dict: dict[str, "Teleoperator"]):
    is_delta_teleoperator = {}
    for name, t in teleop_dict.items():
        action_names = get_teleoperator_action_names(t)
        is_delta_teleoperator[name] = (
            bool(action_names) and
            action_names.issubset(DELTA_ACTION_NAMES) and
            bool(action_names & (DELTA_TRANSLATION_ACTION_NAMES | DELTA_ROTATION_ACTION_NAMES))
        )

    return is_delta_teleoperator


def is_union_with_dict(field_type) -> bool:
    origin = get_origin(field_type)
    if origin is types.UnionType or origin is Union:
        return any(get_origin(arg) is dict for arg in get_args(field_type))
    return False


def env_to_dataset_features(env_features: dict[str, PolicyFeature]) -> dict:
    ds_features = {}
    for key, ft in env_features.items():
        new_ft = {"shape": ft.shape}
        if ft.type == FeatureType.VISUAL:
            new_ft["dtype"] = "video"
            new_ft["names"] = ["channels", "height", "width"]
        else:
            new_ft["dtype"] = "float32"
            new_ft["names"] = None
        ds_features[key] = new_ft

    ds_features[REWARD] = {"dtype": "float32", "shape": (1,), "names": None}
    ds_features[DONE] = {"dtype": "bool", "shape": (1,), "names": None}
    return ds_features
