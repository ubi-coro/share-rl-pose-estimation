from enum import Enum
from lerobot.teleoperators.utils import TeleopEvents as BaseTeleopEvents


class TeleopEvents(Enum):
    SUCCESS = BaseTeleopEvents.SUCCESS.value
    FAILURE = BaseTeleopEvents.FAILURE.value
    RERECORD_EPISODE = BaseTeleopEvents.RERECORD_EPISODE.value
    IS_INTERVENTION = BaseTeleopEvents.IS_INTERVENTION.value
    TERMINATE_EPISODE = BaseTeleopEvents.TERMINATE_EPISODE.value

    INTERVENTION_COMPLETED = "intervention_completed"
    STOP_RECORDING = "stop_recording"
    PAUSE_RECORDING = "pause_recording"
    RESUME_RECORDING = "resume_recording"
