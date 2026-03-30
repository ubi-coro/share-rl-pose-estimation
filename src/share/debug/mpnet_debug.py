"""Async MP-Net debug logging for graph state, targets, and transitions."""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Lock, Thread
from typing import Any

from lerobot.processor import TransitionKey

from share.utils.transformation_utils import get_robot_pose_from_observation
from share.workspace.mpnet import summarize_mpnet_debug

LOGGER = logging.getLogger(__name__)

_NODE_COLOR = (67, 91, 110, 255)
_ACTIVE_NODE_COLOR = (255, 144, 61, 255)
_START_NODE_COLOR = (62, 136, 91, 255)
_RESET_NODE_COLOR = (63, 106, 196, 255)
_TERMINAL_NODE_COLOR = (180, 78, 71, 255)
_EDGE_COLOR = (143, 153, 168, 255)
_TRANSITION_EDGE_COLOR = (255, 144, 61, 255)
_CURRENT_POSE_COLOR = (69, 154, 255, 255)
_TARGET_POSE_COLOR = (255, 189, 46, 255)
_POSE_ERROR_COLOR = (255, 107, 70, 255)


@dataclass
class MPNetDebugConfig:
    """User-facing configuration for the async MP-Net debugger."""

    enabled: bool = False
    live_rerun: bool = True
    trace_path: str | Path | None = None
    queue_size: int = 512
    flush_interval_s: float = 0.25
    include_config_summary: bool = True

    def __post_init__(self) -> None:
        if self.queue_size <= 0:
            raise ValueError("MPNetDebugConfig.queue_size must be > 0.")
        if self.flush_interval_s <= 0.0:
            raise ValueError("MPNetDebugConfig.flush_interval_s must be > 0.")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "reshape"):
        try:
            return float(value.reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            pass
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _safe_float(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_pose(values: Any) -> list[float] | None:
    if values is None:
        return None
    try:
        pose = list(values)
    except TypeError:
        return None
    converted = [_safe_float(value) for value in pose[:6]]
    if any(value is None for value in converted):
        return None
    return [float(value) for value in converted]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    scalar = _safe_float(value)
    if scalar is not None:
        return scalar
    return str(value)


def _resolve_trace_path(
    config: MPNetDebugConfig,
    *,
    dataset_root: str | Path | None = None,
    timestamp: str | None = None,
) -> Path:
    if config.trace_path is not None:
        return Path(config.trace_path)
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path(dataset_root) if dataset_root is not None else Path("/tmp")
    return base_dir / "debug" / f"mpnet-debug-{timestamp}.jsonl"


def _graph_layout(summary: dict[str, Any]) -> dict[str, list[float]]:
    names = [primitive["name"] for primitive in summary.get("primitives", [])]
    if not names:
        return {}
    if len(names) == 1:
        return {names[0]: [0.0, 0.0]}
    layout: dict[str, list[float]] = {}
    count = len(names)
    for index, name in enumerate(names):
        angle = (2.0 * 3.141592653589793 * index) / count
        layout[name] = [round(1.25 * math.cos(angle), 6), round(1.25 * math.sin(angle), 6)]
    return layout


class MPNetDebugSession:
    """Background writer for MP-Net debug traces and optional Rerun logging."""

    def __init__(
        self,
        config: MPNetDebugConfig,
        mpnet_config: Any,
        *,
        trace_path: Path,
        display_ip: str | None = None,
        display_port: int | None = None,
        reuse_existing_rerun: bool = False,
        session_name: str = "mpnet_debug",
    ) -> None:
        self.config = config
        self.trace_path = trace_path
        self.display_ip = display_ip
        self.display_port = display_port
        self.reuse_existing_rerun = reuse_existing_rerun
        self.session_name = session_name
        self.recording_id = uuid.uuid4().hex[:12]
        self.started_at = time.time()
        self.summary = summarize_mpnet_debug(mpnet_config)
        self.summary_path = self.trace_path.with_name("config_summary.json")
        self._layout = _graph_layout(self.summary)
        self._primitive_lookup = {
            primitive["name"]: primitive for primitive in self.summary.get("primitives", [])
        }
        self._transitions_by_source: dict[str, list[dict[str, Any]]] = {}
        self._transitions_by_edge: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for transition in self.summary.get("transitions", []):
            self._transitions_by_source.setdefault(transition["source"], []).append(transition)
            self._transitions_by_edge.setdefault((transition["source"], transition["target"]), []).append(transition)

        self._queue: deque[dict[str, Any]] = deque()
        self._queue_lock = Lock()
        self._condition = Condition(self._queue_lock)
        self._closed = False
        self._event_index = 0
        self._dropped_events = 0
        self._thread = Thread(target=self._run, name="mpnet-debug", daemon=True)

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    def start(self) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.include_config_summary:
            self.summary_path.write_text(json.dumps(self.summary, indent=2) + "\n", encoding="utf-8")
        self._thread.start()
        self._enqueue_event(
            {
                "kind": "session_start",
                "timestamp": self.started_at,
                "recording_id": self.recording_id,
                "trace_path": str(self.trace_path),
                "config_summary_path": str(self.summary_path) if self.config.include_config_summary else None,
                "summary": {
                    "start_primitive": self.summary.get("start_primitive"),
                    "reset_primitive": self.summary.get("reset_primitive"),
                    "primitive_count": self.summary.get("primitive_count"),
                    "transition_count": self.summary.get("transition_count"),
                },
            }
        )

    def _enqueue_event(self, event: dict[str, Any]) -> None:
        with self._condition:
            if self._closed:
                return
            if len(self._queue) >= self.config.queue_size:
                self._queue.popleft()
                self._dropped_events += 1
            self._queue.append(event)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._queue.append(
                {
                    "kind": "session_end",
                    "timestamp": time.time(),
                    "recording_id": self.recording_id,
                    "dropped_events": self._dropped_events,
                    "session_seconds": time.time() - self.started_at,
                }
            )
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        trace_file = self.trace_path.open("a", encoding="utf-8")
        rerun = self._init_rerun()
        last_flush = time.time()
        try:
            while True:
                with self._condition:
                    if not self._queue and not self._closed:
                        self._condition.wait(timeout=self.config.flush_interval_s)
                    if self._queue:
                        event = self._queue.popleft()
                    elif self._closed:
                        break
                    else:
                        event = None
                if event is None:
                    if time.time() - last_flush >= self.config.flush_interval_s:
                        trace_file.flush()
                        last_flush = time.time()
                    continue
                self._write_event(trace_file, event)
                if rerun is not None:
                    self._log_rerun_event(rerun, event)
                if time.time() - last_flush >= self.config.flush_interval_s or event["kind"] == "session_end":
                    trace_file.flush()
                    last_flush = time.time()
        finally:
            trace_file.flush()
            trace_file.close()

    def _write_event(self, trace_file, event: dict[str, Any]) -> None:
        trace_file.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")

    def _init_rerun(self):
        if not self.config.live_rerun:
            return None
        try:
            import rerun as rr
        except ImportError:
            LOGGER.debug("Rerun is unavailable; MP-Net debug live view disabled.")
            return None

        try:
            if not self.reuse_existing_rerun:
                rr.init(self.session_name, recording_id=self.recording_id, spawn=False)
                if self.display_ip is not None or self.display_port is not None:
                    host = self.display_ip or "127.0.0.1"
                    port = self.display_port or 9876
                    rr.connect_grpc(f"{host}:{port}")
                else:
                    rr.spawn(port=self.display_port or 9876, connect=True)
            self._log_rerun_static(rr)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to initialize MP-Net debug Rerun stream: %s", exc)
            return None
        return rr

    def _log_rerun_static(self, rr) -> None:
        summary_md = [
            "# MP-Net",
            f"- start: `{self.summary.get('start_primitive')}`",
            f"- reset: `{self.summary.get('reset_primitive')}`",
            f"- primitives: {self.summary.get('primitive_count', 0)}",
            f"- transitions: {self.summary.get('transition_count', 0)}",
        ]
        rr.log(
            "mpnet/config/summary",
            rr.TextDocument("\n".join(summary_md), media_type="text/markdown"),
            static=True,
        )
        if self.config.include_config_summary:
            rr.log(
                "mpnet/config/summary_path",
                rr.TextDocument(str(self.summary_path), media_type="text/plain"),
                static=True,
            )

        edge_lines = []
        edge_labels = []
        for transition in self.summary.get("transitions", []):
            source = transition["source"]
            target = transition["target"]
            if source not in self._layout or target not in self._layout:
                continue
            edge_lines.append([self._layout[source], self._layout[target]])
            edge_labels.append(transition["condition_summary"])
        if edge_lines:
            rr.log(
                "mpnet/graph/edges",
                rr.LineStrips2D(edge_lines, colors=[_EDGE_COLOR] * len(edge_lines), labels=edge_labels),
                static=True,
            )

    def _log_rerun_event(self, rr, event: dict[str, Any]) -> None:
        self._event_index += 1
        rr.set_time_sequence("debug_step", self._event_index)
        rr.set_time_seconds("wall_time", float(event.get("timestamp", time.time())))

        kind = event["kind"]
        rr.log(
            "mpnet/status/log",
            rr.TextLog(self._status_log_line(event), level="INFO"),
        )
        rr.log(
            "mpnet/status/current",
            rr.TextDocument(self._status_markdown(event), media_type="text/markdown"),
        )
        rr.log("mpnet/metrics/dropped_events", rr.Scalars([float(self._dropped_events)]))
        rr.log(
            "mpnet/metrics/session_seconds",
            rr.Scalars([float(event.get("session_seconds", time.time() - self.started_at))]),
        )

        if kind in {"reset", "step", "transition"}:
            rr.log(
                "mpnet/metrics/episode_step",
                rr.Scalars([float(event.get("episode_step") or 0.0)]),
            )
            rr.log(
                "mpnet/metrics/primitive_step",
                rr.Scalars([float(event.get("primitive_step") or 0.0)]),
            )
            rr.log(
                "mpnet/metrics/trajectory_progress",
                rr.Scalars([float(event.get("trajectory_progress") or 0.0)]),
            )
            rr.log(
                "mpnet/metrics/primitive_complete",
                rr.Scalars([1.0 if event.get("primitive_complete") else 0.0]),
            )

        self._log_rerun_graph(rr, event)
        self._log_rerun_poses(rr, event)

    def _log_rerun_graph(self, rr, event: dict[str, Any]) -> None:
        positions = []
        colors = []
        labels = []
        active = event.get("active_primitive")
        for primitive in self.summary.get("primitives", []):
            name = primitive["name"]
            position = self._layout.get(name)
            if position is None:
                continue
            positions.append(position)
            labels.append(name)
            color = _NODE_COLOR
            if primitive.get("roles", {}).get("is_terminal", False):
                color = _TERMINAL_NODE_COLOR
            elif primitive.get("roles", {}).get("is_reset", False):
                color = _RESET_NODE_COLOR
            elif primitive.get("roles", {}).get("is_start", False):
                color = _START_NODE_COLOR
            if name == active:
                color = _ACTIVE_NODE_COLOR
            colors.append(color)

        if positions:
            rr.log(
                "mpnet/graph/nodes",
                rr.Points2D(positions, colors=colors, labels=labels, show_labels=True),
            )

        if event.get("kind") == "transition":
            source = event.get("transition_from")
            target = event.get("transition_to")
            if source in self._layout and target in self._layout:
                rr.log(
                    "mpnet/graph/last_transition",
                    rr.LineStrips2D(
                        [[self._layout[source], self._layout[target]]],
                        colors=[_TRANSITION_EDGE_COLOR],
                        labels=[event.get("transition_reason") or "transition"],
                    ),
                )
        else:
            rr.log("mpnet/graph/last_transition", rr.Clear(recursive=True))

    def _log_rerun_poses(self, rr, event: dict[str, Any]) -> None:
        rr.log("mpnet/poses", rr.Clear(recursive=True))
        for robot_name, pose_payload in event.get("robots", {}).items():
            current_pose = pose_payload.get("current_pose")
            target_pose = pose_payload.get("target_pose")
            if current_pose is not None:
                rr.log(
                    f"mpnet/poses/{robot_name}/current",
                    rr.Points3D([current_pose[:3]], colors=[_CURRENT_POSE_COLOR], labels=[f"{robot_name} current"]),
                )
            if target_pose is not None:
                rr.log(
                    f"mpnet/poses/{robot_name}/target",
                    rr.Points3D([target_pose[:3]], colors=[_TARGET_POSE_COLOR], labels=[f"{robot_name} target"]),
                )
            if current_pose is not None and target_pose is not None:
                rr.log(
                    f"mpnet/poses/{robot_name}/error",
                    rr.LineStrips3D(
                        [[current_pose[:3], target_pose[:3]]],
                        colors=[_POSE_ERROR_COLOR],
                        labels=[f"{robot_name} error"],
                    ),
                )

    def _status_log_line(self, event: dict[str, Any]) -> str:
        if event["kind"] == "session_start":
            return f"session_start recording_id={self.recording_id} trace={self.trace_path}"
        if event["kind"] == "session_end":
            return f"session_end dropped_events={self._dropped_events}"
        return (
            f"{event['kind']} active={event.get('active_primitive')} "
            f"{event.get('transition_from')}->{event.get('transition_to')} "
            f"reason={event.get('transition_reason')} "
            f"episode_step={event.get('episode_step')} primitive_step={event.get('primitive_step')}"
        )

    def _status_markdown(self, event: dict[str, Any]) -> str:
        if event["kind"] == "session_start":
            return (
                "# MP-Net Debug Session\n"
                f"- recording id: `{self.recording_id}`\n"
                f"- trace: `{self.trace_path}`\n"
                f"- dropped events: {self._dropped_events}\n"
            )

        active = event.get("active_primitive")
        primitive = self._primitive_lookup.get(active, {})
        outgoing = self._transitions_by_source.get(active, [])
        lines = [
            "# MP-Net State",
            f"- event: `{event['kind']}`",
            f"- active primitive: `{active}`",
            f"- transition: `{event.get('transition_from')} -> {event.get('transition_to')}`",
            f"- reason: `{event.get('transition_reason')}`",
            f"- episode step: {event.get('episode_step')}",
            f"- primitive step: {event.get('primitive_step')}",
            f"- trajectory progress: {event.get('trajectory_progress')}",
            f"- primitive complete: {bool(event.get('primitive_complete'))}",
            f"- dropped events: {self._dropped_events}",
        ]
        if primitive.get("notes"):
            lines.append("")
            lines.append(f"Notes: {primitive['notes']}")
        if outgoing:
            lines.append("")
            lines.append("Outgoing transitions:")
            for transition in outgoing:
                lines.append(
                    f"- `{transition['source']} -> {transition['target']}`: {transition['condition_summary']}"
                )
        if event["kind"] == "transition":
            fired = self._transitions_by_edge.get(
                (event.get("transition_from"), event.get("transition_to")),
                [],
            )
            if fired:
                lines.append("")
                lines.append("Fired edge details:")
                for transition in fired:
                    lines.append(f"- {transition['condition_summary']}")
        return "\n".join(lines)


class MPNetDebugger:
    """Thin wrapper around an optional background MP-Net debug session."""

    def __init__(self, session: MPNetDebugSession | None = None) -> None:
        self._session = session

    @classmethod
    def start(
        cls,
        config: MPNetDebugConfig | None,
        mp_net_config: Any,
        *,
        display_ip: str | None = None,
        display_port: int | None = None,
        dataset_root: str | Path | None = None,
        reuse_existing_rerun: bool = False,
        session_name: str = "mpnet_debug",
    ) -> "MPNetDebugger":
        if config is None or not config.enabled:
            return cls()
        trace_path = _resolve_trace_path(config, dataset_root=dataset_root)
        session = MPNetDebugSession(
            config=config,
            mpnet_config=mp_net_config,
            trace_path=trace_path,
            display_ip=display_ip,
            display_port=display_port,
            reuse_existing_rerun=reuse_existing_rerun,
            session_name=session_name,
        )
        session.start()
        return cls(session)

    def log_reset(self, mp_net: Any, transition: dict[str, Any]) -> None:
        if self._session is None:
            return
        self._session._enqueue_event(self._build_event("reset", mp_net, transition))

    def log_step(self, mp_net: Any, transition: dict[str, Any]) -> None:
        if self._session is None:
            return
        event = self._build_event("step", mp_net, transition)
        self._session._enqueue_event(event)
        if event.get("transition_to") != event.get("transition_from"):
            transition_event = dict(event)
            transition_event["kind"] = "transition"
            self._session._enqueue_event(transition_event)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()

    def _build_event(self, kind: str, mp_net: Any, transition: dict[str, Any]) -> dict[str, Any]:
        info = transition.get(TransitionKey.INFO, {}) or {}
        observation = transition.get(TransitionKey.OBSERVATION, {}) or {}
        transition_from = info.get("transition_from")
        transition_to = info.get("transition_to", transition_from)
        active = getattr(mp_net, "active_primitive", transition_to)
        robots = self._robot_debug_payload(observation=observation, info=info)

        return {
            "kind": kind,
            "timestamp": time.time(),
            "active_primitive": active,
            "primitive_step": _safe_float(info.get("primitive_step")),
            "episode_step": _safe_float(info.get("episode_step")),
            "transition_from": transition_from,
            "transition_to": transition_to,
            "transition_reason": info.get("transition_reason"),
            "primitive_complete": bool(info.get("primitive_complete", False)),
            "trajectory_progress": _safe_float(info.get("trajectory_progress")),
            "robots": robots,
            "session_seconds": time.time() - self._session.started_at if self._session is not None else 0.0,
        }

    def _robot_debug_payload(self, *, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        targets = info.get("primitive_target_pose", {}) or {}
        robot_payload: dict[str, Any] = {}
        for robot_name, target_pose_raw in sorted(targets.items()):
            target_pose = _to_pose(target_pose_raw)
            current_pose: list[float] | None = None
            try:
                current_pose = _to_pose(get_robot_pose_from_observation(observation, robot_name))
            except KeyError:
                current_pose = None
            pose_error = None
            if current_pose is not None and target_pose is not None:
                pose_error = [float(current_pose[idx] - target_pose[idx]) for idx in range(min(len(current_pose), len(target_pose)))]
            robot_payload[robot_name] = {
                "current_pose": current_pose,
                "target_pose": target_pose,
                "pose_error": pose_error,
            }
        return robot_payload
