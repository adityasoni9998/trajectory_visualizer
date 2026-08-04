from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Iterable

from .model import Action, Observation, Trajectory, Turn


REWARD_COMPONENT_FIELDS = {
    "f2p_pass_fraction",
    "p2p_fail_fraction",
    "p2p_pass_fraction",
}


class TrajectoryParseError(ValueError):
    """Raised when a JSONL file cannot be interpreted as a trajectory."""


def training_steps(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Rollout root does not exist: {root}")
    steps = [
        child
        for child in root.iterdir()
        if child.is_dir() and child.name.isdigit() and (child / "events").is_dir()
    ]
    return sorted(steps, key=lambda path: int(path.name))


def trajectory_files(step_dir: Path) -> list[Path]:
    event_dir = step_dir / "events"
    return sorted(event_dir.glob("*.jsonl")) if event_dir.is_dir() else []


def resolve_trajectory_path(step_dir: Path, entered: str) -> Path:
    """Resolve a user-entered trajectory path, constrained to the selected step."""
    value = Path(entered.strip()).expanduser()
    candidates = [value]
    if not value.is_absolute():
        candidates.extend((step_dir / value, step_dir / "events" / value))

    events_root = (step_dir / "events").resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(events_root)
        except (FileNotFoundError, ValueError):
            continue
        if resolved.is_file() and resolved.suffix == ".jsonl":
            return resolved
    raise FileNotFoundError(
        f"Not a JSONL trajectory inside training step {step_dir.name}: {entered}"
    )


def _text_parts(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        content = value.get("content")
        if content is not None:
            return _text_parts(content)
        return json.dumps(value, indent=2, ensure_ascii=False)
    return str(value)


def _observation(event: dict[str, Any]) -> Observation:
    body_value = event.get("observation")
    body = body_value if isinstance(body_value, dict) else {}
    text = _text_parts(body.get("content"))
    if not text and body:
        text = _text_parts(body)
    if not text:
        text = _text_parts(
            event.get("error") or event.get("message") or event.get("content")
        )
    return Observation(
        tool_call_id=event.get("tool_call_id"),
        tool_name=str(event.get("tool_name") or "environment"),
        text=text,
        is_error=bool(
            body.get("is_error")
            or event.get("is_error")
            or event.get("error")
            or event.get("kind") == "AgentErrorEvent"
        ),
        raw=event,
    )


def _compact_command(command: str) -> str:
    first_line = command.strip().splitlines()[0] if command.strip() else "(empty command)"
    return first_line


def _action(event: dict[str, Any]) -> Action:
    body_value = event.get("action")
    body = body_value if isinstance(body_value, dict) else {}
    tool_call_value = event.get("tool_call")
    tool_call = tool_call_value if isinstance(tool_call_value, dict) else {}
    arguments_value = tool_call.get("arguments")
    if isinstance(arguments_value, str):
        try:
            parsed_arguments = json.loads(arguments_value)
            arguments = (
                parsed_arguments if isinstance(parsed_arguments, dict) else {}
            )
        except json.JSONDecodeError:
            arguments = {"arguments": arguments_value}
    elif isinstance(arguments_value, dict):
        arguments = arguments_value
    else:
        arguments = {}

    tool = str(
        event.get("tool_name")
        or tool_call.get("name")
        or body.get("kind", "tool").removesuffix("Action").lower()
    )
    kind = str(body.get("kind") or ("ToolCallAction" if tool_call else tool))
    path = body.get("path") or arguments.get("path")

    if not body and tool_call:
        command = str(arguments.get("command") or "")
        if command:
            summary = f"{tool}: {_compact_command(command)}"
            extra_arguments = {
                key: value for key, value in arguments.items() if key != "command"
            }
            detail = command
            if extra_arguments:
                detail += "\n\n" + json.dumps(
                    extra_arguments, indent=2, ensure_ascii=False
                )
        else:
            summary = f"{tool} tool call"
            detail = json.dumps(arguments, indent=2, ensure_ascii=False)
    elif kind == "TerminalAction":
        command = str(body.get("command") or "")
        summary = f"$ {_compact_command(command)}"
        detail = command
    elif kind == "FileEditorAction":
        command = str(body.get("command") or "edit")
        path_text = str(path or "(no path)")
        ranges = body.get("view_range")
        suffix = f" lines {ranges}" if ranges else ""
        summary = f"{command} {path_text}{suffix}"
        detail_bits = [f"command: {command}", f"path: {path_text}"]
        for key in ("view_range", "old_str", "new_str", "insert_line"):
            if body.get(key) is not None:
                detail_bits.append(f"{key}: {body[key]}")
        detail = "\n".join(detail_bits)
    elif kind == "ThinkAction":
        thought = str(body.get("thought") or body.get("summary") or "")
        summary = thought.strip().splitlines()[0] if thought.strip() else "Thinking"
        detail = thought
    elif kind == "FinishAction":
        message = str(body.get("message") or body.get("final") or "")
        summary = message.strip().splitlines()[0] if message.strip() else "Finish"
        detail = message
    elif kind == "TaskTrackerAction":
        command = str(body.get("command") or "update")
        summary = f"task tracker: {command}"
        detail = json.dumps(body, indent=2, ensure_ascii=False)
    else:
        summary = str(event.get("summary") or kind)
        detail = json.dumps(body, indent=2, ensure_ascii=False)

    return Action(
        tool_call_id=event.get("tool_call_id"),
        tool_name=tool,
        kind=kind,
        summary=summary,
        detail=detail,
        path=str(path) if path else None,
        raw=event,
    )


def _pair_events(
    action_events: Iterable[dict[str, Any]],
    observation_events: Iterable[dict[str, Any]],
) -> tuple[list[Action], list[Observation]]:
    actions = [
        _action(event)
        for event in action_events
        if event.get("kind") != "ConversationStateUpdateEvent"
    ]
    observations = [
        _observation(event)
        for event in observation_events
        if event.get("kind") != "ConversationStateUpdateEvent"
    ]
    by_call: dict[str, list[Action]] = {}
    for action in actions:
        if action.tool_call_id:
            by_call.setdefault(action.tool_call_id, []).append(action)

    unmatched: list[Observation] = []
    for observation in observations:
        matches = by_call.get(observation.tool_call_id or "")
        if matches:
            # A tool_call_id normally maps to one action. Keeping a list avoids
            # dropping data if an unusual log duplicates the identifier.
            matches[-1].observations.append(observation)
        else:
            unmatched.append(observation)
    return actions, unmatched


def load_trajectory(path: Path) -> Trajectory:
    task = ""
    trajectory_id = ""
    turns: list[Turn] = []
    reward: float | int | bool | None = None
    reward_components: dict[str, float] = {}
    is_finished = False
    finish_message = ""
    error_message = ""
    warnings: list[str] = []

    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise TrajectoryParseError(str(exc)) from exc

    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue

            event_type = event.get("type")
            trajectory_id = str(event.get("trajectory_id") or trajectory_id)
            if event_type == "trajectory_created":
                trajectory = event.get("trajectory") or {}
                trajectory_id = str(trajectory.get("id") or trajectory_id)
            elif event_type == "trajectory_task_set":
                task_data = event.get("task")
                task = (
                    str(task_data.get("goal") or task_data.get("description") or "")
                    if isinstance(task_data, dict)
                    else str(task_data or "")
                )
            elif event_type == "trajectory_step_added":
                step = event.get("step") or {}
                actions, unmatched = _pair_events(
                    step.get("action_events") or [],
                    step.get("observation_events") or [],
                )
                # Step zero often contains only the initial environment prompt.
                # It remains visible because it is still part of what the agent saw.
                turns.append(
                    Turn(
                        index=int(event.get("step_index", len(turns))),
                        actions=actions,
                        unmatched_observations=unmatched,
                    )
                )
                if event.get("reward") is not None:
                    reward = event["reward"]
                reward_misc = (step.get("misc") or {}).get("reward_misc") or {}
                if isinstance(reward_misc, dict):
                    components = {
                        str(name): float(value)
                        for name, value in reward_misc.items()
                        if (
                            str(name).endswith("_reward")
                            or str(name) in REWARD_COMPONENT_FIELDS
                        )
                        and isinstance(value, (int, float, bool))
                    }
                    if components:
                        reward_components = components
                finish_message = str(event.get("finish_message") or finish_message)
                error_message = str(event.get("error_message") or error_message)
            elif event_type == "trajectory_finished":
                is_finished = True
                reward = event.get("reward", reward)
                reward_misc = event.get("reward_misc") or {}
                if isinstance(reward_misc, dict):
                    reward_components.update(
                        {
                            str(name): float(value)
                            for name, value in reward_misc.items()
                            if (
                                str(name).endswith("_reward")
                                or str(name) in REWARD_COMPONENT_FIELDS
                            )
                            and isinstance(value, (int, float, bool))
                        }
                    )
                finish_message = str(event.get("finish_message") or finish_message)
                error_message = str(event.get("error_message") or error_message)

    if not trajectory_id and not turns:
        raise TrajectoryParseError(f"No trajectory events found in {path}")
    return Trajectory(
        path=path,
        trajectory_id=trajectory_id,
        task=task,
        turns=turns,
        reward=reward,
        reward_components=reward_components,
        is_finished=is_finished,
        finish_message=finish_message,
        error_message=error_message,
        parse_warnings=warnings,
    )


def shell_quoted(path: Path) -> str:
    return shlex.quote(str(path))
