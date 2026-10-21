from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Observation:
    tool_call_id: str | None
    tool_name: str
    text: str
    is_error: bool
    raw: dict[str, Any]


@dataclass(slots=True)
class Action:
    tool_call_id: str | None
    tool_name: str
    kind: str
    summary: str
    detail: str
    path: str | None
    raw: dict[str, Any]
    observations: list[Observation] = field(default_factory=list)


@dataclass(slots=True)
class Turn:
    index: int
    actions: list[Action]
    unmatched_observations: list[Observation]
    reward: float | int | bool | None = None
    cumulative_reward: float | int | bool | None = None
    reward_info: dict[str, Any] = field(default_factory=dict)
    completion_id: str | None = None


@dataclass(slots=True)
class Trajectory:
    path: Path
    trajectory_id: str
    task: str
    turns: list[Turn]
    reward: float | int | bool | None
    reward_components: dict[str, float]
    is_finished: bool
    finish_message: str
    error_message: str
    parse_warnings: list[str] = field(default_factory=list)
    terminal_reward_info: dict[str, Any] = field(default_factory=dict)
    task_id: str = ""

    @property
    def completion_count(self) -> int | None:
        """Count distinct recorded LLM completions, independent of tool/agent types.

        Without completion metadata, action/step counts cannot establish how
        many LLM calls occurred. Return unknown rather than infer from tools.
        """
        ids = {turn.completion_id for turn in self.turns if turn.completion_id}
        return len(ids) if ids else None

    @property
    def action_count(self) -> int:
        return sum(len(turn.actions) for turn in self.turns)

    @property
    def has_error(self) -> bool:
        return bool(self.error_message) or any(
            observation.is_error
            for turn in self.turns
            for action in turn.actions
            for observation in action.observations
        ) or any(
            observation.is_error
            for turn in self.turns
            for observation in turn.unmatched_observations
        )
