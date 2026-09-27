from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajectory_visualizer.parser import (
    load_trajectory,
    resolve_trajectory_path,
    training_steps,
)


def write_events(path: Path, events: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def test_parallel_actions_are_paired_by_tool_call_id(tmp_path: Path) -> None:
    path = tmp_path / "events_parallel.jsonl"
    write_events(
        path,
        [
            {"type": "trajectory_created", "trajectory": {"id": "trajectory-1"}},
            {
                "type": "trajectory_step_added",
                "step_index": 1,
                "step": {
                    "action_events": [
                        {
                            "tool_call_id": "call-a",
                            "tool_name": "terminal",
                            "action": {
                                "kind": "TerminalAction",
                                "command": "pwd",
                            },
                        },
                        {
                            "tool_call_id": "call-b",
                            "tool_name": "file_editor",
                            "action": {
                                "kind": "FileEditorAction",
                                "command": "view",
                                "path": "/workspace/a.py",
                            },
                        },
                    ],
                    "observation_events": [
                        {
                            "tool_call_id": "call-b",
                            "tool_name": "file_editor",
                            "observation": {
                                "content": [{"type": "text", "text": "file"}],
                                "is_error": False,
                            },
                        },
                        {
                            "tool_call_id": "call-a",
                            "tool_name": "terminal",
                            "observation": {
                                "content": [{"type": "text", "text": "/workspace"}],
                                "is_error": False,
                            },
                        },
                    ],
                },
            },
            {
                "type": "trajectory_finished",
                "reward": 1.0,
                "finish_message": "done",
                "error_message": None,
            },
        ],
    )

    trajectory = load_trajectory(path)

    assert trajectory.reward == 1.0
    assert trajectory.is_finished
    assert trajectory.action_count == 2
    assert trajectory.turns[0].actions[0].observations[0].text == "/workspace"
    assert trajectory.turns[0].actions[1].observations[0].text == "file"
    assert not trajectory.turns[0].unmatched_observations


def test_reward_components_are_loaded_from_reward_misc(tmp_path: Path) -> None:
    path = tmp_path / "events_rewards.jsonl"
    write_events(
        path,
        [
            {"type": "trajectory_created", "trajectory": {"id": "trajectory-1"}},
            {
                "type": "trajectory_step_added",
                "step_index": 0,
                "step": {
                    "misc": {
                        "reward_misc": {
                            "binary_reward": 1,
                            "f2p_pass_fraction": 0.75,
                            "p2p_fail_fraction": 0.25,
                            "localization_reward": 0.625,
                            "evaluation_logs": "large diagnostic output",
                        }
                    }
                },
            },
            {"type": "trajectory_finished", "reward": 0.75},
        ],
    )

    trajectory = load_trajectory(path)

    assert trajectory.reward_components == {
        "binary_reward": 1.0,
        "f2p_pass_fraction": 0.75,
        "p2p_fail_fraction": 0.25,
        "localization_reward": 0.625,
    }


def test_unmatched_observations_and_finished_error_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "events_error.jsonl"
    write_events(
        path,
        [
            {
                "type": "trajectory_step_added",
                "step_index": 0,
                "trajectory_id": "t",
                "step": {
                    "action_events": None,
                    "observation_events": [
                        {
                            "tool_call_id": "missing",
                            "observation": {"content": "boom", "is_error": True},
                        }
                    ],
                },
            },
            {
                "type": "trajectory_finished",
                "trajectory_id": "t",
                "reward": 0,
                "error_message": "trajectory crashed",
            },
        ],
    )

    trajectory = load_trajectory(path)

    assert trajectory.has_error
    assert trajectory.error_message == "trajectory crashed"
    assert trajectory.turns[0].unmatched_observations[0].text == "boom"


def test_null_action_and_top_level_agent_error_are_rendered(tmp_path: Path) -> None:
    path = tmp_path / "events_missing_tool.jsonl"
    tool_call_id = "tinker-tool-call-1bf0d948-c6cf-4397-8a9a-2b374e8ce120"
    error = (
        "Tool 'grep' not found. Available: "
        "['terminal', 'file_editor', 'task_tracker', 'finish', 'think']"
    )
    write_events(
        path,
        [
            {
                "type": "trajectory_step_added",
                "trajectory_id": "t",
                "step_index": 3,
                "step": {
                    "action_events": [
                        {
                            "kind": "ActionEvent",
                            "action": None,
                            "tool_name": "grep",
                            "tool_call_id": tool_call_id,
                            "tool_call": {
                                "id": tool_call_id,
                                "name": "grep",
                                "arguments": json.dumps(
                                    {
                                        "command": (
                                            'grep -n "get_object" '
                                            "/workspace/moto/s3/responses.py"
                                        ),
                                        "security_risk": "LOW",
                                    }
                                ),
                            },
                        }
                    ],
                    "observation_events": [
                        {
                            "kind": "AgentErrorEvent",
                            "tool_name": "grep",
                            "tool_call_id": tool_call_id,
                            "error": error,
                        }
                    ],
                },
            }
        ],
    )

    trajectory = load_trajectory(path)
    action = trajectory.turns[0].actions[0]
    observation = action.observations[0]

    assert action.kind == "ToolCallAction"
    assert 'grep -n "get_object"' in action.detail
    assert observation.text == error
    assert observation.is_error
    assert trajectory.has_error


def test_conversation_state_update_events_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "events_with_state_updates.jsonl"
    write_events(
        path,
        [
            {
                "type": "trajectory_step_added",
                "trajectory_id": "t",
                "step_index": 0,
                "step": {
                    "action_events": [
                        {
                            "kind": "ConversationStateUpdateEvent",
                            "key": "execution_status",
                            "value": "running",
                        }
                    ],
                    "observation_events": [
                        {
                            "kind": "ConversationStateUpdateEvent",
                            "key": "stats",
                            "value": {"tokens": 100},
                        },
                        {
                            "kind": "SystemPromptEvent",
                            "content": "You are an agent.",
                        },
                    ],
                },
            }
        ],
    )

    trajectory = load_trajectory(path)

    assert trajectory.action_count == 0
    assert not trajectory.is_finished
    assert len(trajectory.turns[0].unmatched_observations) == 1
    assert trajectory.turns[0].unmatched_observations[0].text == "You are an agent."


def test_steps_sort_numerically_and_path_is_confined(tmp_path: Path) -> None:
    for name in ("10", "2", "not-a-step"):
        (tmp_path / name / "events").mkdir(parents=True)
    allowed = tmp_path / "2" / "events" / "sample.jsonl"
    allowed.write_text("", encoding="utf-8")
    outside = tmp_path / "outside.jsonl"
    outside.write_text("", encoding="utf-8")

    assert [path.name for path in training_steps(tmp_path)] == ["2", "10"]
    assert resolve_trajectory_path(tmp_path / "2", "sample.jsonl") == allowed
    with pytest.raises(FileNotFoundError):
        resolve_trajectory_path(tmp_path / "2", str(outside))


@pytest.mark.parametrize("terminal_metadata", [False, True])
def test_rebench_nested_rewards_preserve_binary_total(tmp_path, terminal_metadata):
    path = tmp_path / "rebench.jsonl"
    metrics = {
        "localization": {
            "file_f1_reward": 1.0,
            "module_f1_reward": 0.5,
            "entity_f1_reward": 0.0,
            "localization_reward": 0.5,
        },
        "binary_reward": 0.0,
        "composite_reward": 0.3,
        "tool_json_error_reward": 1.0,
        "agent_error_event_reward": 0.0,
        "str_replace_reward": 0.5,
    }
    record = (
        {"type": "trajectory_finished", "reward": 0, "reward_misc": metrics}
        if terminal_metadata else
        {"type": "trajectory_step_added", "step": {"misc": {"reward_misc": metrics}}}
    )
    write_events(path, [
        {"type": "trajectory_created", "trajectory": {"id": "rebench"}},
        record,
        {"type": "trajectory_finished", "reward": 0},
    ])
    trajectory = load_trajectory(path)
    assert trajectory.reward == 0
    assert trajectory.reward_components["file_f1_reward"] == 1.0
    assert trajectory.reward_components["module_f1_reward"] == 0.5
    assert trajectory.reward_components["entity_f1_reward"] == 0.0
    assert trajectory.reward_components["tool_json_error_reward"] == 1.0
    assert trajectory.reward_components["tool_format_reward"] == 0.5


def test_explicit_tool_aggregate_is_preserved():
    from trajectory_visualizer.parser import _reward_components
    assert _reward_components({
        "tool_format_reward": 0.2,
        "tool_json_error_reward": 1.0,
        "agent_error_event_reward": 1.0,
        "str_replace_reward": 1.0,
    })["tool_format_reward"] == 0.2
