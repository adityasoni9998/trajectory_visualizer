from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.widgets import DataTable, RichLog, Static

from trajectory_visualizer.app import TrajectoryApp, _all_files_reward_text
from trajectory_visualizer.parser import _reward_components, load_trajectory, training_steps


def write_events(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def step(index: int, reward: float, cumulative: float, info: dict) -> dict:
    return {
        "type": "trajectory_step_added", "trajectory_id": "t", "step_index": index,
        "reward": cumulative,
        "step": {"reward": reward, "misc": {"reward_misc": info}},
    }


def test_unknown_nested_metrics_and_non_numeric_metadata():
    assert _reward_components({
        "new_score": 0.0, "nested": {"score": -0.5},
        "nested/score": 3, "~key": 2, "flag": True, "missing": None,
        "nan": float("nan"), "inf": float("inf"), "huge": 10**400,
        "logs": "text", "array": [1, 2],
    }) == {"new_score": 0.0, "nested/score": -0.5, "nested~1score": 3.0, "~0key": 2.0}


def test_step_rewards_sparse_diagnostics_and_explicit_sums(tmp_path):
    path = tmp_path / "events.jsonl"
    write_events(path, [
        step(0, 0, 0, {}),
        step(1, 0.2, 0.2, {"quality": {"score": 0.8}, "gain": 0.2, "penalty": 0}),
        step(2, -0.05, 0.15, {"gain": 0, "penalty": -0.05, "patch_hash": "abc"}),
        {"type": "trajectory_finished", "reward": 0.15,
         "reward_misc": {"gain": 0, "penalty": -0.05, "final_score": 0.9}},
    ])
    trajectory = load_trajectory(path, sum_metrics=("gain", "penalty"))
    assert [turn.reward for turn in trajectory.turns] == [0, 0.2, -0.05]
    assert [turn.cumulative_reward for turn in trajectory.turns] == [0, 0.2, 0.15]
    assert trajectory.reward == 0.15
    assert trajectory.reward_components == {
        "quality/score": 0.8, "gain": 0, "penalty": -0.05, "final_score": 0.9,
        "sum:gain": 0.2, "sum:penalty": -0.05,
    }
    assert trajectory.turns[-1].reward_info["patch_hash"] == "abc"
    assert trajectory.terminal_reward_info["final_score"] == 0.9
    assert "sum:gain" not in load_trajectory(path).reward_components


def test_partial_jsonl_recovers_and_discovers_new_metrics(tmp_path):
    path = tmp_path / "events.jsonl"
    write_events(path, [step(1, 0, 0, {"first": 1})])
    new_record = json.dumps(step(2, -0.1, -0.1, {"new_metric": -0.1}))
    with path.open("a") as handle:
        handle.write(new_record[:20])
    assert load_trajectory(path).parse_warnings
    with path.open("a") as handle:
        handle.write(new_record[20:] + "\n")
    trajectory = load_trajectory(path)
    assert not trajectory.parse_warnings
    assert trajectory.reward_components == {"first": 1, "new_metric": -0.1}


def test_common_denominator_and_observed_coverage(tmp_path):
    rows = [
        (tmp_path / "a.jsonl", (1, "", 1, "a", {"novel": 0.6}, True)),
        (tmp_path / "b.jsonl", (0, "", 1, "b", {}, True)),
        (tmp_path / "c.jsonl", (None, "", 1, "c", {"novel": 10}, False)),
    ]
    summary = _all_files_reward_text(rows).plain
    assert "ALL FILES AVG  0.50" in summary
    assert "novel  0.300  (present 1/2)" in summary
    assert "missing = 0" in summary


@pytest.mark.asyncio
async def test_reward_only_steps_render_and_events_directory_opens(tmp_path):
    events = tmp_path / "0" / "events"
    events.mkdir(parents=True)
    path = events / "events_sample.jsonl"
    write_events(path, [
        step(1, 0.2, 0.2, {"gain": 0.2, "note": "initial diagnostic"}),
        step(2, -0.05, 0.15, {"gain": -0.05, "note": "negative diagnostic"}),
        {"type": "trajectory_finished", "trajectory_id": "t", "reward": 0.15},
    ])
    assert training_steps(events) == [events.parent]
    assert training_steps(events.parent) == [events.parent]
    app = TrajectoryApp(events, refresh_interval=0, sum_metrics=("gain",))
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        assert app.query_one(DataTable).row_count == 2
        assert "sum:gain  0.150" in str(app.query_one("#reward-summary", Static).render())
        await pilot.press("enter")
        await pilot.pause()
        log = app.query_one(RichLog)
        text = "\n".join(line.text for line in log.lines)
        assert "STEP 2" in text
        assert "step reward=-0.050" in text
        assert "cumulative=0.150" in text
        assert "gain -0.050" in text
        assert "negative diagnostic" not in text
        await pilot.press("i")
        await pilot.pause()
        assert "negative diagnostic" in "\n".join(line.text for line in log.lines)
        await pilot.press("b")
        await pilot.pause()
        assert not app.detail_mode
        assert app.query_one("#reward-summary-scroll").display
