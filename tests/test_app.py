from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajectory_visualizer.app import (
    DEFAULT_REFRESH_INTERVAL,
    TrajectoryApp,
    _average_reward,
    _average_reward_components,
    build_parser,
)
from textual.widgets import DataTable, RichLog, Static


def make_rollout(root: Path, step: int, error_message: str | None = None) -> None:
    events = root / str(step) / "events"
    events.mkdir(parents=True)
    path = events / f"events_step_{step}.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "trajectory_created",
                "trajectory": {"id": f"trajectory-{step}"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "trajectory_finished",
                "trajectory_id": f"trajectory-{step}",
                "reward": step % 2,
                "error_message": error_message,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_all_reward_averages_use_finished_numeric_reward_denominator(
    tmp_path: Path,
) -> None:
    rows = [
        (
            tmp_path / "rewarded.jsonl",
            (
                1.0,
                "",
                0,
                "rewarded.jsonl",
                {
                    "binary_reward": 1.0,
                    "str_replace_reward": 0.6,
                },
            ),
        ),
        (tmp_path / "failed.jsonl", (0.0, "", 0, "failed.jsonl", {})),
        (tmp_path / "partial.jsonl", (0.5, "", 0, "partial.jsonl", {})),
        (tmp_path / "unfinished.jsonl", (None, "", 0, "unfinished.jsonl", {})),
    ]

    assert _average_reward(rows) == pytest.approx(0.5)
    components = _average_reward_components(rows)
    assert components["binary_reward"] == pytest.approx(1 / 3)
    assert components["str_replace_reward"] == pytest.approx(0.2)
    assert "localization_reward" not in components


@pytest.mark.asyncio
async def test_clicking_training_step_changes_selection(tmp_path: Path) -> None:
    make_rollout(tmp_path, 0)
    make_rollout(tmp_path, 1)
    app = TrajectoryApp(tmp_path, initial_step=0)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        assert app.steps[app.selected_step_index].name == "0"

        # The second list row is step 1. Clicking it exercises Textual's own
        # ListView.Selected event and guards against custom-message shadowing.
        await pilot.click("#steps", offset=(2, 1))
        await pilot.pause()

        assert app.steps[app.selected_step_index].name == "1"


@pytest.mark.asyncio
async def test_list_detail_and_back_are_separate_views(tmp_path: Path) -> None:
    make_rollout(tmp_path, 0)
    app = TrajectoryApp(tmp_path, initial_step=0)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        table = app.query_one("#trajectories", DataTable)
        timeline = app.query_one("#timeline", RichLog)

        assert not app.detail_mode
        assert table.styles.display != "none"
        assert timeline.styles.display == "none"
        assert table.size.height > 20

        await pilot.press("enter")
        await pilot.pause()

        assert app.detail_mode
        assert table.styles.display == "none"
        assert timeline.styles.display == "block"
        header_border = next(
            line for line in timeline.lines if "TRAJECTORY" in line.text
        )
        assert header_border.cell_length == timeline.scrollable_content_region.width

        await pilot.resize_terminal(140, 35)
        await pilot.pause()
        resized_border = next(
            line for line in timeline.lines if "TRAJECTORY" in line.text
        )
        assert resized_border.cell_length == timeline.scrollable_content_region.width
        assert resized_border.cell_length > header_border.cell_length

        await pilot.press("b")
        await pilot.pause()

        assert not app.detail_mode
        assert table.styles.display == "block"
        assert timeline.styles.display == "none"


@pytest.mark.asyncio
async def test_path_and_step_dialog_stylesheets_load(tmp_path: Path) -> None:
    make_rollout(tmp_path, 0)
    app = TrajectoryApp(tmp_path, initial_step=0)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()

        await pilot.press("p")
        await pilot.pause()
        assert len(app.screen_stack) == 2
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("g")
        await pilot.pause()
        assert len(app.screen_stack) == 2
        await pilot.press("escape")
        await pilot.pause()

        assert len(app.screen_stack) == 1


def test_table_metadata_contains_final_error_preview(tmp_path: Path) -> None:
    message = "worker failed while finalizing the trajectory"
    make_rollout(tmp_path, 0, error_message=message)
    path = next((tmp_path / "0" / "events").glob("*.jsonl"))

    metadata = TrajectoryApp._row_metadata(path)

    assert metadata[1] == message


def test_rollout_groups_sort_by_average_then_individual_reward(tmp_path: Path) -> None:
    events = tmp_path / "0" / "events"
    events.mkdir(parents=True)
    suffixes = [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]

    def add(group: str, rewards: list[float]) -> None:
        for suffix, reward in zip(suffixes, rewards, strict=True):
            path = events / f"events_{group}_{suffix}.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "type": "trajectory_created",
                        "trajectory": {"id": f"{group}-{suffix}"},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "trajectory_finished",
                        "reward": reward,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    add("high", [0.5, 0.9])
    add("low", [0.1, 0.3])
    add("middle", [0.4, 0.6])

    rows = TrajectoryApp._sorted_step_rows(sorted(events.glob("*.jsonl")))
    order = [(path.name.split("_")[-2], metadata[0]) for path, metadata in rows]

    assert order == [
        ("high", 0.9),
        ("high", 0.5),
        ("middle", 0.6),
        ("middle", 0.4),
        ("low", 0.3),
        ("low", 0.1),
    ]


@pytest.mark.asyncio
async def test_group_separator_shows_average_total_reward(tmp_path: Path) -> None:
    events = tmp_path / "0" / "events"
    events.mkdir(parents=True)
    for index, reward in enumerate((0.25, 0.75), 1):
        path = events / (
            "events_dense_"
            f"00000000-0000-0000-0000-{index:012d}.jsonl"
        )
        path.write_text(
            json.dumps({"type": "trajectory_created", "trajectory": {"id": index}})
            + "\n"
            + json.dumps(
                {
                    "type": "trajectory_step_added",
                    "step_index": 0,
                    "step": {
                        "misc": {
                            "reward_misc": {
                                "binary_reward": float(index - 1),
                                "f2p_pass_fraction": reward,
                                "p2p_fail_fraction": 1.0 - reward,
                                "composite_reward": reward,
                                "file_f1_reward": reward,
                                "entity_f1_reward": reward,
                                "module_f1_reward": reward,
                                "localization_reward": reward,
                                "file_editor_reward": reward,
                                "agent_error_event_reward": 1.0,
                                "tool_format_reward": 0.2,
                                "tool_json_error_reward": 1.0,
                                "evaluation_logs": "not a reward",
                            }
                        }
                    },
                }
            )
            + "\n"
            + json.dumps({"type": "trajectory_finished", "reward": reward})
            + "\n",
            encoding="utf-8",
        )

    app = TrajectoryApp(tmp_path, initial_step=0, refresh_interval=0)
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        table = app.query_one("#trajectories", DataTable)
        reward_summary = str(app.query_one("#reward-summary", Static).render())
        separator = " ".join(str(value) for value in table.get_row_at(0))
        highest_reward_row = " ".join(
            str(value) for value in table.get_row_at(1)
        )

        assert "ALL FILES AVG  0.50" in reward_summary
        for name in ("composite_reward", "binary_reward", "localization_reward",
                     "file_f1_reward", "entity_f1_reward", "module_f1_reward",
                     "f2p_pass_fraction", "p2p_fail_fraction"):
            assert f"{name}  0.500  (present 2/2)" in reward_summary
        assert "tool_json_error_reward  1.000" in reward_summary
        assert "tool_format_reward  0.200" in reward_summary
        assert "avg total 0.50" in separator
        assert "composite_reward 0.500" in separator
        assert "evaluation_logs" not in separator
        assert "0.75" in highest_reward_row
        assert "composite_reward 0.750" in highest_reward_row
        assert "+8 more (open trajectory)" in highest_reward_row
        assert table.rows[table.coordinate_to_cell_key((1, 0)).row_key].height == 1
        assert table.rows[table.coordinate_to_cell_key((0, 0)).row_key].height == 1
        assert app.query_one("#reward-summary-scroll").size.height <= 12
        assert table.size.height > 0

        await pilot.press("enter")
        await pilot.pause()
        timeline = app.query_one("#timeline", RichLog)
        detail = "\n".join(line.text for line in timeline.lines)
        assert "TOTAL REWARD" in detail
        assert "f2p_pass_fraction" in detail
        assert "tool_json_error_reward" in detail
        assert "0.250" in detail
        assert "evaluation_logs" not in detail
        await pilot.press("i")
        await pilot.pause()
        detail = "\n".join(line.text for line in timeline.lines)
        assert "evaluation_logs" in detail



def test_refresh_interval_defaults_to_one_minute() -> None:
    args = build_parser().parse_args([])

    assert args.refresh_interval == DEFAULT_REFRESH_INTERVAL == 60


@pytest.mark.asyncio
async def test_refresh_discovers_new_rollout_file(tmp_path: Path) -> None:
    make_rollout(tmp_path, 0)
    app = TrajectoryApp(tmp_path, initial_step=0, refresh_interval=0)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        assert len(app.files) == 1

        events = tmp_path / "0" / "events"
        new_path = events / "events_new.jsonl"
        new_path.write_text(
            json.dumps(
                {
                    "type": "trajectory_created",
                    "trajectory": {"id": "new-trajectory"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        app.refresh_from_disk()
        await pilot.pause()

        assert len(app.files) == 2
        assert new_path in app.files


@pytest.mark.asyncio
async def test_refresh_rereads_open_growing_trajectory(tmp_path: Path) -> None:
    make_rollout(tmp_path, 0)
    app = TrajectoryApp(tmp_path, initial_step=0, refresh_interval=0)

    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.loaded is not None
        assert app.loaded.action_count == 0

        with app.loaded.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": "trajectory_step_added",
                        "trajectory_id": "trajectory-0",
                        "step_index": 1,
                        "step": {
                            "action_events": [
                                {
                                    "tool_name": "think",
                                    "action": {
                                        "kind": "ThinkAction",
                                        "thought": "new streamed action",
                                    },
                                }
                            ]
                        },
                    }
                )
                + "\n"
            )

        app.refresh_from_disk()
        await pilot.pause()

        assert app.detail_mode
        assert app.loaded is not None
        assert app.loaded.action_count == 1
        timeline = app.query_one("#timeline", RichLog)
        assert "new streamed action" in "\n".join(line.text for line in timeline.lines)


@pytest.mark.asyncio
async def test_group_separator_row_precedes_trajectory_rows(tmp_path: Path) -> None:
    events = tmp_path / "0" / "events"
    events.mkdir(parents=True)
    for suffix, reward in (
        ("00000000-0000-0000-0000-000000000001", 0),
        ("00000000-0000-0000-0000-000000000002", 1),
    ):
        (events / f"events_shared_group_{suffix}.jsonl").write_text(
            json.dumps(
                {
                    "type": "trajectory_created",
                    "trajectory": {"id": suffix},
                }
            )
            + "\n"
            + json.dumps({"type": "trajectory_finished", "reward": reward})
            + "\n",
            encoding="utf-8",
        )

    app = TrajectoryApp(tmp_path, initial_step=0)
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        table = app.query_one("#trajectories", DataTable)

        assert table.row_count == 3  # one group header plus two trajectories
        assert app.table_row_by_file_index == {0: 1, 1: 2}

        await pilot.press("enter")
        await pilot.pause()
        assert app.detail_mode


@pytest.mark.asyncio
async def test_repeated_finish_action_collapses_into_final_message(
    tmp_path: Path,
) -> None:
    events = tmp_path / "0" / "events"
    events.mkdir(parents=True)
    path = events / "events_finish.jsonl"
    message = "The requested change is complete."
    path.write_text(
        json.dumps(
            {
                "type": "trajectory_created",
                "trajectory": {"id": "finish-trajectory"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "trajectory_step_added",
                "trajectory_id": "finish-trajectory",
                "step_index": 1,
                "step": {
                    "action_events": [
                        {
                            "tool_name": "finish",
                            "tool_call_id": "finish-call",
                            "action": {
                                "kind": "FinishAction",
                                "message": message,
                            },
                        }
                    ],
                    "observation_events": [
                        {
                            "tool_name": "finish",
                            "tool_call_id": "finish-call",
                            "observation": {
                                "content": [{"type": "text", "text": message}],
                                "is_error": False,
                            },
                        }
                    ],
                },
                "finish_message": message,
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "trajectory_finished",
                "trajectory_id": "finish-trajectory",
                "reward": 1,
                "finish_message": message,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    app = TrajectoryApp(tmp_path, initial_step=0)
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        timeline = app.query_one("#timeline", RichLog)
        rendered = "\n".join(line.text for line in timeline.lines)

        assert "FINAL MESSAGE" in rendered
        assert rendered.index("TOTAL REWARD") > rendered.index(message)
        assert "1.00" in rendered[rendered.index("TOTAL REWARD"):]
        assert "FinishAction" not in rendered


@pytest.mark.asyncio
async def test_provisional_reward_is_hidden_until_trajectory_finishes(
    tmp_path: Path,
) -> None:
    events = tmp_path / "0" / "events"
    events.mkdir(parents=True)
    (events / "events_running.jsonl").write_text(
        json.dumps(
            {
                "type": "trajectory_step_added",
                "trajectory_id": "still-running",
                "step_index": 1,
                "reward": 1,
                "finish_message": "provisional message",
                "step": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    app = TrajectoryApp(tmp_path, initial_step=0)
    async with app.run_test(size=(100, 35)) as pilot:
        await pilot.pause()
        table = app.query_one("#trajectories", DataTable)
        assert table.get_cell_at((1, 1)).plain == "?"

        await pilot.press("enter")
        await pilot.pause()
        timeline = app.query_one("#timeline", RichLog)
        rendered = "\n".join(line.text for line in timeline.lines)

        assert "provisional message" in rendered
        assert "REWARD" not in rendered


def test_completed_groups_require_all_expected_rollouts(tmp_path):
    from trajectory_visualizer.app import _all_files_reward_text
    from uuid import uuid4

    rows = []
    for group, finished in (
        ("complete", [True] * 8),
        ("active", [True] * 7 + [False]),
        ("not_all_created", [True] * 7),
    ):
        for done in finished:
            path = tmp_path / f"events_{group}_{uuid4()}.jsonl"
            rows.append((path, (0.0 if done else None, "", 1, path.name, {}, done)))
    assert "COMPLETED GROUPS  1/3" in _all_files_reward_text(rows).plain
    assert "COMPLETED GROUPS  2/3" in _all_files_reward_text(rows, group_size=7).plain


def test_json_tool_reward_in_summary_and_detail():
    from trajectory_visualizer.app import _all_files_reward_text, _format_reward_components
    from trajectory_visualizer.parser import _reward_components
    components = _reward_components({
        "tool_json_error_reward": 1.0,
        "agent_error_event_reward": 0.0,
        "str_replace_reward": 0.5,
        "localization": {"file_f1_reward": 1.0, "module_f1_reward": 0.5, "entity_f1_reward": 0.0},
    })
    summary = _all_files_reward_text([(Path('sample.jsonl'), (0, '', 1, 'sample', components, True))]).plain
    detail = _format_reward_components(components)
    for text in (summary, detail):
        normalized = " ".join(text.split())
        assert "tool_json_error_reward 1.000" in normalized
        assert "localization/file_f1_reward 1.000" in normalized
        assert "localization/entity_f1_reward 0.000" in normalized
        assert "localization/module_f1_reward 0.500" in normalized
    assert "tool_format_reward" not in components
