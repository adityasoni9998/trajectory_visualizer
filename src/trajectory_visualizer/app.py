from __future__ import annotations

import argparse
import math
import random
import re
from numbers import Real
from pathlib import Path

from rich import box
from rich.columns import Columns
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.timer import Timer
from textual import work
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    LoadingIndicator,
    RichLog,
    Static,
)
from textual.worker import Worker, WorkerState

from .model import Action, Observation, Trajectory
from .parser import (
    TrajectoryParseError,
    load_trajectory,
    resolve_trajectory_path,
    training_steps,
    trajectory_files,
)
from .widgets import PathDialog, StepDialog


DEFAULT_ROOT = Path(
    "/data/user_data/adityabs/platoon_skyrl_tinker/logs/"
    "issue_resolution-platoon-tinker/"
    "skyrl-flame-swesmith-fft-cispo-20260802-133531/rollouts/train"
)
DEFAULT_REFRESH_INTERVAL = 60.0

ROLLOUT_UUID_SUFFIX = re.compile(
    r"_[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.jsonl$"
)

REWARD_HIERARCHY = (
    (
        "UNIT TESTS",
        "composite_reward",
        (
            ("binary", ("binary_reward",), False),
            ("f2p pass", ("f2p_pass_fraction",), False),
            (
                "p2p fail",
                ("p2p_fail_fraction", "p2p_pass_fraction"),
                True,
            ),
        ),
    ),
    (
        "LOCALIZATION",
        "localization_reward",
        (
            ("file f1", ("file_f1_reward",), False),
            (
                "function f1",
                ("function_f1_reward", "entity_f1_reward"),
                False,
            ),
            ("module f1", ("module_f1_reward",), False),
        ),
    ),
    (
        "TOOL FORMAT",
        "tool_format_reward",
        (
            (
                "str replace",
                ("str_replace_reward", "file_editor_reward"),
                False,
            ),
            (
                "agent error",
                ("agent_error_reward", "agent_error_event_reward"),
                False,
            ),
        ),
    ),
)

KNOWN_REWARD_COMPONENTS = frozenset(
    component_key
    for _, aggregate_key, breakdown in REWARD_HIERARCHY
    for component_key in (
        aggregate_key,
        *(key for _, keys, _ in breakdown for key in keys),
    )
)

AVERAGED_REWARD_COMPONENTS = frozenset(
    component_key
    for _, aggregate_key, breakdown in REWARD_HIERARCHY
    for component_key in (
        aggregate_key,
        *(keys[0] for _, keys, _ in breakdown),
    )
)


def _clip(text: str, length: int = 90) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= length else clean[: length - 1] + "…"


def _same_text(left: str, right: str) -> bool:
    """Compare logged messages while ignoring formatting-only whitespace."""
    return " ".join(left.split()) == " ".join(right.split())


def _reward_text(reward: object) -> Text:
    if reward is None:
        return Text("?", style="bold yellow")
    return Text(_format_reward(reward), style="bold bright_cyan")


def _rollout_group(path: Path) -> str:
    """Return the shared filename prefix for a group of rollout samples."""
    return ROLLOUT_UUID_SUFFIX.sub("", path.name)


def _numeric_reward(reward: object) -> float | None:
    if not isinstance(reward, Real):
        return None
    value = float(reward)
    return value if math.isfinite(value) else None


def _format_reward(reward: object) -> str:
    numeric = _numeric_reward(reward)
    if numeric is None:
        return str(reward)
    return f"{numeric:.2f}"


def _format_breakdown_reward(reward: object) -> str:
    numeric = _numeric_reward(reward)
    if numeric is None:
        return str(reward)
    return f"{numeric:.3f}"


def _component_value(
    components: dict[str, float], keys: tuple[str, ...], invert_alias: bool
) -> float | None:
    selected_key = next((key for key in keys if key in components), None)
    if selected_key is None:
        return None
    value = components[selected_key]
    if invert_alias and selected_key != keys[0]:
        value = 1.0 - value
    return value


def _reward_sort_key(reward: object, filename: str) -> tuple[bool, float, str]:
    numeric = _numeric_reward(reward)
    return numeric is None, -(numeric or 0.0), filename


def _average_reward(
    rows: list[tuple[Path, tuple[object, ...]]],
) -> float | None:
    rewards = [
        numeric
        for _, metadata in rows
        if (numeric := _numeric_reward(metadata[0])) is not None
    ]
    return sum(rewards) / len(rewards) if rewards else None


def _format_reward_components(
    components: dict[str, float], *, include_breakdown: bool = True
) -> str:
    lines: list[str] = []
    for header, aggregate_key, breakdown in REWARD_HIERARCHY:
        children: list[str] = []
        for label, keys, invert_alias in breakdown:
            value = _component_value(components, keys, invert_alias)
            if value is None:
                continue
            children.append(f"{label} {_format_breakdown_reward(value)}")
        aggregate = components.get(aggregate_key)
        if aggregate is None and not children:
            continue
        heading = header + (
            f" {_format_reward(aggregate)}" if aggregate is not None else ""
        )
        lines.append(
            heading
            + (
                f"  ›  {' · '.join(children)}"
                if include_breakdown and children
                else ""
            )
        )
    return "\n".join(lines)


def _reward_box(name: str, value: object, *, breakdown: bool = False) -> Panel:
    formatted = (
        _format_breakdown_reward(value) if breakdown else _format_reward(value)
    )
    return Panel(
        Text(formatted, style="bold bright_white", justify="center"),
        title=Text(name.upper(), style="bold bright_cyan"),
        title_align="left",
        border_style="orange3",
        box=box.ROUNDED,
        padding=(0, 1),
        expand=False,
    )


def _reward_boxes(total: object, components: dict[str, float]) -> Group:
    sections: list[object] = [
        Columns([_reward_box("total reward", total)], padding=(0, 1))
    ]
    for header, aggregate_key, breakdown in REWARD_HIERARCHY:
        boxes: list[Panel] = []
        if aggregate_key in components:
            boxes.append(_reward_box(header, components[aggregate_key]))
        for label, keys, invert_alias in breakdown:
            value = _component_value(components, keys, invert_alias)
            if value is not None:
                boxes.append(_reward_box(label, value, breakdown=True))
        if boxes:
            sections.append(Text(header, style="bold orange3"))
            sections.append(Columns(boxes, padding=(0, 1)))
    return Group(*sections)


def _average_reward_components(
    rows: list[tuple[Path, tuple[object, ...]]],
) -> dict[str, float]:
    eligible_metadata = [
        metadata
        for _, metadata in rows
        if _numeric_reward(metadata[0]) is not None
    ]
    if not eligible_metadata:
        return {}

    totals = {name: 0.0 for name in AVERAGED_REWARD_COMPONENTS}
    for metadata in eligible_metadata:
        components = metadata[4]
        if not isinstance(components, dict):
            continue

        # Normalize fallback field names to the primary display field before
        # averaging, so aliases also use the common rollout denominator.
        for _, aggregate_key, breakdown in REWARD_HIERARCHY:
            if (numeric := _numeric_reward(components.get(aggregate_key))) is not None:
                totals[aggregate_key] += numeric
            for _, keys, invert_alias in breakdown:
                numeric = _numeric_reward(
                    _component_value(components, keys, invert_alias)
                )
                if numeric is not None:
                    totals[keys[0]] += numeric

        # Preserve additional numeric reward fields even if they are not part
        # of the displayed hierarchy. Missing values still contribute zero.
        for name, value in components.items():
            name = str(name)
            if name in KNOWN_REWARD_COMPONENTS:
                continue
            if (numeric := _numeric_reward(value)) is not None:
                totals[name] = totals.get(name, 0.0) + numeric

    denominator = len(eligible_metadata)
    return {name: total / denominator for name, total in totals.items()}


def _all_files_reward_text(
    rows: list[tuple[Path, tuple[object, ...]]],
) -> Text:
    average = _average_reward(rows)
    components = _average_reward_components(rows)
    rewarded_count = sum(
        _numeric_reward(metadata[0]) is not None for _, metadata in rows
    )
    summary = Text()
    summary.append("ALL FILES AVG  ", style="bold orange3")
    summary.append(
        _format_reward(average) if average is not None else "?",
        style="bold bright_white",
    )
    summary.append(
        f"  ({rewarded_count}/{len(rows)} rewarded)\n", style="dim"
    )
    shown_aggregates = 0
    for header, aggregate_key, breakdown in REWARD_HIERARCHY:
        value = components.get(aggregate_key)
        child_values = [
            (label, child_value)
            for label, keys, invert_alias in breakdown
            if (child_value := _component_value(
                components, keys, invert_alias
            ))
            is not None
        ]
        if value is None and not child_values:
            continue
        if shown_aggregates:
            summary.append("\n")
        summary.append(f"{header}  ", style="bold bright_cyan")
        if value is not None:
            summary.append(_format_reward(value), style="bold bright_white")
        if child_values:
            summary.append("  ›  ", style="orange3")
        for index, (label, child_value) in enumerate(child_values):
            if index:
                summary.append("  ·  ", style="orange3")
            summary.append(f"{label} ", style="bright_cyan")
            summary.append(
                _format_breakdown_reward(child_value),
                style="bold bright_white",
            )
        shown_aggregates += 1
    return summary


def _observation_panel(observation: Observation, number: int) -> Panel:
    style = "red" if observation.is_error else "cyan"
    title = f"OBS {number} · {observation.tool_name}"
    if observation.tool_call_id:
        title += f" · {observation.tool_call_id[-8:]}"
    body = observation.text or "(empty observation)"
    return Panel(
        Syntax(body, "text", word_wrap=True),
        title=title,
        title_align="left",
        border_style=style,
        box=box.HEAVY,
        padding=(0, 1),
    )


def _action_panel(action: Action, number: int) -> Panel:
    styles = {
        "TerminalAction": "yellow",
        "FileEditorAction": "magenta",
        "ThinkAction": "blue",
        "FinishAction": "green",
        "TaskTrackerAction": "bright_cyan",
        "ToolCallAction": "bright_yellow",
    }
    style = styles.get(action.kind, "white")
    title = f"ACTION {number} · {action.kind}"
    if action.tool_call_id:
        title += f" · {action.tool_call_id[-8:]}"
    body = action.detail or action.summary or "(empty action)"
    return Panel(
        Syntax(body, "bash" if action.kind == "TerminalAction" else "text", word_wrap=True),
        title=title,
        title_align="left",
        border_style=style,
        box=box.HEAVY,
        padding=(0, 1),
    )


class StepList(ListView):
    class StepChosen(Message):
        def __init__(self, step_index: int) -> None:
            self.step_index = step_index
            super().__init__()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.post_message(self.StepChosen(event.index))


class TrajectoryApp(App[None]):
    TITLE = "Trajectory Visualizer"
    SUB_TITLE = "actions · observations · outcome"

    CSS = """
    Screen {
        background: #0b1020;
    }
    #body {
        height: 1fr;
    }
    #sidebar {
        width: 12;
        min-width: 10;
        border-right: solid #334155;
        background: #111827;
    }
    #sidebar-title {
        height: 3;
        padding: 1;
        color: #94a3b8;
        text-style: bold;
    }
    #steps {
        height: 1fr;
    }
    #content {
        width: 1fr;
    }
    #step-summary {
        height: 3;
        padding: 1 2;
        background: #111827;
        color: #cbd5e1;
    }
    #reward-summary {
        height: 6;
        padding: 0 1;
        border: solid #f59e0b;
        background: #111827;
    }
    #trajectories {
        height: 1fr;
    }
    #timeline {
        height: 1fr;
        display: none;
        padding: 0 1;
        background: #0b1020;
    }
    #loading {
        display: none;
        height: 3;
    }
    ListView > ListItem {
        padding: 0 1;
    }
    ListView > ListItem.--highlight {
        background: #1d4ed8;
    }
    DataTable > .datatable--header {
        background: #1e293b;
        color: #e2e8f0;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "random_trajectory", "Random"),
        Binding("u", "refresh", "Refresh"),
        Binding("s", "toggle_stream", "Stream"),
        Binding("p", "open_path", "Open path"),
        Binding("g", "jump_step", "Go to step"),
        Binding("b", "back_to_list", "Back"),
        Binding("escape", "back_to_list", "Back", show=False),
        Binding("[", "previous_step", "Prev step", show=False),
        Binding("]", "next_step", "Next step", show=False),
        Binding("f", "toggle_focus", "Focus", show=False),
        Binding("home", "timeline_home", "Top", show=False),
        Binding("end", "timeline_end", "Bottom", show=False),
    ]

    def __init__(
        self,
        root: Path,
        initial_step: int | None = None,
        initial_file: Path | None = None,
        observation_limit: int = 500,
        refresh_interval: float = DEFAULT_REFRESH_INTERVAL,
    ) -> None:
        super().__init__()
        self.root = root
        self.initial_step = initial_step
        self.initial_file = initial_file
        self.observation_limit = observation_limit
        self.refresh_interval = refresh_interval
        self.streaming_enabled = refresh_interval > 0
        self.refresh_timer: Timer | None = None
        self.steps: list[Path] = []
        self.files: list[Path] = []
        self.files_signature: tuple[tuple[Path, int, int], ...] = ()
        self.table_row_by_file_index: dict[int, int] = {}
        self.selected_step_index = 0
        self.loaded: Trajectory | None = None
        self.detail_mode = False
        self.step_loading = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("TRAIN STEP", id="sidebar-title")
                yield StepList(id="steps")
            with Vertical(id="content"):
                yield Static("Loading steps…", id="step-summary")
                yield Static("Calculating all-file averages…", id="reward-summary")
                yield LoadingIndicator(id="loading")
                yield DataTable(cursor_type="row", zebra_stripes=True, id="trajectories")
                yield RichLog(
                    highlight=False,
                    markup=True,
                    wrap=True,
                    auto_scroll=False,
                    id="timeline",
                )
        yield Footer()

    def on_mount(self) -> None:
        try:
            self.steps = training_steps(self.root)
        except FileNotFoundError as exc:
            self.notify(str(exc), severity="error", timeout=10)
            return
        if not self.steps:
            self.notify(f"No numeric training steps under {self.root}", severity="error")
            return

        step_list = self.query_one("#steps", StepList)
        for step in self.steps:
            step_list.append(ListItem(Label(step.name)))

        table = self.query_one("#trajectories", DataTable)
        table.add_columns(
            "#",
            "reward",
            "final error",
            "actions",
            "trajectory file",
            "reward breakdown",
        )

        target_index = len(self.steps) - 1
        if self.initial_step is not None:
            target_index = next(
                (
                    index
                    for index, step in enumerate(self.steps)
                    if int(step.name) == self.initial_step
                ),
                target_index,
            )
        elif self.initial_file is not None:
            target_index = next(
                (
                    index
                    for index, step in enumerate(self.steps)
                    if step.resolve() in self.initial_file.resolve().parents
                ),
                target_index,
            )
        self.select_step(target_index)
        self.refresh_timer = self.set_interval(
            self.refresh_interval or DEFAULT_REFRESH_INTERVAL, self._poll_rollouts
        )
        if not self.streaming_enabled:
            self.refresh_timer.pause()

    def on_unmount(self) -> None:
        if self.refresh_timer is not None:
            self.refresh_timer.stop()

    @staticmethod
    def _files_signature(files: list[Path]) -> tuple[tuple[Path, int, int], ...]:
        """Describe directory contents and file growth without reading JSONL."""
        signature: list[tuple[Path, int, int]] = []
        for path in files:
            try:
                stat = path.stat()
            except OSError:
                continue
            signature.append((path, stat.st_mtime_ns, stat.st_size))
        return tuple(signature)

    def _update_step_summary(self) -> None:
        if not self.steps:
            return
        stream = (
            f"[bold green]streaming "
            f"{self.refresh_interval or DEFAULT_REFRESH_INTERVAL:g}s[/]"
            if self.streaming_enabled
            else "[dim]stream off[/]"
        )
        self.query_one("#step-summary", Static).update(
            f"[bold cyan]STEP {self.steps[self.selected_step_index].name}[/]  "
            f"[dim]{len(self.files):,} trajectories[/]  {stream}  "
            "[dim]Enter: open · u: refresh · s: stream · r: random[/]"
        )

    def select_step(self, index: int) -> None:
        if not self.steps:
            return
        self.show_list_view()
        self.selected_step_index = max(0, min(index, len(self.steps) - 1))
        step_list = self.query_one("#steps", StepList)
        step_list.index = self.selected_step_index
        self.files = trajectory_files(self.steps[self.selected_step_index])
        self.files_signature = self._files_signature(self.files)
        self._update_step_summary()
        table = self.query_one("#trajectories", DataTable)
        table.clear()
        self.query_one("#reward-summary", Static).update(
            "Calculating all-file averages…"
        )
        self.query_one("#timeline", RichLog).clear()
        self._set_loading(True)
        self.load_step_rows(
            self.steps[self.selected_step_index], self.files, None, False
        )

    def _set_loading(self, loading: bool) -> None:
        self.step_loading = loading
        self.query_one("#loading").styles.display = (
            "block" if loading and not self.detail_mode else "none"
        )

    def show_list_view(self) -> None:
        self.detail_mode = False
        self.query_one("#sidebar").styles.display = "block"
        self.query_one("#step-summary").styles.display = "block"
        self.query_one("#reward-summary").styles.display = "block"
        self.query_one("#trajectories").styles.display = "block"
        self.query_one("#timeline").styles.display = "none"
        self.query_one("#loading").styles.display = (
            "block" if self.step_loading else "none"
        )
        if self.is_mounted:
            self.query_one("#trajectories", DataTable).focus()

    def show_detail_view(self) -> None:
        self.detail_mode = True
        self.query_one("#sidebar").styles.display = "none"
        self.query_one("#step-summary").styles.display = "none"
        self.query_one("#reward-summary").styles.display = "none"
        self.query_one("#trajectories").styles.display = "none"
        self.query_one("#loading").styles.display = "none"
        timeline = self.query_one("#timeline", RichLog)
        timeline.styles.display = "block"
        timeline.focus()

    @staticmethod
    def _row_metadata(path: Path) -> tuple[object, ...]:
        try:
            trajectory = load_trajectory(path)
            return (
                trajectory.reward if trajectory.is_finished else None,
                _clip(trajectory.error_message, 60) if trajectory.error_message else "",
                trajectory.action_count,
                path.name,
                trajectory.reward_components if trajectory.is_finished else {},
            )
        except (TrajectoryParseError, OSError) as exc:
            return ("?", "PARSE", 0, f"{path.name} ({exc})", {})

    @classmethod
    def _sorted_step_rows(
        cls, files: list[Path]
    ) -> list[tuple[Path, tuple[object, ...]]]:
        rows = [(path, cls._row_metadata(path)) for path in files]
        grouped: dict[str, list[tuple[Path, tuple[object, ...]]]] = {}
        for row in rows:
            grouped.setdefault(_rollout_group(row[0]), []).append(row)

        ordered: list[tuple[Path, tuple[object, ...]]] = []
        groups: list[
            tuple[float | None, str, list[tuple[Path, tuple[object, ...]]]]
        ] = []
        for group_name, group_rows in grouped.items():
            groups.append((_average_reward(group_rows), group_name, group_rows))

        for _, _, group_rows in sorted(
            groups,
            key=lambda group: (
                group[0] is None,
                -(group[0] or 0.0),
                group[1],
            ),
        ):
            group_rows.sort(
                key=lambda row: _reward_sort_key(row[1][0], row[0].name)
            )
            ordered.extend(group_rows)
        return ordered

    @work(thread=True, exclusive=True, group="step-loader")
    def load_step_rows(
        self,
        step: Path,
        files: list[Path],
        restore_file: Path | None,
        restore_detail: bool,
    ) -> tuple[
        Path,
        list[tuple[Path, tuple[object, ...]]],
        Path | None,
        bool,
    ]:
        return step, self._sorted_step_rows(files), restore_file, restore_detail

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.group != "step-loader" or event.state != WorkerState.SUCCESS:
            return
        step, rows, restore_file, restore_detail = event.worker.result
        if not self.steps or step != self.steps[self.selected_step_index]:
            return
        self.files = [path for path, _ in rows]
        table = self.query_one("#trajectories", DataTable)
        self.query_one("#reward-summary", Static).update(
            _all_files_reward_text(rows)
        )
        self.table_row_by_file_index.clear()
        group_rows: dict[str, list[tuple[Path, tuple[object, ...]]]] = {}
        for row in rows:
            group_rows.setdefault(_rollout_group(row[0]), []).append(row)

        current_group: str | None = None
        group_number = 0
        for index, (_, row) in enumerate(rows):
            path = rows[index][0]
            group_name = _rollout_group(path)
            if group_name != current_group:
                current_group = group_name
                group_number += 1
                members = group_rows[group_name]
                average_reward = _average_reward(members)
                average_label = (
                    _format_reward(average_reward)
                    if average_reward is not None
                    else "?"
                )
                average_components = _average_reward_components(members)
                group_style = "bold bright_cyan"
                display_name = group_name.removeprefix("events_")
                table.add_row(
                    Text("━━", style=group_style),
                    Text("━━━━", style=group_style),
                    Text("━" * 8, style=group_style),
                    Text(f"{len(members)} rollouts", style=group_style),
                    Text(
                        f"━━ {display_name} " + "━" * 8,
                        style=group_style,
                    ),
                    Text(
                        f"avg total {average_label}"
                        + (
                            f" · {_format_reward_components(average_components)}"
                            if average_components
                            else ""
                        ),
                        style=group_style,
                    ),
                    key=f"group:{group_number}",
                )

            reward, error, actions, filename, reward_components = row
            self.table_row_by_file_index[index] = table.row_count
            table.add_row(
                str(index + 1),
                _reward_text(reward),
                Text(str(error), style="bold red" if error else "dim"),
                str(actions),
                str(filename),
                _format_reward_components(
                    reward_components, include_breakdown=False
                ),
                key=f"trajectory:{index}",
            )
        self._set_loading(False)
        if not self.files:
            self.notify("This training step has no trajectories", severity="warning")
            return
        initial_index = 0
        open_initial_file = False
        if self.initial_file is not None:
            try:
                initial_index = self.files.index(self.initial_file.resolve())
                open_initial_file = True
            except ValueError:
                pass
            self.initial_file = None
        elif restore_file is not None:
            try:
                initial_index = self.files.index(restore_file)
            except ValueError:
                restore_detail = False
        if open_initial_file:
            self.open_trajectory(initial_index)
        else:
            table.move_cursor(row=self.table_row_by_file_index.get(initial_index, 0))
            if not restore_detail:
                self.loaded = None
                table.focus()

    def on_step_list_step_chosen(self, event: StepList.StepChosen) -> None:
        if event.step_index != self.selected_step_index:
            self.select_step(event.step_index)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = str(event.row_key.value)
        if not row_key.startswith("trajectory:"):
            return
        try:
            index = int(row_key.removeprefix("trajectory:"))
        except (TypeError, ValueError):
            return
        self.open_trajectory(index)

    def open_trajectory(self, index: int) -> None:
        if not 0 <= index < len(self.files):
            return
        table = self.query_one("#trajectories", DataTable)
        table.move_cursor(row=self.table_row_by_file_index.get(index, index))
        try:
            self.loaded = load_trajectory(self.files[index])
        except TrajectoryParseError as exc:
            self.notify(str(exc), severity="error")
            return
        self.show_detail_view()
        # The list/sidebar have just been hidden, so render after layout gives
        # panels the full detail-view width rather than the old split-view width.
        self.call_after_refresh(
            self.render_trajectory, self.loaded, self.streaming_enabled
        )

    def _truncated_observation(self, text: str) -> tuple[str, bool]:
        if self.observation_limit <= 0 or len(text) <= self.observation_limit:
            return text, False
        return text[: self.observation_limit].rstrip() + "\n… [observation truncated]", True

    def render_trajectory(
        self, trajectory: Trajectory, scroll_to_end: bool = False
    ) -> None:
        log = self.query_one("#timeline", RichLog)
        log.clear()
        identity = trajectory.trajectory_id or "(unknown id)"
        log.write(
            Panel(
                Text.from_markup(
                    f"[bold]FILE[/] {escape(trajectory.path.name)}\n"
                    f"[bold]ID[/]   {escape(identity)}"
                ),
                border_style="bright_blue",
                title="TRAJECTORY",
                title_align="left",
            ),
            expand=True,
        )
        if trajectory.task:
            log.write(
                Panel(
                    Syntax(
                        self._truncated_observation(trajectory.task)[0],
                        "text",
                        word_wrap=True,
                    ),
                    title="TASK / INITIAL ENVIRONMENT",
                    title_align="left",
                    border_style="bright_blue",
                ),
                expand=True,
            )

        action_number = 0
        observation_number = 0
        for turn in trajectory.turns:
            if not turn.actions and not turn.unmatched_observations:
                continue
            log.write(Text(f" STEP {turn.index} ", style="bold reverse"))
            # Each action retains every matched observation. This correctly renders
            # parallel tool calls recorded together in a single trajectory step.
            for action in turn.actions:
                finish_is_repeated = (
                    action.kind == "FinishAction"
                    and bool(trajectory.finish_message)
                    and _same_text(action.detail, trajectory.finish_message)
                    and not any(
                        observation.is_error for observation in action.observations
                    )
                )
                if finish_is_repeated:
                    # The logger normally stores the same final response as a
                    # FinishAction, its echo observation, and finish_message.
                    # Keep only the dedicated final-message panel.
                    continue
                action_number += 1
                log.write(_action_panel(action, action_number), expand=True)
                for observation in action.observations:
                    observation_number += 1
                    shown, _ = self._truncated_observation(observation.text)
                    rendered = Observation(
                        tool_call_id=observation.tool_call_id,
                        tool_name=observation.tool_name,
                        text=shown,
                        is_error=observation.is_error,
                        raw=observation.raw,
                    )
                    log.write(
                        _observation_panel(rendered, observation_number), expand=True
                    )
            for observation in turn.unmatched_observations:
                observation_number += 1
                shown, _ = self._truncated_observation(observation.text)
                rendered = Observation(
                    tool_call_id=observation.tool_call_id,
                    tool_name=observation.tool_name,
                    text=shown,
                    is_error=observation.is_error,
                    raw=observation.raw,
                )
                log.write(
                    _observation_panel(rendered, observation_number), expand=True
                )

        if trajectory.finish_message:
            finish_message = Syntax(
                trajectory.finish_message, "text", word_wrap=True
            )
            log.write(
                Panel(
                    finish_message,
                    title="FINAL MESSAGE",
                    title_align="left",
                    border_style="green",
                ),
                expand=True,
            )
        if trajectory.error_message:
            log.write(
                Panel(
                    Syntax(trajectory.error_message, "text", word_wrap=True),
                    title="TRAJECTORY_FINISHED ERROR",
                    title_align="left",
                    border_style="bold red",
                ),
                expand=True,
            )
        if trajectory.parse_warnings:
            log.write(
                Panel(
                    "\n".join(trajectory.parse_warnings),
                    title="PARSE WARNINGS",
                    border_style="yellow",
                ),
                expand=True,
            )
        if trajectory.is_finished:
            log.write(
                _reward_boxes(trajectory.reward, trajectory.reward_components),
                expand=True,
            )
        if scroll_to_end:
            log.scroll_end(animate=False)
        else:
            log.scroll_home(animate=False)

    def on_resize(self) -> None:
        if self.detail_mode and self.loaded is not None:
            # RichLog stores rendered lines, so explicitly re-render on terminal
            # resize to let every panel grow or shrink with the available width.
            # A short zero-work timer runs after the resized layout is committed.
            self.set_timer(0.01, self._render_loaded_after_resize)

    def _render_loaded_after_resize(self) -> None:
        if self.detail_mode and self.loaded is not None:
            self.render_trajectory(self.loaded)

    def _poll_rollouts(self) -> None:
        self.refresh_from_disk()

    def refresh_from_disk(self, force: bool = False) -> None:
        """Discover new rollout steps/files and re-read a growing open JSONL."""
        try:
            discovered_steps = training_steps(self.root)
        except FileNotFoundError as exc:
            self.notify(str(exc), severity="error", timeout=8)
            return
        if not discovered_steps:
            return

        old_steps = self.steps
        old_selected = (
            old_steps[self.selected_step_index] if old_steps else None
        )
        followed_latest = bool(old_steps) and self.selected_step_index == len(old_steps) - 1
        if discovered_steps != old_steps:
            # Training writes steps monotonically. Append the common case without
            # disturbing ListView's selection or keyboard focus.
            if discovered_steps[: len(old_steps)] == old_steps:
                step_list = self.query_one("#steps", StepList)
                for step in discovered_steps[len(old_steps) :]:
                    step_list.append(ListItem(Label(step.name)))
            else:
                self.notify(
                    "Training step directories changed; restart to rebuild the sidebar",
                    severity="warning",
                )
                return
            self.steps = discovered_steps
            if followed_latest:
                self.select_step(len(self.steps) - 1)
                return
            if old_selected in self.steps:
                self.selected_step_index = self.steps.index(old_selected)

        step = self.steps[self.selected_step_index]
        discovered_files = trajectory_files(step)
        signature = self._files_signature(discovered_files)
        if not force and signature == self.files_signature:
            return

        restore_file = self.loaded.path if self.detail_mode and self.loaded else None
        restore_detail = restore_file is not None and restore_file in discovered_files
        self.files = discovered_files
        self.files_signature = signature
        self._update_step_summary()
        self.query_one("#trajectories", DataTable).clear()
        self._set_loading(True)
        self.load_step_rows(step, discovered_files, restore_file, restore_detail)

        if restore_detail and restore_file is not None:
            try:
                self.loaded = load_trajectory(restore_file)
            except TrajectoryParseError as exc:
                self.notify(str(exc), severity="error")
            else:
                self.call_after_refresh(
                    self.render_trajectory,
                    self.loaded,
                    self.streaming_enabled,
                )

    def action_refresh(self) -> None:
        self.refresh_from_disk(force=True)
        self.notify("Refreshed rollouts", timeout=2)

    def action_toggle_stream(self) -> None:
        if self.refresh_timer is None:
            return
        self.streaming_enabled = not self.streaming_enabled
        if self.streaming_enabled:
            self.refresh_timer.resume()
            self.refresh_from_disk(force=True)
            self.notify(
                f"Streaming every "
                f"{self.refresh_interval or DEFAULT_REFRESH_INTERVAL:g} seconds",
                timeout=3,
            )
        else:
            self.refresh_timer.pause()
            self.notify("Streaming paused", timeout=3)
        self._update_step_summary()

    def action_random_trajectory(self) -> None:
        if self.files:
            self.open_trajectory(random.randrange(len(self.files)))
            self.query_one("#timeline", RichLog).focus()

    def action_open_path(self) -> None:
        if not self.steps:
            return

        def chosen(value: str | None) -> None:
            if not value:
                return
            step = self.steps[self.selected_step_index]
            try:
                path = resolve_trajectory_path(step, value)
                index = self.files.index(path)
            except (FileNotFoundError, ValueError) as exc:
                self.notify(str(exc), severity="error", timeout=8)
                return
            self.open_trajectory(index)
            self.query_one("#timeline", RichLog).focus()

        self.push_screen(PathDialog(self.steps[self.selected_step_index].name), chosen)

    def action_jump_step(self) -> None:
        def chosen(value: str | None) -> None:
            if not value:
                return
            index = next(
                (
                    index
                    for index, step in enumerate(self.steps)
                    if step.name == value.strip()
                ),
                None,
            )
            if index is None:
                self.notify(f"Training step {value!r} does not exist", severity="error")
                return
            self.select_step(index)

        self.push_screen(StepDialog(), chosen)

    def action_back_to_list(self) -> None:
        if self.detail_mode:
            self.show_list_view()

    def action_previous_step(self) -> None:
        self.select_step(self.selected_step_index - 1)

    def action_next_step(self) -> None:
        self.select_step(self.selected_step_index + 1)

    def action_toggle_focus(self) -> None:
        timeline = self.query_one("#timeline", RichLog)
        table = self.query_one("#trajectories", DataTable)
        (table if timeline.has_focus else timeline).focus()

    def action_timeline_home(self) -> None:
        self.query_one("#timeline", RichLog).scroll_home(animate=False)

    def action_timeline_end(self) -> None:
        self.query_one("#timeline", RichLog).scroll_end(animate=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trajectory-viewer",
        description="Inspect rollout trajectories in a colored terminal UI.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Training rollout root containing numeric step directories.",
    )
    parser.add_argument("--step", type=int, help="Training step to select initially.")
    parser.add_argument(
        "--file",
        type=Path,
        help="Specific trajectory JSONL file to open initially.",
    )
    parser.add_argument(
        "--observation-limit",
        type=int,
        default=500,
        help="Maximum displayed characters per observation; 0 disables truncation.",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=DEFAULT_REFRESH_INTERVAL,
        metavar="SECONDS",
        help=(
            "Poll for new steps, rollouts, and trajectory events at this interval; "
            "defaults to 60 seconds; 0 starts with streaming disabled."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.refresh_interval < 0:
        raise SystemExit("--refresh-interval must be zero or greater")
    root = args.root.expanduser().resolve()
    initial_file = args.file.expanduser().resolve() if args.file else None
    if initial_file is not None and not args.root.exists():
        # Let --file infer the train root: .../train/<step>/events/<file>.
        parents = initial_file.parents
        if len(parents) >= 3:
            root = parents[2]
    TrajectoryApp(
        root=root,
        initial_step=args.step,
        initial_file=initial_file,
        observation_limit=args.observation_limit,
        refresh_interval=args.refresh_interval,
    ).run()


if __name__ == "__main__":
    main()
