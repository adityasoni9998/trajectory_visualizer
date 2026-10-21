# Trajectory Visualizer

A focused, read-only TUI for inspecting SkyRL/OpenHands rollout JSONL files. It
shows what the agent did, the environment response after each tool call, the
final total reward, and any `trajectory_finished` error. The trajectory table
shows a short error preview; opening the trajectory shows the complete message.

The viewer understands multiple action and observation events in one trajectory
step. It matches parallel tool calls to observations by `tool_call_id`, without
discarding unmatched events.

Rollouts sharing the filename prefix before the final UUID are treated as one
group. Groups are sorted by decreasing average total reward, and rollouts within
each group are sorted by decreasing individual reward. Unfinished or invalid
rewards appear last. Styled separator rows show each group's rollout count and
average total reward.
Reward fields are discovered automatically from `step.misc.reward_misc` and
optional `trajectory_finished.reward_misc`. Any finite numeric leaf is shown,
including unfamiliar keys, nested scores, deltas, and penalties. Nested paths use
`/` separators (`localization/file_f1_reward`); literal `~` and `/` in keys are
escaped as `~0` and `~1`. The viewer does not invent aggregates or infer weights
from field names. Booleans, text, arrays, and other diagnostics are retained in
the full reward-info view (`i`). No plugin or log-format changes are required.

Every recorded step shows its **step reward**, **cumulative event reward**, and
numeric diagnostics, including reward-only settlement steps. The final total
remains the recorded training reward; it is never reconstructed from metadata.
Missing step or cumulative rewards display `?`, rather than an inferred value.

The file list and final detail summary show each metric's **last observed numeric
value**, retaining sparse metrics that are absent from later steps. These are
snapshots and can come from different steps. For fields that you know represent
additive per-step contributions, request sums explicitly, without editing code:

```bash
uv run trajectory-viewer /path/to/rollouts/train \
  --sum-metric '*_delta' --sum-metric '*_penalty'
```

Patterns match the displayed paths (case sensitive); quote them to prevent shell
expansion. `sum:<path>` entries sum only recorded step metadata, never repeated
terminal metadata or cumulative event rewards. The last observed values remain
visible too. For rebench, raw localization/test deltas are **unweighted**; these
sums are diagnostics and should not be expected to equal the training reward.

The summary counts groups whose expected rollouts have all finished. Use
`--group-size N` to set the expected count (default: 8). Missing or unfinished
rollouts do not count as complete. Total and metadata averages use every finished
rollout with a finite numeric total, including zeroes. A missing metric contributes
zero, preserving the original viewer's averaging convention; `present n/N` shows
coverage so missing data can be distinguished from observed zeroes. Fields absent
from all rollouts are not fabricated.

The orange summary scrolls independently so newly discovered metrics cannot push
the rollout list off screen. File rows preview four metrics; opening a trajectory
shows all of them. Press `i` in the detail view to toggle full reward-info JSON,
including text diagnostics and patch hashes.

## Run

```bash
uv sync
uv run trajectory-viewer
```

The requested rollout directory is the built-in default. To inspect another run:

```bash
uv run trajectory-viewer /path/to/rollouts/train
uv run trajectory-viewer /path/to/rollouts/train --step 42
uv run trajectory-viewer /path/to/rollouts/train/42/events
uv run trajectory-viewer /path/to/rollouts/train --file /path/to/events_file.jsonl
```

The viewer polls for updates every 60 seconds by default. To use a different
refresh interval while rollouts are being written:

```bash
uv run trajectory-viewer /path/to/rollouts/train --refresh-interval 1
```

When opened on a training root, the viewer discovers new training steps and rollout files and re-reads an open
trajectory as JSONL events are appended. When streaming is following the newest
step, it automatically advances to each newly created step. Press `s` to pause
or resume streaming; when launched with `--refresh-interval 0`, `s` starts
60-second polling.
Press `u` for a one-time refresh. Opening a numeric step or its `events/` directory
limits browsing to that step.

Observations are limited to 500 displayed characters by default:

```bash
uv run trajectory-viewer --observation-limit 12000
uv run trajectory-viewer --observation-limit 0  # no truncation
```

## Keys

| Key | Action |
|---|---|
| `Enter` | Open the highlighted trajectory in the full-screen detail view |
| `b` / `Escape` | Return from trajectory detail to the file list |
| `u` | Refresh steps, rollout files, and the open trajectory once |
| `s` | Toggle live streaming / polling |
| `i` | Toggle numeric / full JSON reward info in trajectory details |
| `r` | Open a random trajectory from the selected step |
| `p` | Paste a filename/path from the selected step |
| `g` | Jump directly to a training step |
| `[` / `]` | Previous / next training step |
| `f` | Switch focus between trajectory list and timeline |
| Arrow keys / Page Up / Page Down | Navigate or scroll |
| `Home` / `End` | Top / bottom of timeline |
| `q` | Quit |

The file-path dialog only opens `.jsonl` files inside the current step's
`events/` directory. The application never edits rollout data.

Press `t` to collapse or restore the narrow step sidebar. The selection remains
visible above the table; `[` / `]` and `g` work while the sidebar is hidden.
