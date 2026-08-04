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
When reward metadata is available, the final column also shows the individual
reward components, F2P pass fraction, and P2P fail fraction for each rollout and
their per-group averages. The breakdown is grouped under unit tests,
localization, and tool-format rewards. Totals and parent rewards use two decimal places; child
metrics use three. File-list rows show only the three parent rewards; opening a
trajectory shows its full child breakdown.
An orange summary panel above the file list shows the all-file average total,
parent rewards, and individual child metrics for the selected training step.
The total average uses every finished rollout with a finite numeric reward as
its denominator, including zeroes. Every component average uses that same
denominator; a missing component contributes zero, regardless of the rollout's
total reward.

## Run

```bash
uv sync
uv run trajectory-viewer
```

The requested rollout directory is the built-in default. To inspect another run:

```bash
uv run trajectory-viewer /path/to/rollouts/train
uv run trajectory-viewer /path/to/rollouts/train --step 42
uv run trajectory-viewer /path/to/rollouts/train --file /path/to/events_file.jsonl
```

The viewer polls for updates every 60 seconds by default. To use a different
refresh interval while rollouts are being written:

```bash
uv run trajectory-viewer /path/to/rollouts/train --refresh-interval 1
```

The viewer discovers new training steps and rollout files and re-reads an open
trajectory as JSONL events are appended. When streaming is following the newest
step, it automatically advances to each newly created step. Press `s` to pause
or resume streaming; when launched with `--refresh-interval 0`, `s` starts
60-second polling.
Press `u` for a one-time refresh.

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
