"""Read-only local HTTP viewer. No web framework or frontend build is required."""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .model import Observation, Trajectory
from .parser import (
    TrajectoryParseError,
    load_trajectory,
    training_steps,
    trajectory_files,
)

ASSETS = Path(__file__).with_name("static")
UUID_SUFFIX = re.compile(r"_[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")


def finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def json_safe(value: Any) -> Any:
    """Keep diagnostics inspectable while emitting strict, browser-compatible JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_safe(child) for child in value]
    return value


def task_id(trajectory: Trajectory) -> str:
    return trajectory.task_id or UUID_SUFFIX.sub("", trajectory.path.stem).removeprefix(
        "events_"
    )


def observation_data(observation: Observation) -> dict[str, Any]:
    return {
        "tool_name": observation.tool_name,
        "tool_call_id": observation.tool_call_id,
        "text": observation.text,
        "is_error": observation.is_error,
    }


def trajectory_data(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "file": trajectory.path.name,
        "trajectory_id": trajectory.trajectory_id,
        "task_id": task_id(trajectory),
        "turn_count": trajectory.completion_count,
        "step_count": len(trajectory.turns),
        "task": trajectory.task,
        "reward": finite_number(trajectory.reward),
        "is_finished": trajectory.is_finished,
        "finish_message": trajectory.finish_message,
        "error_message": trajectory.error_message,
        "parse_warnings": trajectory.parse_warnings,
        "reward_components": trajectory.reward_components,
        "terminal_reward_info": trajectory.terminal_reward_info,
        "turns": [
            {
                "index": turn.index,
                "completion_id": turn.completion_id,
                "reward": finite_number(turn.reward),
                "cumulative_reward": finite_number(turn.cumulative_reward),
                "reward_info": turn.reward_info,
                "actions": [
                    {
                        "tool_name": action.tool_name,
                        "kind": action.kind,
                        "summary": action.summary,
                        "detail": action.detail,
                        "tool_call_id": action.tool_call_id,
                        "observations": [
                            observation_data(item) for item in action.observations
                        ],
                    }
                    for action in turn.actions
                ],
                "unmatched_observations": [
                    observation_data(item) for item in turn.unmatched_observations
                ],
            }
            for turn in trajectory.turns
        ],
    }


class RolloutStore:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        training_steps(self.root)  # Fail immediately for an invalid root.
        # Cache only summaries, not entire trajectories or their potentially large logs.
        self._cache: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def steps(self) -> list[Path]:
        return training_steps(self.root)

    def step(self, name: str) -> Path:
        for step in self.steps():
            if step.name == name:
                return step
        raise FileNotFoundError("Unknown training step")

    def file(self, step: str, name: str) -> Path:
        events = (self.step(step) / "events").resolve()
        return self._confined_file(events, name)

    @staticmethod
    def _confined_file(events: Path, name: str) -> Path:
        if Path(name).name != name or not name.endswith(".jsonl"):
            raise FileNotFoundError("Unknown rollout file")
        path = (events / name).resolve()
        if path.parent != events or not path.is_file():
            raise FileNotFoundError("Unknown rollout file")
        return path

    def summary(self, path: Path) -> dict[str, Any]:
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        cached = self._cache.get(path)
        if cached and cached[0] == signature:
            return cached[1]
        try:
            trajectory = load_trajectory(path)
            result = {
                "file": path.name,
                "task_id": task_id(trajectory),
                "trajectory_id": trajectory.trajectory_id,
                "reward": finite_number(trajectory.reward)
                if trajectory.is_finished
                else None,
                "is_finished": trajectory.is_finished,
                "turn_count": trajectory.completion_count,
                "step_count": len(trajectory.turns),
                "action_count": trajectory.action_count,
                "error": trajectory.error_message,
                "warning_count": len(trajectory.parse_warnings),
            }
        except (TrajectoryParseError, ValueError, TypeError, AttributeError) as exc:
            result = {
                "file": path.name,
                "task_id": UUID_SUFFIX.sub("", path.stem).removeprefix("events_"),
                "trajectory_id": "",
                "reward": None,
                "is_finished": False,
                "turn_count": None,
                "step_count": 0,
                "action_count": 0,
                "error": str(exc),
                "warning_count": 1,
            }
        result["version"] = f"{signature[0]}:{signature[1]}"
        self._cache[path] = signature, result
        return result

    def rollouts(self, name: str) -> dict[str, Any]:
        # Serialize index refreshes so concurrent browser tabs don't repeatedly parse a run.
        with self._lock:
            groups: dict[str, list[dict[str, Any]]] = {}
            present: set[Path] = set()
            step = self.step(name)
            events = (step / "events").resolve()
            for candidate in trajectory_files(step):
                try:
                    path = self._confined_file(events, candidate.name)
                    present.add(path)
                    row = self.summary(path)
                except OSError:
                    continue  # A live writer may remove or replace a file during discovery.
                groups.setdefault(row["task_id"], []).append(row)
            for path in list(self._cache):
                if path.parent == events and path not in present:
                    del self._cache[path]
            result = []
            rewards = []
            for group_id, rows in groups.items():
                rows.sort(
                    key=lambda row: (
                        row["reward"] is None,
                        -(row["reward"] or 0),
                        row["file"],
                    )
                )
                values = [row["reward"] for row in rows if row["reward"] is not None]
                rewards.extend(values)
                result.append(
                    {
                        "task_id": group_id,
                        "rollouts": rows,
                        "average_reward": sum(values) / len(values) if values else None,
                        "finished": sum(row["is_finished"] for row in rows),
                    }
                )
            result.sort(
                key=lambda group: (
                    group["average_reward"] is None,
                    -(group["average_reward"] or 0),
                    group["task_id"],
                )
            )
            return {
                "step": name,
                "groups": result,
                "total": sum(len(rows) for rows in groups.values()),
                "finished": sum(group["finished"] for group in result),
                "average_reward": sum(rewards) / len(rewards) if rewards else None,
            }


def make_server(
    root: Path, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    store = RolloutStore(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; object-src 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def json(self, value: Any, status: int = HTTPStatus.OK) -> None:
            self.respond(
                status,
                json.dumps(json_safe(value), allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            try:
                assets = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                    "/style.css": ("style.css", "text/css; charset=utf-8"),
                }
                if url.path in assets:
                    name, content_type = assets[url.path]
                    self.respond(
                        HTTPStatus.OK, (ASSETS / name).read_bytes(), content_type
                    )
                elif url.path == "/api/steps":
                    self.json(
                        {
                            "root": str(store.root),
                            "steps": [step.name for step in store.steps()],
                        }
                    )
                elif url.path == "/api/rollouts":
                    self.json(store.rollouts(query.get("step", [""])[0]))
                elif url.path == "/api/trajectory":
                    path = store.file(
                        query.get("step", [""])[0], query.get("file", [""])[0]
                    )
                    self.json(trajectory_data(load_trajectory(path)))
                else:
                    self.json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            except (FileNotFoundError, PermissionError):
                self.json(
                    {"error": "Training step or rollout file not found"},
                    HTTPStatus.NOT_FOUND,
                )
            except (TrajectoryParseError, ValueError, TypeError, AttributeError) as exc:
                self.json(
                    {"error": f"Unable to read rollout: {exc}"},
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                )
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError:
                self.json(
                    {"error": "Unable to read rollout data"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse rollout JSONL files in a local HTML viewer."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Rollout train root, numeric step directory, or events directory",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()
    try:
        server = make_server(args.root, args.host, args.port)
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(1, f"Cannot start viewer: {exc}\n")
    print(f"Trajectory viewer: http://{args.host}:{server.server_port}", flush=True)
    print(f"Reading {args.root.expanduser().resolve()} · Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
