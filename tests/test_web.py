from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

from trajectory_visualizer.web import RolloutStore, make_server, trajectory_data
from trajectory_visualizer.parser import load_trajectory


def write_rollout(path: Path, task: str = "task-from-event", reward: float = -0.05):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"type": "trajectory_created", "trajectory": {"id": "t"}},
        {"type": "trajectory_task_set", "task": {"id": task, "goal": "Fix the issue"}},
        {
            "type": "trajectory_step_added",
            "step_index": 1,
            "reward": reward,
            "step": {
                "reward": reward,
                "misc": {"reward_misc": {"nested": {"score": 0.6}, "penalty": reward}},
            },
        },
        {"type": "trajectory_finished", "reward": reward},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_groups_use_logged_task_id_and_cache_refreshes(tmp_path, monkeypatch):
    from trajectory_visualizer import web

    first = tmp_path / "2/events/unrelated-filename.jsonl"
    second = tmp_path / "2/events/another-name.jsonl"
    write_rollout(first, reward=0.5)
    write_rollout(second, reward=-0.1)
    store = RolloutStore(tmp_path)
    real_load = web.load_trajectory
    reads = []

    def count(path):
        reads.append(path)
        return real_load(path)

    monkeypatch.setattr(web, "load_trajectory", count)
    result = store.rollouts("2")
    assert len(result["groups"]) == 1
    assert result["groups"][0]["task_id"] == "task-from-event"
    assert result["average_reward"] == pytest.approx(0.2)
    assert [row["reward"] for row in result["groups"][0]["rollouts"]] == [0.5, -0.1]
    assert store.rollouts("2") == result
    assert len(reads) == 2
    write_rollout(second, reward=-0.25)
    assert store.rollouts("2")["average_reward"] == pytest.approx(0.125)
    assert len(reads) == 3
    second.unlink()
    assert store.rollouts("2")["total"] == 1


def test_step_rewards_and_dictionary_are_preserved(tmp_path):
    path = tmp_path / "events.jsonl"
    write_rollout(path)
    data = trajectory_data(load_trajectory(path))
    assert data["task_id"] == "task-from-event"
    assert data["turns"][0]["reward"] == -0.05
    assert data["turns"][0]["cumulative_reward"] == -0.05
    assert data["turns"][0]["reward_info"] == {
        "nested": {"score": 0.6},
        "penalty": -0.05,
    }


def test_setup_prompts_are_visible_as_observations(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "trajectory_step_added",
                "step_index": 0,
                "step": {
                    "observation_events": [
                        {
                            "kind": "SystemPromptEvent",
                            "system_prompt": {"text": "System instructions"},
                        },
                        {
                            "kind": "MessageEvent",
                            "llm_message": {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Fix the failing test"}
                                ],
                            },
                        },
                    ]
                },
            }
        )
        + "\n"
    )
    observations = trajectory_data(load_trajectory(path))["turns"][0][
        "unmatched_observations"
    ]
    assert [(obs["tool_name"], obs["text"]) for obs in observations] == [
        ("system", "System instructions"),
        ("user", "Fix the failing test"),
    ]


def test_conversation_errors_preserve_detail(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({
        "type": "trajectory_step_added",
        "step": {"observation_events": [{
            "kind": "ConversationErrorEvent", "code": "LLMServiceUnavailableError",
            "detail": "Upstream connection error",
        }]},
    }) + "\n")
    observation = trajectory_data(load_trajectory(path))["turns"][0]["unmatched_observations"][0]
    assert observation["text"] == "Upstream connection error"
    assert observation["tool_name"] == "LLMServiceUnavailableError"
    assert observation["is_error"]


def test_incomplete_rollout_is_not_averaged_and_recovers(tmp_path):
    path = tmp_path / "0/events/events_task_00000000-0000-0000-0000-000000000001.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"type":"trajectory_step_added","step":{"reward":0.5},"reward":0.5}\n{"type":'
    )
    store = RolloutStore(tmp_path)
    result = store.rollouts("0")
    assert result["groups"][0]["task_id"] == "task"
    assert result["average_reward"] is None
    assert result["groups"][0]["rollouts"][0]["warning_count"] == 1
    with path.open("a") as handle:
        handle.write('"trajectory_finished","reward":0.5}\n')
    result = store.rollouts("0")
    assert result["average_reward"] == 0.5
    assert result["finished"] == 1


@pytest.fixture
def server(tmp_path):
    write_rollout(tmp_path / "0/events/events_one.jsonl")
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", tmp_path
    server.shutdown()
    server.server_close()
    thread.join()


def test_http_assets_api_and_path_confinement(server):
    base, root = server
    for asset in ("/", "/style.css", "/app.js"):
        with urlopen(base + asset) as response:
            assert response.status == 200
            assert response.headers["Content-Security-Policy"]
            assert len(response.read()) > 100
    with urlopen(base + "/api/steps") as response:
        assert json.load(response)["steps"] == ["0"]
    with urlopen(base + "/api/rollouts?step=0") as response:
        assert json.load(response)["total"] == 1
    with urlopen(base + "/api/trajectory?step=0&file=events_one.jsonl") as response:
        assert json.load(response)["turns"][0]["reward"] == -0.05
    outside = root / "outside.jsonl"
    write_rollout(outside)
    (root / "0/events/link.jsonl").symlink_to(outside)
    for route in (
        "/../pyproject.toml",
        "/api/rollouts?step=../0",
        "/api/trajectory?" + urlencode({"step": "0", "file": "../../outside.jsonl"}),
        "/api/trajectory?step=0&file=link.jsonl",
    ):
        with pytest.raises(HTTPError) as error:
            urlopen(base + route)
        assert error.value.code == 404


def test_strict_json_handles_nonfinite_diagnostics(server):
    base, root = server
    path = root / "0/events/events_nan.jsonl"
    path.write_text(
        '{"type":"trajectory_step_added","step":{"reward":NaN,"misc":{"reward_misc":{"score":Infinity}}}}\n'
    )
    with urlopen(base + "/api/trajectory?step=0&file=events_nan.jsonl") as response:
        data = json.load(response)
    assert data["turns"][0]["reward"] is None
    assert data["turns"][0]["reward_info"]["score"] == "inf"
