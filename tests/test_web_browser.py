"""Optional browser checks: uv run --group browser pytest tests/test_web_browser.py."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from trajectory_visualizer.web import make_server


def write_sample(root: Path, step: int, file: str, task: str, reward: float = 0.15):
    directory = root / str(step) / "events"
    directory.mkdir(parents=True, exist_ok=True)
    command = "printf 'hello'\n" * 70 + "ACTION_END"
    output = "observation output\n" * 70 + "OBSERVATION_END"
    events = [
        {"type": "trajectory_created", "trajectory": {"id": file}},
        {
            "type": "trajectory_task_set",
            "task": {"id": task, "goal": "Fix a sample issue"},
        },
        {
            "type": "trajectory_step_added",
            "step_index": 1,
            "reward": 0.2,
            "step": {
                "reward": 0.2,
                "action_events": [
                    {
                        "tool_name": "terminal",
                        "tool_call_id": "call-a",
                        "action": {"kind": "TerminalAction", "command": command},
                    }
                ],
                "observation_events": [
                    {
                        "tool_name": "terminal",
                        "tool_call_id": "call-a",
                        "observation": {"content": output},
                    }
                ],
                "misc": {
                    "reward_misc": {
                        "nested": {
                            "score": 0.7,
                            "note": '<img src=x onerror="window.pwned=1">',
                        },
                        "delta": 0.2,
                    }
                },
            },
        },
        {
            "type": "trajectory_step_added",
            "step_index": 2,
            "reward": reward,
            "step": {"reward": -0.05, "misc": {"reward_misc": {"penalty": -0.05}}},
        },
        {"type": "trajectory_finished", "reward": reward},
    ]
    (directory / file).write_text("".join(json.dumps(event) + "\n" for event in events))


@pytest.fixture
def browser_view(tmp_path):
    write_sample(tmp_path, 0, "arbitrary-a.jsonl", "same-task-id")
    write_sample(tmp_path, 0, "different-prefix.jsonl", "same-task-id", 0.05)
    write_sample(tmp_path, 1, "next-step.jsonl", "next-task-id")
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as manager:
            browser = manager.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/#step=0")
            playwright.expect(page.locator(".trajectory-header h1")).to_have_text(
                "same-task-id"
            )
            yield page, tmp_path, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_grouping_rewards_expansions_and_navigation(browser_view):
    page, root, errors = browser_view
    expect = playwright.expect
    expect(page.locator(".group-button")).to_have_count(1)
    expect(page.locator(".rollout")).to_have_count(2)
    expect(page.locator(".group-meta")).to_have_text("2 rollouts · 2 finished")
    expect(page.locator("#turn-1 .turn-rewards")).to_contain_text("Step reward-0.050")
    expect(page.locator(".reward-info[open]")).to_have_count(0)
    expect(page.locator(".json-tree")).to_have_count(0)

    action = page.locator(".event").nth(0)
    observation = page.locator(".event").nth(1)
    expect(action.locator("pre")).not_to_contain_text("ACTION_END")
    action.get_by_role("button", name="Show all", exact=False).click()
    expect(action.locator("pre")).to_contain_text("ACTION_END")
    action.get_by_role("button", name="Show less").click()
    expect(action.locator("pre")).not_to_contain_text("ACTION_END")
    observation.get_by_role("button", name="Show all", exact=False).click()
    expect(observation.locator("pre")).to_contain_text("OBSERVATION_END")

    page.locator(".reward-info > summary").first.click()
    page.locator(".reward-info summary").filter(has_text="nested {2 fields}").click()
    expect(page.locator(".json-entry").filter(has_text="score")).to_contain_text("0.7")
    expect(page.locator(".json-entry").filter(has_text="note")).to_contain_text(
        "<img src=x"
    )
    assert page.evaluate("window.pwned") is None
    expect(page.locator(".reward-info img")).to_have_count(0)

    page.get_by_role("button", name="Step 2: reward -0.050", exact=True).click()
    expect(page.locator("#turn-1")).to_be_focused()
    page.locator("#search").fill("missing-task")
    expect(page.locator(".group-button")).to_have_count(0)
    page.locator("#search").fill("")
    expect(page.locator(".group-button")).to_have_count(1)
    page.locator(".step-button").filter(has_text="Step1").click()
    expect(page.locator(".trajectory-header h1")).to_have_text("next-task-id")
    expect(page.locator("#step-title")).to_have_text("Training step 1")
    assert not errors


def test_live_refresh_preserves_selection_and_expansion(browser_view):
    page, root, errors = browser_view
    expect = playwright.expect
    page.locator(".reward-info > summary").first.click()
    write_sample(root, 0, "arbitrary-a.jsonl", "same-task-id", 0.3)
    write_sample(root, 2, "new-step.jsonl", "new-task-id")
    page.get_by_role("button", name="Refresh", exact=False).click()
    expect(page.locator(".step-button")).to_have_count(3)
    expect(page.locator(".headline-stats .reward")).to_have_text("0.300")
    expect(page.locator(".reward-info").first).to_have_attribute("open", "")
    expect(page.locator("#step-title")).to_have_text("Training step 0")
    assert not errors


def test_mobile_layout_has_no_horizontal_overflow(browser_view):
    page, _, errors = browser_view
    page.set_viewport_size({"width": 390, "height": 844})
    playwright.expect(page.locator(".trajectory-header h1")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert page.locator("#main").bounding_box()["height"] > 250
    assert not errors
