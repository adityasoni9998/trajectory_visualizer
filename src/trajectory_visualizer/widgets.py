from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


class PathDialog(ModalScreen[str | None]):
    """Ask for a trajectory filename/path in the selected training step."""

    CSS = """
    PathDialog {
        align: center middle;
    }
    PathDialog > Vertical {
        width: 90%;
        max-width: 90;
        height: auto;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    PathDialog Label {
        margin-bottom: 1;
    }
    PathDialog Input {
        margin-bottom: 1;
    }
    PathDialog Button {
        width: 100%;
    }
    """

    def __init__(self, step: str) -> None:
        super().__init__()
        self.step = step

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(
                f"Open a .jsonl trajectory inside training step {self.step}\n"
                "Paste an absolute path, filename, or events/filename."
            )
            yield Input(placeholder="events_...jsonl", id="path-input")
            yield Button("Open", variant="primary", id="open")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(self.query_one(Input).value)

    def key_escape(self) -> None:
        self.dismiss(None)


class StepDialog(ModalScreen[str | None]):
    """Jump directly to a numeric training step."""

    CSS = PathDialog.CSS

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Jump to training step")
            yield Input(placeholder="e.g. 42", type="integer", id="step-input")
            yield Button("Go", variant="primary", id="go")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(self.query_one(Input).value)

    def key_escape(self) -> None:
        self.dismiss(None)
