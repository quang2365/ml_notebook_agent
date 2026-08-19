from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static


class ReviewProblemScreen(ModalScreen[dict | None]):
    CSS = """
    ReviewProblemScreen {
        align: center middle;
    }

    #review-container {
        width: 80;
        height: auto;
        padding: 2 4;
        border: round $primary;
        background: $surface;
    }

    #review-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 2;
    }

    .field-title {
        text-style: bold;
        margin-top: 1;
    }

    #actions {
        margin-top: 2;
        align-horizontal: center;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, payload: dict):
        super().__init__()
        self.payload = payload
        self.proposal = payload.get("proposal") or {}

    def compose(self) -> ComposeResult:
        target = self.proposal.get("target_column") or ""
        problem_type = self.proposal.get("problem_type") or ""
        reasons = self.proposal.get("reasons") or []
        reason_text = "\n".join(str(reason) for reason in reasons)

        yield Vertical(
            Label("Review ML Problem", id="review-title"),
            Label("Target", classes="field-title"),
            Input(value=str(target), id="target-column"),
            Label("Problem Type", classes="field-title"),
            Input(value=str(problem_type), id="problem-type"),
            Label("Reasons", classes="field-title"),
            Static(reason_text or "No reason provided."),
            Horizontal(
                Button("Approve", id="approve", variant="success"),
                Button("Edit", id="edit", variant="warning"),
                Button("Reject", id="reject", variant="error"),
                id="actions",
            ),
            id="review-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "approve":
            self.dismiss({"action": "approve"})
            return

        if button_id == "reject":
            self.dismiss({"action": "reject"})
            return

        if button_id == "edit":
            target_column = self.query_one("#target-column", Input).value.strip()
            problem_type = self.query_one("#problem-type", Input).value.strip().lower()

            if not target_column or problem_type not in {
                "regression",
                "classification",
            }:
                self.notify(
                    "Enter a target and a valid problem type.",
                    severity="error",
                )
                return

            self.dismiss(
                {
                    "action": "edit",
                    "target_column": target_column,
                    "problem_type": problem_type,
                    "feedback": "Updated from QIU review screen.",
                }
            )