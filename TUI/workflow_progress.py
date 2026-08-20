from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


NODE_LABELS = {
    "inspect_data": "Inspecting dataset",
    "analyze_data": "Analyzing dataset",
    "propose_problem": "Proposing ML problem",
    "review_problem": "Waiting for problem confirmation",
    "analyze_target": "Analyzing target",
    "plan_notebook": "Planning notebook",
    "validate_plan_node": "Validating notebook plan",
    "fix_plan_node": "Repairing notebook plan",
    "prepare_generation": "Preparing section generation",
    "generate_section": "Generating notebook section",
    "validate_cells": "Validating notebook cells",
    "review_pipeline": "Reviewing ML pipeline",
    "fix_cells": "Repairing notebook cells",
    "notebook_builder": "Building Jupyter notebook",
    "execute_notebook": "Executing notebook",
    "fix_execution_cell": "Repairing runtime error",
}


class WorkflowDashboard(Vertical):
    """Live progress and final summary for the graph workflow."""

    DEFAULT_CSS = """
    WorkflowDashboard {
        height: auto;
        min-height: 12;
        border: round $secondary;
        padding: 1 2;
        margin-top: 1;
    }

    #workflow-stage {
        text-style: bold;
        margin-bottom: 1;
    }

    #workflow-summary {
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.completed_nodes: list[str] = []
        self.current_node: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("Ready", id="workflow-stage")
        yield Static("No workflow has been started.", id="workflow-summary")

    def reset(self) -> None:
        self.completed_nodes.clear()
        self.current_node = None
        self.query_one("#workflow-stage", Static).update("Ready")
        self.query_one("#workflow-summary", Static).update(
            "No workflow has been started."
        )

    def update_node(self, node_name: str) -> None:
        if node_name == "__interrupt__":
            return
        if self.current_node and self.current_node != node_name:
            if self.current_node not in self.completed_nodes:
                self.completed_nodes.append(self.current_node)
        self.current_node = node_name
        label = NODE_LABELS.get(node_name, node_name)
        completed = "\n".join(
            f"✓ {NODE_LABELS.get(node, node)}"
            for node in self.completed_nodes[-8:]
        )
        current = f"● {label}"
        text = f"{completed}\n{current}" if completed else current
        self.query_one("#workflow-stage", Static).update(text)

    def update_summary(self, state: dict) -> None:
        lines: list[str] = []
        target = state.get("target_column")
        problem_type = state.get("problem_type")
        if target or problem_type:
            lines.append(f"Target: {target or 'pending'}")
            lines.append(f"Problem type: {problem_type or 'pending'}")

        plan = state.get("notebook_plan") or {}
        sections = plan.get("sections") or []
        if plan:
            lines.append(f"Notebook plan: {len(sections)} sections")

        cells = state.get("notebook_cells") or []
        if cells:
            lines.append(f"Generated cells: {len(cells)}")

        generated_sections = state.get("generated_section_ids") or []
        if generated_sections or sections:
            current_index = state.get("current_section_index", 0)
            lines.append(
                f"Sections generated: {len(generated_sections)}"
                f" / {len(sections)} (next index: {current_index})"
            )

        repair_counts = (
            ("Plan repairs", state.get("fix_plan_attempts")),
            ("Cell repairs", state.get("fix_cell_attempts")),
            ("Pipeline repairs", state.get("pipeline_fix_attempts")),
            ("Execution repairs", state.get("execution_fix_attempts")),
        )
        lines.extend(
            f"{label}: {count}"
            for label, count in repair_counts
            if count
        )

        error_counts = (
            ("Plan errors", state.get("plan_validation_errors")),
            ("Cell errors", state.get("validation_cell_errors")),
            ("Pipeline errors", state.get("pipeline_review_errors")),
        )
        lines.extend(
            f"{label}: {len(errors)}"
            for label, errors in error_counts
            if errors
        )

        statuses = (
            ("Plan", state.get("plan_validation_status")),
            ("Generation", state.get("section_generation_status")),
            ("Cells", state.get("validation_cell_status")),
            ("Pipeline review", state.get("pipeline_review_status")),
            ("Build", state.get("build_status")),
            ("Execution", state.get("execution_status")),
        )
        lines.extend(
            f"{label}: {status}"
            for label, status in statuses
            if status
        )

        notebook_path = state.get("notebook_path")
        if notebook_path:
            lines.append(f"Output: {notebook_path}")

        error = state.get("error")
        if error:
            lines.append(f"Error: {error}")

        self.query_one("#workflow-summary", Static).update(
            "\n".join(lines) or "Workflow is preparing..."
        )

    def mark_waiting_for_review(self) -> None:
        self.query_one("#workflow-stage", Static).update(
            "● Waiting for your confirmation"
        )

    def mark_finished(self, state: dict) -> None:
        if self.current_node and self.current_node not in self.completed_nodes:
            self.completed_nodes.append(self.current_node)
        self.current_node = None
        self.update_summary(state)
        self.query_one("#workflow-stage", Static).update(
            "✓ Workflow finished"
        )
