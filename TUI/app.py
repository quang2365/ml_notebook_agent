from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Static,
    LoadingIndicator,
)

from config.providers import PROVIDERS
from TUI.datasetpicker import DatasetPickerScreen
from TUI.review_problem import ReviewProblemScreen
from graph import build_graph
from state import create_initial_state
from langgraph.types import Command

import uuid


class QiuApp(App[None]):

    TITLE = "QIU"

    SUB_TITLE = (
        "AI Machine Learning "
        "Notebook Agent"
    )

    CSS = """
    Screen {
        align: center middle;
    }

    #main-container {
        width: 70;
        height: auto;
        padding: 2 4;
        border: round $primary;
    }

    #title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #provider-info {
        margin-bottom: 2;
    }

    #dataset-label {
        text-style: bold;
        margin-bottom: 1;
    }

    #dataset-path {
        height: auto;
        min-height: 3;
        border: round $secondary;
        padding: 1;
        margin-bottom: 1;
    }
    #loading-indicator {
        display: none;
        height: 1;
        margin-top: 1;
    }

    #processing-status {
        display: none;
        text-align: center;
        margin: 1 0;
    }
    #select-dataset {
        width: 100%;
        margin-bottom: 1;
    }

    #start-button {
        width: 100%;
    }
    """

    def __init__(
        self,
        config: dict,
    ):
        super().__init__()

        self.config = config

        self.dataset_path: (
            str | None
        ) = None

        self.thread_id = str(
            uuid.uuid4()
        )

        self.graph_config = {
            "configurable": {
                "thread_id":
                    self.thread_id
            }
        }
        self.workflow_running = False
        self.graph = build_graph()
    def compose(self) -> ComposeResult:

        provider_id = ( self.config["provider"] )

        model = ( self.config["model"] )

        if provider_id in PROVIDERS:
            provider_name = ( PROVIDERS[ provider_id ]["label"] )

        else:
            provider_name = (
                provider_id
            )

        yield Header()

        yield Vertical(
            Label(
                "QIU",
                id="title",
            ),

            Static(
                (
                    f"Provider: "
                    f"{provider_name}\n"
                    f"Model: {model}"
                ),
                id="provider-info",
            ),

            Label(
                "Dataset",
                id="dataset-label",
            ),

            Static(
                "No dataset selected",
                id="dataset-path",
            ),

            Button(
                "Select Dataset",
                id="select-dataset",
                variant="primary",
            ),
            LoadingIndicator(
                id="loading-indicator",
            ),

            Static(
                "",
                id="processing-status",
            ),
            Button(
                "Start",
                id="start-button",
                variant="success",
                disabled=True,
            ),

            id="main-container",
        )

        yield Footer()
    def on_button_pressed( self, event: Button.Pressed, ) -> None:

        button_id = ( event.button.id )

        if button_id == "select-dataset":
            self.open_dataset_picker()

        elif button_id == "start-button":
            if self.workflow_running:
                return

            self.workflow_running = True

            self.show_processing(
                "Processing dataset...\n"
                "QIU is inspecting and "
                "analyzing your data."
            )
    def open_dataset_picker( self, ) -> None:
        self.push_screen(
            DatasetPickerScreen(),
            self.handle_dataset_selected,
        )
    def handle_dataset_selected( self, dataset_path: str | None, ) -> None:
        if dataset_path is None:
            return

        self.dataset_path = (
            dataset_path
        )

        dataset_label = self.query_one(
            "#dataset-path",
            Static,
        )

        dataset_label.update(
            dataset_path
        )

        start_button = self.query_one(
            "#start-button",
            Button,
        )

        start_button.disabled = False
    def show_processing( self, message: str, ) -> None:

        loading = self.query_one(
            "#loading-indicator",
            LoadingIndicator,
        )

        status = self.query_one(
            "#processing-status",
            Static,
        )

        start_button = self.query_one(
            "#start-button",
            Button,
        )

        select_button = self.query_one(
            "#select-dataset",
            Button,
        )

        loading.styles.display = "block"
        status.styles.display = "block"

        status.update(
            message
        )

        start_button.disabled = True
        select_button.disabled = True
    def hide_processing( self, enable_start: bool = True, ) -> None:

        loading = self.query_one(
            "#loading-indicator",
            LoadingIndicator,
        )

        status = self.query_one(
            "#processing-status",
            Static,
        )

        start_button = self.query_one(
            "#start-button",
            Button,
        )

        select_button = self.query_one(
            "#select-dataset",
            Button,
        )

        loading.styles.display = "none"
        status.styles.display = "none"

        status.update("")

        start_button.disabled = (
            not enable_start
        )

        select_button.disabled = False
    def show_graph_error( self, message: str, ) -> None:
        self.workflow_running = False
        self.hide_processing()
        self.notify(
            message,
            title="Graph Error",
            severity="error",
            timeout=10,
        )
    def handle_graph_result( self, result: dict, ) -> None:

        interrupts = result.get( "__interrupt__" )
        if interrupts:
            self.hide_processing(
                enable_start=False
            )

            self.handle_interrupt(
                interrupts
            )
            return
        self.workflow_running = False

        self.hide_processing()

        approval_status = result.get(
            "approval_status"
        )

        if approval_status == "rejected":
            self.notify(
                (
                    "Machine Learning problem "
                    "proposal was rejected."
                ),
                title="QIU",
                severity="warning",
            )
            return

        self.notify(
            "Workflow completed.",
            title="QIU",
            severity="information",
        )
    def handle_interrupt( self, interrupts, ) -> None:
        interrupt_item = interrupts[0]
        self.workflow_running = False
        payload = interrupt_item.value

        self.push_screen(
            ReviewProblemScreen(
                payload=payload
            ),
            self.handle_review_decision,
        )
    def handle_review_decision( self, decision: dict | None, ) -> None:
        if decision is None:
            return
        self.show_processing(
            "Applying your decision..."
        )
        self.resume_workflow(
            decision
        )
    @work( thread=True, exclusive=True, )
    def start_workflow( self, ) -> None:

        if not self.dataset_path:
            return

        initial_state = ( create_initial_state( self.dataset_path ) )

        try:
            result = self.graph.invoke(
                initial_state,
                config=self.graph_config,
            )

        except Exception as exc:

            self.call_from_thread(
                self.show_graph_error,
                str(exc),
            )

            return

        self.call_from_thread(
            self.handle_graph_result,
            result,
        )
    @work( thread=True, exclusive=True, )
    def resume_workflow( self, decision: dict, ) -> None:
        try:
            result = self.graph.invoke(
                Command(
                    resume=decision
                ),
                config=self.graph_config,
            )

        except Exception as exc:

            self.call_from_thread(
                self.show_graph_error,
                str(exc),
            )

            return

        self.call_from_thread(
            self.handle_graph_result,
            result,
        )
