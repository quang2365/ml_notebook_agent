from textual.app import App, ComposeResult
from textual.containers import  Vertical
from textual.widgets import Button, Footer, Header, Label, Static
from config.providers import  PROVIDERS
from TUI.datasetpicker import DatasetPickerScreen


class QiuApp(
    App[None]
):
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

    #select-dataset {
        width: 100%;
        margin-bottom: 1;
    }

    #start-button {
        width: 100%;
    }
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        self.dataset_path: (str | None) = None

    def compose( self, ) -> ComposeResult:
        provider_id = ( self.config[ "provider"] )
        model = self.config[
            "model"
        ]
        if ( provider_id in PROVIDERS ):
            provider_name = ( PROVIDERS[ provider_id ]["label"] )
        else:
            provider_name = ( provider_id )

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

        if ( button_id == "select-dataset" ):
            self.open_dataset_picker()

        elif ( button_id == "start-button" ):
            self.start_workflow()
    def open_dataset_picker(self) -> None:
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

        dataset_label = (
            self.query_one(
                "#dataset-path",
                Static,
            )
        )

        dataset_label.update(
            dataset_path
        )

        start_button = (
            self.query_one(
                "#start-button",
                Button,
            )
        )

        start_button.disabled = False
    def start_workflow( self, ) -> None:

        if not self.dataset_path:
            return

        self.notify(
            (
                "Dataset selected:\n"
                f"{self.dataset_path}"
            ),
            title="QIU",
        )