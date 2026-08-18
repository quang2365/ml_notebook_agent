from pathlib import Path
import os
import string
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import  Button, DirectoryTree, Label, Select
def get_windows_drives() -> list[tuple[str, str]]:
    drives = []

    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"

        if os.path.exists(drive):
            drives.append(
                (
                    drive,
                    drive,
                )
            )

    return drives
class DatasetDirectoryTree(DirectoryTree):
    def filter_paths(self,paths):
        allowed_extensions = {
            ".csv",
            ".xlsx",
            ".xls",
            ".parquet",
        }

        result = []

        for path in paths:
            if path.is_dir(): result.append(path)

            elif (
                path.suffix.lower()
                in allowed_extensions
            ):
                result.append(
                    path
                )

        return result


class DatasetPickerScreen(ModalScreen[str | None]):
    CSS = """
    DatasetPickerScreen {
        align: center middle;
    }

    #picker-container {
        width: 90%;
        height: 90%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }

    #picker-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #drive-select {
        width: 100%;
        margin-bottom: 1;
    }

    #dataset-tree {
        height: 1fr;
    }

    #cancel-button {
        margin-top: 1;
    }
    """

    def compose( self, ) -> ComposeResult:
        drives = ( get_windows_drives() )
        default_drive = (
            drives[0][1]
            if drives
            else str(
                Path.home()
            )
        )

        yield Vertical(
            Label( "Select Dataset", id="picker-title", ),
            Select( drives, value=default_drive, id="drive-select", ),
            DatasetDirectoryTree( default_drive, id="dataset-tree", ),
            Button( "Cancel", id="cancel-button", ),
            id="picker-container",
        )

    def on_directory_tree_file_selected( self, event: DirectoryTree.FileSelected, ) -> None:
        path = event.path
        self.dismiss(str(path))


    def on_button_pressed( self, event: Button.Pressed, ) -> None:
        if (
            event.button.id
            == "cancel-button"
        ):
            self.dismiss(None)
    def on_select_changed( self, event: Select.Changed, ) -> None:

        if ( event.select.id != "drive-select" ):
            return

        if ( event.value is Select.BLANK ):
            return

        tree = self.query_one( "#dataset-tree", DatasetDirectoryTree, )

        tree.path = Path( str(event.value) )
