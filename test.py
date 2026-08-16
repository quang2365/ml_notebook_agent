from pathlib import Path

from nodes.notebook_builder_node import notebook_builder


file_path = Path("./test/json_samples/generated_notebook.json")
state = {
    "notebook_cells": [
        {
            "cell_id": "section_1_code_1",
            "section_id": "section_1",
            "cell_type": "code",
            "title": "Load data",
            "source": "print('hello')",
            "purpose": "Test cell",
            "expected_output": None,
        }
    ],
    "notebook_path": "test/output/test_notebook.ipynb",
}
notebook_data = file_path.read_text(encoding="utf-8")
print(notebook_builder(state))
