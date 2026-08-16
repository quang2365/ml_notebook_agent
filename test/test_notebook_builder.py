"""Offline tests for conversion and notebook file creation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nodes.notebook_builder_node import notebook_builder
from tools.json_to_cell_tool import json_object_to_cell, json_string_to_cells
from test.fakes import make_agent_cell


class JsonCellConversionTests(unittest.TestCase):
    def test_object_to_code_cell(self) -> None:
        result = json_object_to_cell(make_agent_cell())

        self.assertEqual(result["cell_type"], "code")
        self.assertEqual(result["execution_count"], None)
        self.assertEqual(result["outputs"], [])
        self.assertIsInstance(result["source"], list)

    def test_json_string_with_cells_wrapper(self) -> None:
        payload = json.dumps(
            {
                "cells": [make_agent_cell()],
            }
        )

        result = json_string_to_cells(payload)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["cell_type"], "code")


class NotebookBuilderTests(unittest.TestCase):
    def test_builder_writes_valid_ipynb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "offline_test.ipynb"
            state = {
                "notebook_cells": [make_agent_cell()],
                "notebook_path": str(output_path),
            }

            result = notebook_builder(state)

            self.assertEqual(result["build_status"], "success")
            self.assertIsNone(result["build_error"])
            self.assertTrue(output_path.exists())

            notebook = json.loads(
                output_path.read_text(encoding="utf-8")
            )
            self.assertEqual(notebook["nbformat"], 4)
            self.assertEqual(len(notebook["cells"]), 1)
            self.assertEqual(notebook["cells"][0]["cell_type"], "code")


if __name__ == "__main__":
    unittest.main()
