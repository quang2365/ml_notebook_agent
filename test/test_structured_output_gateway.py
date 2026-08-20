"""Offline tests for model capabilities and structured-output strategies."""

from __future__ import annotations

import json
import unittest
from collections import deque
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from model.capabilities import (
    ModelProfile,
    OutputStrategy,
    resolve_model_profile,
)
from model.structured_output import (
    build_structured_llm,
    invoke_structured,
    parse_json_response,
)
from schemas.fixed_cell_schema import FixedCell
from test.fakes import FakeRunnable


class TextLLM:
    """Fake text model used to test prompt-parser execution."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = deque(responses)
        self.calls: list[list] = []

    def invoke(self, messages: list) -> AIMessage:
        self.calls.append(messages)
        return self.responses.popleft()


class JsonModeLLM(TextLLM):
    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__(responses)
        self.bind_kwargs: dict = {}

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self


class NativeLLM:
    def __init__(self, runnable: FakeRunnable) -> None:
        self.runnable = runnable
        self.calls: list[tuple] = []

    def with_structured_output(self, schema, method: str):
        self.calls.append((schema, method))
        return self.runnable


def fixed_cell_json() -> str:
    return json.dumps(
        FixedCell(
            cell_id="section_1_code_1",
            source="print('fixed')",
            changes="Corrected the source.",
        ).model_dump()
    )


class ModelCapabilityTests(unittest.TestCase):
    def test_deepseek_uses_prompt_parser_profile(self) -> None:
        profile = resolve_model_profile(
            "deepseek",
            "deepseek-v4-flash",
            "https://api.deepseek.com/v1",
        )

        self.assertEqual(profile.strategy, OutputStrategy.PROMPT_PARSER)
        self.assertFalse(profile.thinking_enabled)
        self.assertFalse(profile.supports_tool_choice)

    def test_nvidia_uses_function_calling_profile(self) -> None:
        profile = resolve_model_profile(
            "nvidia",
            "nvidia/minimax-3",
            "https://integrate.api.nvidia.com/v1",
        )

        self.assertEqual(profile.strategy, OutputStrategy.FUNCTION_CALLING)
        self.assertTrue(profile.supports_tool_choice)


class StructuredOutputGatewayTests(unittest.TestCase):
    def test_builds_native_function_calling_runnable(self) -> None:
        fake_runnable = FakeRunnable([FixedCell(
            cell_id="section_1_code_1",
            source="print('ok')",
            changes="No changes.",
        )])
        fake_llm = NativeLLM(fake_runnable)

        with patch(
            "model.structured_output.load_config",
            return_value={
                "provider": "nvidia",
                "model": "nvidia/minimax-3",
                "base_url": "https://example.com/v1",
            },
        ):
            runnable = build_structured_llm(fake_llm, FixedCell)

        self.assertIs(runnable, fake_runnable)
        self.assertEqual(fake_llm.calls[0][1], "function_calling")

    def test_prompt_parser_removes_thinking_and_validates_schema(self) -> None:
        fake_llm = TextLLM([
            AIMessage(
                content=(
                    "<think>internal reasoning</think>\n"
                    f"```json\n{fixed_cell_json()}\n```"
                )
            )
        ])

        with patch(
            "model.structured_output.load_config",
            return_value={
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com/v1",
            },
        ):
            result = invoke_structured(
                runnable=None,
                llm=fake_llm,
                schema=FixedCell,
                messages=[HumanMessage(content="Fix the cell.")],
            )

        self.assertIsInstance(result, FixedCell)
        self.assertEqual(result.cell_id, "section_1_code_1")
        self.assertIn("JSON schema", fake_llm.calls[0][0].content)

    def test_prompt_parser_retries_invalid_json_once(self) -> None:
        fake_llm = TextLLM([
            AIMessage(content="not json"),
            AIMessage(content=fixed_cell_json()),
        ])

        profile = ModelProfile(
            provider="custom",
            model="text-model",
            strategy=OutputStrategy.PROMPT_PARSER,
        )
        result = invoke_structured(
            runnable=None,
            llm=fake_llm,
            schema=FixedCell,
            messages=[HumanMessage(content="Fix the cell.")],
            profile=profile,
        )

        self.assertIsInstance(result, FixedCell)
        self.assertEqual(len(fake_llm.calls), 2)
        self.assertIn("previous response was invalid", fake_llm.calls[1][0].content)

    def test_json_mode_uses_response_format_and_validates(self) -> None:
        fake_llm = JsonModeLLM([
            AIMessage(content=fixed_cell_json()),
        ])
        profile = ModelProfile(
            provider="json-provider",
            model="json-model",
            strategy=OutputStrategy.JSON_MODE,
        )

        result = invoke_structured(
            runnable=None,
            llm=fake_llm,
            schema=FixedCell,
            messages=[HumanMessage(content="Fix the cell.")],
            profile=profile,
        )

        self.assertIsInstance(result, FixedCell)
        self.assertEqual(
            fake_llm.bind_kwargs,
            {"response_format": {"type": "json_object"}},
        )

    def test_native_capability_error_falls_back_to_prompt_parser(self) -> None:
        native_runnable = FakeRunnable([
            ValueError(
                "Thinking mode does not support this tool_choice"
            )
        ])
        fake_llm = TextLLM([
            AIMessage(content=fixed_cell_json()),
        ])
        profile = ModelProfile(
            provider="nvidia",
            model="nvidia/minimax-3",
            strategy=OutputStrategy.FUNCTION_CALLING,
        )

        result = invoke_structured(
            runnable=native_runnable,
            llm=fake_llm,
            schema=FixedCell,
            messages=[HumanMessage(content="Fix the cell.")],
            profile=profile,
        )

        self.assertIsInstance(result, FixedCell)
        self.assertEqual(len(native_runnable.calls), 1)
        self.assertEqual(len(fake_llm.calls), 1)

    def test_parse_json_response_accepts_nested_objects(self) -> None:
        response = AIMessage(
            content=(
                "The answer is:\n"
                '{"cell_id":"section_1_code_1",'
                '"source":"values = {\\"key\\": 1}",'
                '"changes":"Keep nested object."}'
            )
        )

        result = parse_json_response(response, FixedCell)

        self.assertEqual(result.cell_id, "section_1_code_1")
        self.assertIn("key", result.source)


if __name__ == "__main__":
    unittest.main()
