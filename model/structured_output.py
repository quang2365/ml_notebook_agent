from __future__ import annotations

import json
import re
from typing import Any

from config.managers import load_config
from langchain_core.messages import SystemMessage

from model.capabilities import (
    ModelProfile,
    OutputStrategy,
    profile_from_config,
)


class StructuredOutputError(ValueError):
    """Raised when a model response cannot be converted to a schema."""


def get_active_model_profile() -> ModelProfile:
    return profile_from_config(load_config())


def is_deepseek_config() -> bool:
    return get_active_model_profile().provider == "deepseek"


def build_structured_llm(
    llm: Any,
    schema: Any,
    profile: ModelProfile | None = None,
) -> Any:
    """Build native structured output only for supported profiles."""

    profile = profile or get_active_model_profile()
    if profile.strategy != OutputStrategy.FUNCTION_CALLING:
        return None

    return llm.with_structured_output(
        schema,
        method="function_calling",
    )


def _response_content(response: Any) -> str:
    """Read final content without parsing reasoning metadata."""

    if isinstance(response, str):
        return response

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "".join(parts)

    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def _remove_reasoning_tags(content: str) -> str:
    return re.sub(
        r"<(?:think|thinking)>.*?</(?:think|thinking)>",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def extract_json_payload(content: str) -> dict | list:
    """Extract the first valid JSON object or array from model text."""

    cleaned = _remove_reasoning_tags(content)
    decoder = json.JSONDecoder()

    for index, character in enumerate(cleaned):
        if character not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, (dict, list)):
            return payload

    raise StructuredOutputError(
        "The model did not return a valid JSON object or array."
    )


def parse_json_response(response: Any, schema: Any) -> Any:
    """Extract JSON from a response and validate it with Pydantic."""

    if isinstance(response, schema):
        return response

    payload = extract_json_payload(_response_content(response))
    try:
        return schema.model_validate(payload)
    except Exception as exc:
        raise StructuredOutputError(
            f"The JSON response does not match {schema.__name__}: {exc}"
        ) from exc


def _prompt_messages(messages: list, schema: Any, error: str | None = None) -> list:
    schema_json = json.dumps(
        schema.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    instruction = (
        "Return only one valid JSON object matching this schema. "
        "Do not return Markdown, explanations, or tool calls.\n\n"
        f"JSON schema:\n{schema_json}"
    )
    if error:
        instruction += (
            "\n\nYour previous response was invalid. "
            f"Correct this problem and return JSON only:\n{error}"
        )
    return [SystemMessage(content=instruction), *messages]


def _is_capability_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "tool_choice",
            "tool choice",
            "response_format",
            "function calling",
            "structured output",
            "not support",
            "unsupported",
        )
    )


def _invoke_prompt_parser(
    llm: Any,
    schema: Any,
    messages: list,
    max_attempts: int = 2,
) -> Any:
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        prompt_messages = _prompt_messages(
            messages,
            schema,
            error=str(last_error) if last_error else None,
        )
        try:
            return parse_json_response(
                llm.invoke(prompt_messages),
                schema,
            )
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                break

    raise StructuredOutputError(
        f"Unable to produce valid structured output: {last_error}"
    ) from last_error


def invoke_structured(
    runnable: Any,
    llm: Any,
    schema: Any,
    messages: list,
    profile: ModelProfile | None = None,
) -> Any:
    """Invoke a model and always return a validated Pydantic object."""

    profile = profile or get_active_model_profile()

    if runnable is not None:
        try:
            return runnable.invoke(messages)
        except Exception as exc:
            if not _is_capability_error(exc):
                raise

    if profile.strategy == OutputStrategy.JSON_MODE:
        try:
            response = llm.bind(
                response_format={"type": "json_object"}
            ).invoke(messages)
            return parse_json_response(response, schema)
        except Exception as exc:
            if not _is_capability_error(exc):
                raise

    return _invoke_prompt_parser(
        llm=llm,
        schema=schema,
        messages=messages,
    )
