"""Model capability profiles used by the structured-output gateway."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class OutputStrategy(str, Enum):
    """Available ways to request and validate structured model output."""

    FUNCTION_CALLING = "function_calling"
    JSON_MODE = "json_mode"
    PROMPT_PARSER = "prompt_parser"


@dataclass(frozen=True)
class ModelProfile:
    """Provider/model capabilities relevant to QIU's LLM gateway."""

    provider: str
    model: str
    strategy: OutputStrategy
    thinking_enabled: bool | None = None
    supports_tool_choice: bool = False
    supports_json_mode: bool = False


MODEL_REGISTRY: dict[tuple[str, str], ModelProfile] = {
    ("deepseek", "deepseek-v4-flash"): ModelProfile(
        provider="deepseek",
        model="deepseek-v4-flash",
        strategy=OutputStrategy.PROMPT_PARSER,
        thinking_enabled=False,
        supports_tool_choice=False,
        supports_json_mode=False,
    ),
    ("deepseek", "deepseek-v4-pro"): ModelProfile(
        provider="deepseek",
        model="deepseek-v4-pro",
        strategy=OutputStrategy.PROMPT_PARSER,
        thinking_enabled=False,
        supports_tool_choice=False,
        supports_json_mode=False,
    ),
}


PROVIDER_DEFAULTS: dict[str, ModelProfile] = {
    "openai": ModelProfile(
        provider="openai",
        model="",
        strategy=OutputStrategy.FUNCTION_CALLING,
        supports_tool_choice=True,
    ),
    "nvidia": ModelProfile(
        provider="nvidia",
        model="",
        strategy=OutputStrategy.FUNCTION_CALLING,
        supports_tool_choice=True,
    ),
    "deepseek": ModelProfile(
        provider="deepseek",
        model="",
        strategy=OutputStrategy.PROMPT_PARSER,
        thinking_enabled=False,
        supports_tool_choice=False,
    ),
}


DEFAULT_PROFILE = ModelProfile(
    provider="unknown",
    model="",
    strategy=OutputStrategy.PROMPT_PARSER,
)


def resolve_model_profile(
    provider: str | None,
    model: str | None,
    base_url: str | None = None,
) -> ModelProfile:
    """Resolve the safest output strategy for a provider/model pair."""

    provider_key = str(provider or "").strip().lower()
    model_key = str(model or "").strip().lower()
    base_url_key = str(base_url or "").strip().lower()

    profile = MODEL_REGISTRY.get((provider_key, model_key))

    if profile is None and (
        provider_key == "deepseek"
        or "deepseek" in model_key
        or "deepseek" in base_url_key
    ):
        profile = PROVIDER_DEFAULTS["deepseek"]

    if profile is None:
        profile = PROVIDER_DEFAULTS.get(provider_key, DEFAULT_PROFILE)

    return replace(
        profile,
        provider=provider_key or profile.provider,
        model=str(model or profile.model),
    )


def profile_from_config(config: dict | None) -> ModelProfile:
    """Resolve a profile from the persisted QIU configuration."""

    config = config or {}
    return resolve_model_profile(
        provider=config.get("provider"),
        model=config.get("model"),
        base_url=config.get("base_url"),
    )
