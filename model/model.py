from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI

from config.managers import load_config
from model.capabilities import ModelProfile, profile_from_config
from security.api_key_store import get_api_key
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class ModelRuntime:
    """The configured LLM and its provider capability profile."""

    llm: Any
    profile: ModelProfile


def create_model_runtime() -> ModelRuntime:
    config = load_config()

    if not config:
        raise RuntimeError(
            "QIU has not been configured. "
            "Please run `qiu setup` first."
        )

    provider = config.get(
        "provider"
    )

    model = config.get(
        "model"
    )

    base_url = config.get(
        "base_url"
    )

    if not provider:
        raise RuntimeError(
            "Config is missing provider."
        )

    if not model:
        raise RuntimeError(
            "Config is missing model."
        )

    api_key = get_api_key(
        provider
    )

    if not api_key:
        raise RuntimeError(
            f"Could not find API key "
            f"cho provider `{provider}`."
        )
    profile = profile_from_config(config)

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = (
            base_url
        )
    if profile.thinking_enabled is not None:
        kwargs["extra_body"] = {
            "thinking": {
                "type": (
                    "enabled"
                    if profile.thinking_enabled
                    else "disabled"
                ),
            },
        }

    return ModelRuntime(
        llm=ChatOpenAI(**kwargs),
        profile=profile,
    )


def create_llm() -> ChatOpenAI:
    return create_model_runtime().llm


runtime = create_model_runtime()
llm = runtime.llm
model_profile = runtime.profile
