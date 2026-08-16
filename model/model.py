"""LLM configuration with NVIDIA as default and optional DeepSeek support."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv(override=True)


NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def create_llm(
    use_deepseek: bool | None = None,
) -> ChatOpenAI:
    """
    Create the selected chat model.

    DeepSeek is used only when explicitly requested. If no choice is supplied,
    USE_DEEPSEEK from the environment is checked and defaults to false.
    """
    if use_deepseek is None:
        use_deepseek = _env_flag("USE_DEEPSEEK")

    if use_deepseek:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Đã chọn DeepSeek nhưng DEEPSEEK_API_KEY chưa có trong .env."
            )

        return ChatOpenAI(
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key=api_key,
            #AI: Structured output của LangChain ép tool_choice cụ thể.
            # DeepSeek V4 chỉ chấp nhận cách này khi tắt Thinking mode.
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
        )

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY chưa có trong .env."
        )

    return ChatOpenAI(
        model=NVIDIA_MODEL,
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
    )


def selected_model_name(
    use_deepseek: bool | None = None,
) -> str:
    """Return the model name selected by argument or environment."""
    if use_deepseek is None:
        use_deepseek = _env_flag("USE_DEEPSEEK")

    return DEEPSEEK_MODEL if use_deepseek else NVIDIA_MODEL


# Existing nodes keep importing this shared object.
llm = create_llm()
