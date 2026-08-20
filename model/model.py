from langchain_openai import ChatOpenAI

from config.managers import load_config
from security.api_key_store import get_api_key
from dotenv import load_dotenv

load_dotenv()

def create_llm() -> ChatOpenAI:
    config = load_config()

    if not config:
        raise RuntimeError(
            "QIU chưa được cấu hình. "
            "Hãy chạy `qiu setup` trước."
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
            "Config thiếu provider."
        )

    if not model:
        raise RuntimeError(
            "Config thiếu model."
        )

    api_key = get_api_key(
        provider
    )

    if not api_key:
        raise RuntimeError(
            f"Không tìm thấy API key "
            f"cho provider `{provider}`."
        )
    kwargs = {
        "model": model,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = (
            base_url
        )
    return ChatOpenAI(
        **kwargs
    )
llm = create_llm()
