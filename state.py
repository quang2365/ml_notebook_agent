from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    #message của người dùng/ hệ thông/ Tool/ AI
    messages: Annotated[list[BaseMessage],add_messages]

    #đường dẫn của dataset
    dataset_path: str | None

    #kết quả tổng quan dataset
    summary: dict | None

    #kết quả tổng quan dataset thông qua llm
    summary_llm: str | None

    #Lỗi xảy ra khi đọc phân tích dataset
    error: str | None