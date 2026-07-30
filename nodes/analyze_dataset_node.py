from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from state import State
from model.model import llm
from langchain.agents import create_agent
def analyze_dataset_note(state:State) -> dict:
    dataset_summary = state.get("summary")
    systemprompt = """
    "Bạn là một chuyên gia tóm tắt và phân tích dữ liêu.
    bạn sẽ được cung cấp các dữ liệu liên quan đến dataset hãy tóm tắt nội dung ấy cho người dùng
    chỉ trình bày từ các thông tin được cung cấp, không bịa hay suy luận.
    
    quy ước chung:
    Khi trình bày số tiền, hãy phân biệt rõ dấu phân cách hàng nghìn và
    dấu thập phân. Có thể viết theo đơn vị nghìn USD để tránh nhầm lẫn.
    """
    message=f"""Bạn là một Data Scientist.

                    Đây là thông tin dataset:
                    {dataset_summary}
                    Hãy phân tích:
                    - Dataset lớn hay nhỏ
                    - Có bao nhiêu cột số
                    - Có bao nhiêu cột phân loại
                    - Dự đoán đây là bài toán gì
                    - Viết khoảng 5 câu."""
    if not dataset_summary:
        return {"error":" Tôi không nhận được bản tóm tắt của Dataset",
                "messages": AIMessage(content = "Tôi không nhận được bản tóm tắt của Dataset")}
    try:
        result = llm.invoke([HumanMessage(message),SystemMessage(content=)])
        return {
            "messages":[AIMessage(content=result.content)],
            "summary_llm": result.content
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "messages": [AIMessage(content=f"Có lỗi gì đó khi tôi đang cố đọc dataset: {str(exc)}")]
        }
    