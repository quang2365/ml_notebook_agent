from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from state import State
from model.model import llm
def analyze_dataset_note(state:State) -> dict:
    # bản phân tích đầy đủ dataset
    dataset_summary = state.get("summary")
    # bản phân tích tóm tắt
    analys_context = dataset_summary.get('analysis_context')
    if not dataset_summary:
        return {"error":" Tôi không nhận được bản tóm tắt của Dataset",
                "messages": AIMessage(content = "Tôi không nhận được bản tóm tắt của Dataset")}
    system_prompt = """
    Bạn là một chuyên gia tóm tắt và phân tích dữ liêu.
    bạn sẽ được cung cấp các dữ liệu liên quan đến dataset hãy tóm tắt nội dung ấy cho người dùng
    chỉ trình bày từ các thông tin được cung cấp, không bịa hay suy luận.
    
    quy ước chung:
    Khi trình bày số tiền, hãy phân biệt rõ dấu phân cách hàng nghìn và
    dấu thập phân. Có thể viết theo đơn vị nghìn USD để tránh nhầm lẫn.
    
        Hãy phân tích:

        1. Kích thước và cấu trúc dataset.
        2. Cột số và cột phân loại.
        3. Missing value và duplicate.
        4. Phân phối lệch và outlier.
        5. Những tương quan đáng chú ý.
        6. Cột có khả năng là ID hoặc không hữu ích.
        7. Đề xuất target và loại bài toán Machine Learning.
        8. Những bước cần thực hiện tiếp theo.

        Quy tắc:
        - Chỉ sử dụng dữ liệu được cung cấp.
        - Không khẳng định target nếu người dùng chưa xác nhận.
        - Outlier theo IQR chỉ là dấu hiệu thống kê, không mặc định là dữ liệu sai.
        - Tương quan không đồng nghĩa với quan hệ nhân quả.
        - Trình bày bằng Markdown tiếng Việt
    """
    message=f"""Bạn là một Data Scientist.

                    Đây là thông tin dataset:
                    {analys_context}"""

    try:
        result = llm.invoke([SystemMessage(content= system_prompt),HumanMessage(message)])
        return {
            "messages": [result],
            "summary_llm": result.content
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "messages": [AIMessage(content=f"Có lỗi gì đó khi tôi đang cố đọc dataset: {str(exc)}")]
        }
    