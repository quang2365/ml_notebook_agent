import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from model.model import llm
from schemas.pipeline_review_schema import (
    PipelineReviewResult,
)
from state import State

review_pipeline_llm = llm.with_structured_output(
    PipelineReviewResult,
    method="function_calling",
)


PIPELINE_REVIEW_SYSTEM_PROMPT = """
Bạn là chuyên gia kiểm tra Python Machine Learning notebook.

Nhiệm vụ của bạn là đánh giá tính đúng đắn của TOÀN BỘ
Machine Learning pipeline dựa trên:

1. Dataset context.
2. Target column.
3. Problem type.
4. Notebook plan.
5. Toàn bộ code cells theo đúng thứ tự thực thi.

Validator Python trước đó đã kiểm tra cấu trúc, cú pháp và một
phần dependency. Bạn phải tập trung vào lỗi NGỮ NGHĨA và lỗi
tương tác giữa các cell.

CÁC NHÓM LỖI CẦN KIỂM TRA:

1. DATA LEAKAGE

- Không được fit preprocessing trên test data.
- Không được dùng target làm feature.
- Không được sử dụng dữ liệu test để lựa chọn model.
- Việc split train/test phải diễn ra trước những bước học
  tham số từ dữ liệu.

2. PREPROCESSING

- Train và test phải dùng cùng một fitted preprocessor.
- Không được gọi fit_transform riêng trên test data.
- Feature engineering phải nhất quán giữa train và test.
- Pipeline phải tương thích với kiểu dữ liệu truyền giữa
  các bước.

3. PANDAS VÀ NUMPY COMPATIBILITY

- Nếu FunctionTransformer truy cập cột bằng cú pháp
  X["column"], đầu vào của nó phải là pandas DataFrame.
- SimpleImputer, StandardScaler hoặc ColumnTransformer
  có thể biến DataFrame thành NumPy array.
- Không được đặt FunctionTransformer dùng tên cột sau một
  transformer đã chuyển dữ liệu thành NumPy array, trừ khi
  notebook đảm bảo đầu ra pandas.
- Cấu trúc ưu tiên là:

  Pipeline([
      ("feature_engineering", FunctionTransformer(...)),
      ("columns", ColumnTransformer(...)),
  ])

4. VARIABLE CONSISTENCY

- Một biến phải được tạo trước khi sử dụng.
- Tên model phải nhất quán giữa trained_models,
  predictions và model_results.
- Không được sử dụng nhầm biến metric của model trước hoặc
  model sau.
- Không được âm thầm ghi đè dữ liệu quan trọng bằng kiểu
  dữ liệu không tương thích.

5. MODEL TRAINING

- Model phải phù hợp với regression hoặc classification.
- Model phải được fit trên đúng training features và target.
- Prediction phải dùng đúng fitted model.
- Không được tạo metric hoặc prediction giả.

6. METRICS VÀ MODEL COMPARISON

- Metric phải phù hợp với problem type.
- Mỗi model phải lưu metric riêng ngay sau khi đánh giá.
- Không được dùng metric của model cuối cùng cho tất cả model.
- Bảng model_results phải được tạo từ kết quả thực tế.
- Việc chọn best_model phải dựa trên metric phù hợp.

7. TARGET VÀ DATASET

- Target phải đúng với TARGET COLUMN đã cung cấp.
- Không được tự thay đổi dataset path.
- Không được thay đổi problem type.
- Không được đưa target vào preprocessing của features.

QUY TẮC TRẢ KẾT QUẢ:

1. Chỉ trả status="valid" khi toàn bộ pipeline nhất quán.
2. Khi status="valid", errors phải là danh sách rỗng.
3. Khi status="invalid", errors phải chứa ít nhất một lỗi.
4. Mỗi lỗi phải chỉ ra cell_id chính xác cần sửa.
5. cell_id phải tồn tại trong danh sách code cells.
6. Không yêu cầu sửa markdown cell.
7. Không báo lỗi chỉ dựa trên suy đoán không có bằng chứng.
8. suggestion phải mô tả cách sửa nhỏ nhất có thể.
9. Không viết lại code của notebook.
10. Không trả Markdown code fence.
"""


def review_pipeline_node(state: State) -> dict:
    """
    Đánh giá ngữ nghĩa toàn bộ notebook pipeline sau khi
    các cell đã vượt qua static validation.
    """

    cells = state.get("notebook_cells") or []

    #AI: Chỉ gửi code cell để giảm token và tránh markdown
    # không liên quan làm nhiễu quá trình đánh giá.
    code_cells = [
        {
            "cell_id": cell.get("cell_id"),
            "section_id": cell.get("section_id"),
            "title": cell.get("title"),
            "purpose": cell.get("purpose"),
            "source": normalize_source(
                cell.get("source")
            ),
        }
        for cell in cells
        if cell.get("cell_type") == "code"
    ]

    if not code_cells:
        message = (
            "Không có code cell để đánh giá pipeline."
        )

        return {
            "pipeline_review_status": "failed",
            "pipeline_review_errors": [
                {
                    "cell_id": "unknown",
                    "error_type": "other",
                    "message": message,
                    "suggestion": (
                        "Kiểm tra lại bước generate_cells."
                    ),
                    "related_cell_ids": [],
                }
            ],
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    review_context = {
        "dataset": {
            "dataset_path": state.get(
                "dataset_path"
            ),
            "summary": state.get("summary"),
            "summary_llm": state.get(
                "summary_llm"
            ),
        },
        "problem": {
            "target_column": state.get(
                "target_column"
            ),
            "problem_type": state.get(
                "problem_type"
            ),
            "target_analysis": state.get(
                "target_analysis"
            ),
        },
        "notebook_plan": state.get(
            "notebook_plan"
        ),
        "code_cells": code_cells,
    }

    try:
        result = review_pipeline_llm.invoke(
            [
                SystemMessage(
                    content=(
                        PIPELINE_REVIEW_SYSTEM_PROMPT
                    )
                ),
                HumanMessage(
                    content=(
                        "Hãy đánh giá pipeline sau đây:\n\n"
                        + json.dumps(
                            review_context,
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                ),
            ]
        )

        result_dict = result.model_dump()
        status = result_dict["status"]
        errors = result_dict.get("errors") or []

        #AI: Không tin hoàn toàn vào status do LLM trả về.
        # Chuẩn hóa status dựa trên danh sách lỗi thực tế.
        if errors:
            status = "invalid"
        else:
            status = "valid"

        #AI: Loại bỏ lỗi trỏ tới cell_id không tồn tại.
        valid_cell_ids = {
            cell["cell_id"]
            for cell in code_cells
            if cell.get("cell_id")
        }

        normalized_errors = []
        invalid_cell_references = []

        for review_error in errors:
            cell_id = review_error.get("cell_id")

            if cell_id not in valid_cell_ids:
                invalid_cell_references.append(cell_id)
                continue

            normalized_errors.append(
                {
                    "cell_id": cell_id,
                    "error_type": review_error.get(
                        "error_type",
                        "other",
                    ),
                    "message": review_error.get(
                        "message",
                        "",
                    ),
                    "suggestion": review_error.get(
                        "suggestion",
                        "",
                    ),
                    "related_cell_ids": [
                        related_id
                        for related_id in review_error.get(
                            "related_cell_ids",
                            [],
                        )
                        if related_id in valid_cell_ids
                    ],
                    #AI: Cho fix_cells_node biết đây là lỗi
                    # semantic do pipeline reviewer phát hiện.
                    "source": "pipeline_review",
                }
            )

        #AI: Không được biến lỗi thật thành trạng thái valid chỉ vì
        # reviewer trả về cell_id không tồn tại.
        if invalid_cell_references:
            invalid_ids = ", ".join(
                str(cell_id)
                for cell_id in invalid_cell_references
            )
            message = (
                "Pipeline reviewer trả về cell_id không tồn tại: "
                f"{invalid_ids}."
            )
            return {
                "pipeline_review_status": "failed",
                "pipeline_review_errors": [],
                "error": message,
                "messages": [AIMessage(content=message)],
            }

        #AI: Sau khi chuẩn hóa lỗi, tính lại trạng thái.
        final_status = (
            "invalid"
            if normalized_errors
            else "valid"
        )

        summary = result_dict.get(
            "summary",
            "",
        )

        return {
            "pipeline_review_status": final_status,
            "pipeline_review_errors": (
                normalized_errors
            ),
            #AI: Chuyển lỗi pipeline sang cùng nơi mà
            # fix_cells_node đang đọc.
            "validation_cell_errors": (
                normalized_errors
                if final_status == "invalid"
                else []
            ),
            "validation_cell_status": (
                "invalid"
                if final_status == "invalid"
                else "valid"
            ),
            "error": (
                None
                if final_status == "valid"
                else (
                    "LLM phát hiện "
                    f"{len(normalized_errors)} lỗi "
                    "trong Machine Learning pipeline."
                )
            ),
            "messages": [
                AIMessage(
                    content=build_review_message(
                        status=final_status,
                        summary=summary,
                        errors=normalized_errors,
                    )
                )
            ],
        }

    except Exception as exc:
        message = (
            "Không thể đánh giá notebook pipeline: "
            f"{exc}"
        )

        return {
            "pipeline_review_status": "failed",
            "pipeline_review_errors": [],
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }


def normalize_source(source: str | list | None) -> str:
    """Chuẩn hóa source của notebook cell thành chuỗi."""

    if source is None:
        return ""

    if isinstance(source, list):
        return "".join(source)

    return str(source)


def build_review_message(
    status: str,
    summary: str,
    errors: list[dict],
) -> str:
    """Tạo message ngắn để hiển thị kết quả review."""

    if status == "valid":
        return (
            "# Pipeline Review\n\n"
            "Pipeline đã vượt qua đánh giá ngữ nghĩa "
            "của LLM.\n\n"
            f"{summary}"
        )

    lines = [
        "# Pipeline Review",
        "",
        (
            f"Phát hiện **{len(errors)} lỗi "
            "ngữ nghĩa**."
        ),
        "",
    ]

    for index, review_error in enumerate(
        errors,
        start=1,
    ):
        lines.extend(
            [
                (
                    f"## {index}. "
                    f"{review_error['cell_id']}"
                ),
                (
                    "- Type: "
                    f"`{review_error['error_type']}`"
                ),
                (
                    "- Message: "
                    f"{review_error['message']}"
                ),
                (
                    "- Suggestion: "
                    f"{review_error['suggestion']}"
                ),
                "",
            ]
        )

    return "\n".join(lines)
