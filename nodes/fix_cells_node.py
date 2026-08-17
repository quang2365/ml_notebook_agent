from copy import deepcopy

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import ast
import json
from validators.dependency_validator import (
    BUILTIN_NAMES,
    CellDependencyAnalyzer,
    validate_dependencies,  #AI
)
from model.model import llm
from schemas.fixed_cell_schema import FixedCell
from state import State


#AI: Graph quản lý các vòng retry; mỗi lần vào node chỉ gọi LLM một lần.
MAX_CELL_FIX_RETRIES = 1


fix_llm = llm.with_structured_output(
    FixedCell,
    method="function_calling",  #AI: tương thích NVIDIA và DeepSeek
)


SYSTEM_PROMPT = """
Bạn là chuyên gia Python và Machine Learning.

Nhiệm vụ của bạn là sửa MỘT notebook code cell dựa trên lỗi
được cung cấp và ngữ cảnh của các code cell đứng trước nó.

CÁC LOẠI LỖI CÓ THỂ NHẬN:

1. syntax_error
2. undefined_variable
3. data_leakage
4. pipeline_incompatibility
5. inconsistent_variable
6. invalid_model_flow
7. invalid_metric
8. invalid_preprocessing
9. invalid_feature_engineering
10. invalid_evaluation
11. wrong_problem_type
12. wrong_target
13. other

QUY TẮC BẮT BUỘC:

1. Chỉ sửa CURRENT CELL và chỉ sửa những vấn đề liên quan
   trực tiếp đến VALIDATION ERRORS được cung cấp.

2. Không viết lại toàn bộ notebook, không sửa các cell đứng
   trước hoặc giả định rằng các cell đứng sau đã được chạy.

3. Không thay đổi target column.

4. Không thay đổi dataset path.

5. Không tạo metric hoặc kết quả giả.

6. Không tạo biến tùy tiện chỉ để làm lỗi biến mất.

7. Với syntax_error:

   - Sửa cú pháp bằng thay đổi nhỏ nhất có thể.
   - Không thay đổi mục đích Machine Learning của cell.

8. Với undefined_variable hoặc inconsistent_variable:

   - Kiểm tra AVAILABLE NAMES.
   - Kiểm tra PREVIOUS CODE CONTEXT.
   - Nếu biến lỗi chỉ là tên không nhất quán,
     ưu tiên sử dụng biến đã tồn tại.
   - Nếu thực sự cần định nghĩa biến mới,
     chỉ tạo khi logic Machine Learning yêu cầu.

   - Không được tạo biến bằng None hoặc giá trị giả chỉ để
     validator không còn báo lỗi.

9. Với data_leakage:

   - Chỉ fit preprocessor, feature selector và model trên
     training data.
   - Không dùng test data để fit, chọn feature, tuning hoặc
     lựa chọn model.
   - Không đưa target column vào tập features.
   - Test data chỉ được transform và predict bằng object đã
     fit trên training data.

10. Với pipeline_incompatibility, invalid_preprocessing hoặc
    invalid_feature_engineering:

    - Theo dõi kiểu dữ liệu đi qua từng bước của pipeline.
    - SimpleImputer, StandardScaler và ColumnTransformer có
      thể trả về NumPy array.
    - Nếu FunctionTransformer truy cập X["column"], nó phải
      chạy khi X vẫn là pandas DataFrame.
    - Ưu tiên đặt feature engineering dùng tên cột trước
      ColumnTransformer:

      Pipeline([
          ("feature_engineering", FunctionTransformer(...)),
          ("columns", ColumnTransformer(...)),
      ])

    - Không đặt FunctionTransformer dùng tên cột sau bước đã
      chuyển DataFrame thành NumPy array, trừ khi đầu ra pandas
      được bảo đảm rõ ràng.
    - Train và test phải dùng cùng một fitted preprocessor.
    - Không gọi fit_transform trên test data.

11. Với invalid_model_flow:

    - Model phải được fit trước khi predict.
    - Mỗi model phải sử dụng đúng preprocessing và đúng dữ
      liệu train/test.
    - Tên model phải nhất quán giữa trained_models,
      predictions và model_results.
    - Không thay một model bằng model khác nếu lỗi không yêu
      cầu việc đó.

12. Với invalid_metric hoặc invalid_evaluation:

    - Sử dụng metric phù hợp với PROBLEM TYPE.
    - Regression có thể dùng MAE, RMSE, R2 và MAPE khi hợp lệ.
    - Classification có thể dùng accuracy, precision, recall,
      F1, ROC-AUC hoặc metric phù hợp với dữ liệu.
    - RMSE phải được tính đúng theo phiên bản thư viện hiện có,
      ví dụ np.sqrt(mean_squared_error(y_true, y_pred)).
    - Mỗi model phải lưu metric riêng ngay sau khi đánh giá.
    - Không tái sử dụng metric của model cuối cho các model khác.
    - Không tạo metric, prediction hoặc model_results giả.

13. Với wrong_problem_type hoặc wrong_target:

    - Tuân thủ chính xác TARGET COLUMN và PROBLEM TYPE trong
      context.
    - Không tự ý đổi target hoặc loại bài toán.
    - Nếu CURRENT CELL dùng nhầm target, sửa về target đã được
      xác nhận bằng những biến thực sự có trong context.

14. Đọc trường suggestion trong lỗi như một gợi ý, không coi
    đó là mệnh lệnh tuyệt đối. Chỉ áp dụng nếu phù hợp với code
    và PREVIOUS CODE CELLS.

15. Phải giữ nhất quán với các object đã được tạo trong
    PREVIOUS CODE CELLS. Không được tham chiếu cell đứng sau
    CURRENT CELL.

16. Nếu giá trị cần lấy từ dictionary, phải sử dụng đúng key
    đã được tạo trước đó.

17. Preprocessing có học tham số chỉ được fit trên training
    data. Không tạo data leakage dưới bất kỳ hình thức nào.

18. Giữ thay đổi nhỏ nhất có thể và bảo toàn mục đích ban đầu
    của CURRENT CELL.

19. Source trả về phải là Python thuần, không dùng Markdown
    code fence và phải giữ nguyên cell_id.

20. Nếu lỗi không thể sửa chỉ bằng CURRENT CELL mà bắt buộc
    phải thay đổi một cell trước đó, không được bịa biến hoặc
    kết quả để che lỗi. Hãy giữ code hợp lý nhất có thể; hệ
    thống sẽ phát hiện nếu bản sửa không giải quyết được lỗi.
"""

def fix_cells_node(
    state: State,
) -> dict:

    cells = state.get("notebook_cells") or []
    validation_cell_errors = (
        state.get("validation_cell_errors")
        or []
    )

    fix_cell_attempts = state.get(
        "fix_cell_attempts",
        0,
    )

    if not cells:
        message = (
            "Không có notebook_cells để sửa."
        )

        return {
            "error": message,
            "fix_cell_attempts": fix_cell_attempts + 1,
            "messages": [
                AIMessage(content=message)
            ],
        }

    if not validation_cell_errors:
        message = (
            "Không có validation_cell_errors để sửa."
        )
        new_fix_attempts = fix_cell_attempts + 1
        return {
            "validation_cell_status": "invalid",
            "validation_cell_errors": [],
            "fix_cell_attempts": new_fix_attempts,
            "fixed_cell_ids": [],
            "fix_cell_failures": [],
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    # Không mutate trực tiếp state cũ
    updated_cells = deepcopy(cells)

    # cell_id -> cell
    cell_map = {
        cell.get("cell_id"): cell
        for cell in updated_cells
        if cell.get("cell_id")
    }

    # Gom lỗi theo cell_id
    errors_by_cell = group_errors_by_cell(
        validation_cell_errors
    )

    fixed_cell_ids = []
    failed_cell_ids = []
    fix_cell_failures = []
    for cell_id, errors in errors_by_cell.items():
        cell_index = next(
            (
                index
                for index, item in enumerate(updated_cells)
                if item.get("cell_id") == cell_id
            ),
            None,
        )

        #AI: Phải kiểm tra trước khi dùng cell_index để slice danh sách.
        if cell_index is None:
            failed_cell_ids.append(cell_id)
            continue

        previous_code_cells = [
            {
                "cell_id": item.get("cell_id"),
                "section_id": item.get("section_id"),
                "title": item.get("title"),
                "source": item.get("source"),
            }
            for item in updated_cells[:cell_index]
            if item.get("cell_type") == "code"
        ]
        cell = cell_map.get(cell_id)

        if not cell:
            failed_cell_ids.append(cell_id)
            continue

        # Markdown không cần sửa
        if cell.get("cell_type") != "code":
            continue

        try:
            available_names = collect_available_names(
                previous_code_cells
            )

            fixed_source = fix_single_cell(
                cell=cell,
                errors=errors,
                previous_code_cells=previous_code_cells,
                available_names=available_names,
                target_column=state.get("target_column"),
                problem_type=state.get("problem_type"),
            )

            # Kiểm tra syntax ngay sau khi LLM sửa
            compile(
                fixed_source,
                filename=cell_id,
                mode="exec",
            )

            #AI: Compile thành công chưa chứng minh undefined_variable đã hết.
            # Thay source trên một bản sao và validate dependency toàn notebook.
            candidate_cells = deepcopy(updated_cells)
            candidate_cells[cell_index]["source"] = fixed_source
            remaining_errors = validate_dependencies(candidate_cells)
            remaining_cell_errors = [
                error
                for error in remaining_errors
                if error.get("cell_id") == cell_id
            ]

            if remaining_cell_errors:
                raise ValueError(
                    "Bản sửa vẫn còn lỗi dependency: "
                    f"{remaining_cell_errors}"
                )

            #AI: Chỉ ghi nhận bản sửa sau khi syntax và dependency cùng hợp lệ.
            cell["source"] = fixed_source

            fixed_cell_ids.append(
                cell_id
            )

        except Exception as exc:
            failed_cell_ids.append(
                cell_id
            )

            fix_cell_failures.append(
                {
                    "cell_id": cell_id,
                    "title": cell.get("title"),
                    "exception_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                }
            )

    new_fix_attempts = fix_cell_attempts + 1

    message = build_fix_message(
        fix_attempt=new_fix_attempts,
        fixed_cell_ids=fixed_cell_ids,
        failed_cell_ids=failed_cell_ids,
    )

    return {
        "notebook_cells": updated_cells,

        # Validator tiếp theo sẽ xác định lại
        "validation_cell_status": "pending",  #AI: đặt lại để validator kiểm tra
        "validation_cell_errors": None,

        "fix_cell_attempts": new_fix_attempts,
        "fixed_cell_ids": fixed_cell_ids,
        "fix_cell_failures": fix_cell_failures,
        "error": (
            None
            if not failed_cell_ids
            else (
                "Một số cell chưa sửa "
                "thành công."
            )
        ),

        "messages": [
            AIMessage(content=message)
        ],
    }
def group_errors_by_cell(
    errors: list[dict],
) -> dict[str, list[dict]]:

    grouped = {}

    for error in errors:
        cell_id = error.get("cell_id")

        if not cell_id:
            continue

        if cell_id not in grouped:
            grouped[cell_id] = []

        grouped[cell_id].append(
            error
        )

    return grouped
def fix_single_cell(
    cell: dict,
    errors: list[dict],
    previous_code_cells: list[dict],
    available_names: list[str],
    target_column: str | None,
    problem_type: str | None,
) -> str:

    cell_id = cell["cell_id"]
    source = cell["source"]

    current_source = source
    current_errors = errors
    last_syntax_error = None
    for attempt in range(1,MAX_CELL_FIX_RETRIES + 1,):
        error_text = format_errors(current_errors)
        prompt = f"""
            TARGET COLUMN:
            {target_column}

            PROBLEM TYPE:
            {problem_type}

            AVAILABLE NAMES:
            {json.dumps(
                available_names,
                ensure_ascii=False,
            )}

            PREVIOUS CODE CELLS:{json.dumps(previous_code_cells,ensure_ascii=False,default=str,)}

            CURRENT CELL:{json.dumps({
                    "cell_id": cell_id,
                    "title": cell.get("title"),
                    "purpose": cell.get("purpose"),
                    "source": current_source,},
                ensure_ascii=False,
                default=str,)}
            VALIDATION ERRORS:
            {error_text}

            NHIỆM VỤ:

            1. Chỉ sửa CURRENT CELL.
            2. Không tạo lại toàn bộ notebook.
            3. Chỉ sử dụng biến có trong AVAILABLE NAMES
            hoặc biến được định nghĩa trong CURRENT CELL.
            4. Nếu một tên biến không tồn tại, tìm biến có vai trò
            tương ứng trong PREVIOUS CODE CELLS.
            5. Không tự tạo biến giả chỉ để validator cho qua.
            6. Giữ nguyên cell_id.
            7. Source trả về phải là Python thuần.
            8. Không dùng Markdown code fence.
            """

        result = fix_llm.invoke(
            [
                SystemMessage(
                    content=SYSTEM_PROMPT
                ),
                HumanMessage(
                    content=prompt
                ),
            ]
        )

        result_dict = (
            result.model_dump()
        )

        # LLM không được thay ID
        if (
            result_dict["cell_id"]
            != cell_id
        ):
            raise ValueError(
                "LLM thay đổi cell_id: "
                f"{cell_id} -> "
                f"{result_dict['cell_id']}"
            )

        candidate_source = (
            result_dict["source"]
        )

        # Không chấp nhận Markdown fence
        candidate_source = (
            remove_code_fence(
                candidate_source
            )
        )

        try:
            compile(
                candidate_source,
                filename=cell_id,
                mode="exec",
            )

            return candidate_source

        except SyntaxError as exc:
            last_syntax_error = exc
            # Lần retry kế tiếp sẽ dùng
            # chính lỗi mới này
            current_source = (
                candidate_source
            )

            current_errors = [
                {
                    "cell_id": cell_id,
                    "line": exc.lineno,
                    "offset": exc.offset,
                    "message": exc.msg,
                    "error_line": (
                        exc.text.strip()
                        if exc.text
                        else None
                    ),
                }
            ]

    if last_syntax_error:
        raise RuntimeError(
            f"Cell `{cell_id}` vẫn lỗi sau "
            f"{MAX_CELL_FIX_RETRIES} lần. "
            f"Line {last_syntax_error.lineno}: "
            f"{last_syntax_error.msg}. "
            f"Code: "
            f"{last_syntax_error.text!r}"
        )

    raise RuntimeError(
        f"Không thể sửa cell `{cell_id}`."
    )
def format_errors(
    errors: list[dict],
) -> str:
    lines = []

    for error in errors:
        lines.append(
            f"Type: {error.get('error_type')}"
        )

        if error.get("line"):
            lines.append(
                f"Line: {error.get('line')}"
            )

        lines.append(
            f"Message: {error.get('message')}"
        )

        suggestion = error.get("suggestion")

        if suggestion:
            lines.append(
                f"Suggestion: {suggestion}"
            )

        related_cell_ids = error.get(
            "related_cell_ids"
        )

        if related_cell_ids:
            lines.append(
                "Related cells: "
                + ", ".join(related_cell_ids)
            )

        error_line = error.get("error_line")

        if error_line:
            lines.append(
                f"Problematic line: {error_line}"
            )

        lines.append("")

    return "\n".join(lines)
def remove_code_fence(
    source: str,
) -> str:

    source = source.strip()

    if source.startswith("```python"):
        source = source[
            len("```python"):
        ]

    elif source.startswith("```"):
        source = source[3:]

    if source.endswith("```"):
        source = source[:-3]

    return source.strip()
def build_fix_message(
    fix_attempt: int,
    fixed_cell_ids: list[str],
    failed_cell_ids: list[str],
) -> str:

    lines = [
        "# Notebook Cell Repair",
        "",
        f"**Fix round:** {fix_attempt}",
        "",
        (
            f"**Sửa thành công:** "
            f"{len(fixed_cell_ids)} cell"
        ),
        (
            f"**Chưa sửa được:** "
            f"{len(failed_cell_ids)} cell"
        ),
    ]

    if fixed_cell_ids:
        lines.extend(
            [
                "",
                "## Cells đã sửa",
            ]
        )

        for cell_id in fixed_cell_ids:
            lines.append(
                f"- `{cell_id}`"
            )

    if failed_cell_ids:
        lines.extend(
            [
                "",
                "## Cells chưa sửa được",
            ]
        )

        for cell_id in failed_cell_ids:
            lines.append(
                f"- `{cell_id}`"
            )

    lines.extend(
        [
            "",
            (
                "Các cell sẽ được đưa trở lại "
                "`validate_cells`."
            ),
        ]
    )

    return "\n".join(lines)

def collect_available_names(
    previous_code_cells: list[dict],
) -> list[str]:
    available_names = set(BUILTIN_NAMES)

    for cell in previous_code_cells:
        source = cell.get("source") or ""

        if isinstance(source, list):
            source = "".join(source)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        analyzer = CellDependencyAnalyzer()
        analyzer.visit(tree)

        available_names.update(
            analyzer.defined_names
        )

    # Không cần gửi toàn bộ builtin cho LLM.
    return sorted(
        name
        for name in available_names
        if name not in BUILTIN_NAMES
    )
