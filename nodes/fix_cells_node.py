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
    validate_dependencies,
)
from model.model import llm
from schemas.fixed_cell_schema import FixedCell
from state import State



MAX_CELL_FIX_RETRIES = 1


fix_llm = llm.with_structured_output(
    FixedCell,
    method="function_calling",
)


SYSTEM_PROMPT = """
You are a Python and Machine Learning expert.

Your task is to fix ONE notebook code cell based on the error
provided and the context of the code cells preceding it.

POSSIBLE ERROR TYPES:

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

MANDATORY RULES:

1. Only modify the CURRENT CELL and only fix issues related
   directly to the provided VALIDATION ERRORS.

2. Do not rewrite the entire notebook, do not modify preceding cells
   or assume that the following cells have already run.

3. Do not change the target column.

4. Do not change the dataset path.

5. Do not create fake metrics or results.

6. Do not create arbitrary variables just to make the error disappear.

7. For syntax_error:

   - Fix the syntax with the smallest possible change.
   - Do not change the Machine Learning purpose of the cell.

8. For undefined_variable or inconsistent_variable:

   - Check AVAILABLE NAMES.
   - Check PREVIOUS CODE CONTEXT.
   - If the erroneous variable is only an inconsistent name,
     prefer using an existing variable.
   - If you truly need to define a new variable,
     only create it when the Machine Learning logic requires it.

   - Do not create a variable with None or a dummy value just to
     stop the validator from reporting the error.

9. For data_leakage:

   - Only fit the preprocessor, feature selector, and the model above
     training data.
   - Do not use test data to fit, select features, tune, or
     select models.
   - Do not include the target column in the feature set.
   - Test data may only be transformed and predicted using objects already
     fitted on training data.

10. For pipeline_incompatibility, invalid_preprocessing or
    invalid_feature_engineering:

    - Track the data type through each step of the pipeline.
    - SimpleImputer, StandardScaler, and ColumnTransformer can
      return a NumPy array.
    - If FunctionTransformer accesses X["column"], it must
      run while X is still a pandas DataFrame.
    - Prefer to put feature engineering that uses column names first
      ColumnTransformer:

      Pipeline([
          ("feature_engineering", FunctionTransformer(...)),
          ("columns", ColumnTransformer(...)),
      ])

    - Do not place a column-name-based FunctionTransformer after a step that has
      converted the DataFrame to a NumPy array, unless the pandas output
      is clearly guaranteed.
    - Train and test must use the same fitted preprocessor.
    - Do not call fit_transform on test data.

11. For invalid_model_flow:

    - The model must be fitted before predicting.
    - Each model must use the correct preprocessing and the correct
      train/test data.
    - Model names must be consistent across trained_models,
      predictions, and model_results.
    - Do not replace one model with another if the error does not require
      it.

12. For invalid_metric or invalid_evaluation:

    - Use a metric appropriate for the PROBLEM TYPE.
    - Regression may use MAE, RMSE, R2, and MAPE when valid.
    - Classification may use accuracy, precision, recall,
      F1, ROC-AUC, or a metric appropriate for the data.
    - RMSE must be calculated correctly according to the current library version,
      e.g., np.sqrt(mean_squared_error(y_true, y_pred)).
    - Each model must store its own metric immediately after evaluation.
    - Do not reuse the last model's metric for other models.
    - Do not create fake metrics, predictions, or model_results.

13. For wrong_problem_type or wrong_target:

    - Strictly follow the TARGET COLUMN and PROBLEM TYPE in
      context.
    - Do not arbitrarily change the target or problem type.
    - If CURRENT CELL uses the wrong target, change it to the target that has been
      confirmed by variables that actually exist in the context.

14. Read the suggestion field in the error as a suggestion; do not treat
    it as an absolute command. Apply it only if it fits the code
    and PREVIOUS CODE CELLS.

15. Must stay consistent with the objects created in
    PREVIOUS CODE CELLS. Do not reference later cells.
    CURRENT CELL.

16. If a value needs to be retrieved from a dictionary, use the exact key
    that was created earlier.

17. Preprocessing that learns parameters may only be fitted on training
    data. Do not create data leakage in any form.

18. Keep the change as minimal as possible and preserve the original purpose
    of the CURRENT CELL.

19. The returned Source must be pure Python, not using Markdown
    code fence, and must preserve the cell_id.

20. If the error cannot be fixed using only the CURRENT CELL and it is mandatory
    to change a previous cell, do not invent variables or
    results to hide the error. Keep the code as reasonable as possible; the system
    will detect if the fix does not resolve the error.
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
            "No notebook_cells to fix."
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
            "No validation_cell_errors to fix."
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

    # Do not directly mutate old state
    updated_cells = deepcopy(cells)

    # cell_id -> cell
    cell_map = {
        cell.get("cell_id"): cell
        for cell in updated_cells
        if cell.get("cell_id")
    }

    # Group errors by cell_id
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

        # Markdown does not need fixing
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

            # Check syntax immediately after the LLM fixes it
            compile(
                fixed_source,
                filename=cell_id,
                mode="exec",
            )


            # Replace source on a copy and validate dependency for the whole notebook.
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
                    "The dependency error remains after the fix: "
                    f"{remaining_cell_errors}"
                )


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

        # The next validator will re-validate
        "validation_cell_status": "pending",
        "validation_cell_errors": None,

        "fix_cell_attempts": new_fix_attempts,
        "fixed_cell_ids": fixed_cell_ids,
        "fix_cell_failures": fix_cell_failures,
        "error": (
            None
            if not failed_cell_ids
            else (
                "Some cells were not fixed "
                "successfully."
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

            TASK:

            1. Only modify the CURRENT CELL.
            2. Do not recreate the entire notebook.
            3. Only use variables present in AVAILABLE NAMES
            or variables defined in the CURRENT CELL.
            4. If a variable name does not exist, find a variable with a role
            corresponding to it in PREVIOUS CODE CELLS.
            5. Do not create fake variables just to make the validator pass.
            6. Preserve the cell_id.
            7. The returned Source must be pure Python.
            8. Do not use Markdown code fence.
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

        # LLM must not change the ID
        if (
            result_dict["cell_id"]
            != cell_id
        ):
            raise ValueError(
                "LLM changed cell_id: "
                f"{cell_id} -> "
                f"{result_dict['cell_id']}"
            )

        candidate_source = (
            result_dict["source"]
        )

        # Do not accept Markdown fence
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
            # The next retry will use
            # this exact new error
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
            f"Cell `{cell_id}` still fails after "
            f"{MAX_CELL_FIX_RETRIES} times. "
            f"Line {last_syntax_error.lineno}: "
            f"{last_syntax_error.msg}. "
            f"Code: "
            f"{last_syntax_error.text!r}"
        )

    raise RuntimeError(
        f"Could not fix cell `{cell_id}`."
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
            f"**Fixed successfully:** "
            f"{len(fixed_cell_ids)} cell"
        ),
        (
            f"**Not fixed:** "
            f"{len(failed_cell_ids)} cell"
        ),
    ]

    if fixed_cell_ids:
        lines.extend(
            [
                "",
                "## Fixed cells",
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
                "## Unfixed cells",
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
                "The cells will be put back "
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

    # No need to send all builtins to the LLM.
    return sorted(
        name
        for name in available_names
        if name not in BUILTIN_NAMES
    )
