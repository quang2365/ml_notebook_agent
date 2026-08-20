import ast
import json
from copy import deepcopy

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from model.model import llm
from model.structured_output import build_structured_llm, invoke_structured
from schemas.fixed_cell_schema import FixedCell
from state import State
from validators.dependency_validator import (
    BUILTIN_NAMES,
    CellDependencyAnalyzer,
    validate_dependencies,
)



# The total number of rounds will be controlled by route.
runtime_fix_llm = build_structured_llm(llm,
    FixedCell,
)


RUNTIME_FIX_SYSTEM_PROMPT = """
You are an expert in fixing runtime errors in Python Machine Learning
Notebook.

The notebook has passed:

1. Structure check.
2. Syntax check.
3. Static dependency check.
4. Pipeline semantic evaluation.

However, the notebook produced an error during actual execution.

Your task is to fix only CURRENT CELL based on:

- Runtime exception.
- Traceback.
- Source of the current cell.
- Previously executed code cells.
- Variables that have been defined.
- Target column and problem type.

POSSIBLE ERRORS THAT MAY APPEAR:

1. TypeError.
2. ValueError.
3. KeyError.
4. IndexError.
5. AttributeError.
6. NameError.
7. ImportError.
8. FileNotFoundError.
9. pandas DataFrame and Series errors.
10. NumPy array errors.
11. scikit-learn pipeline errors.
12. Shape or dtype incompatibility.
13. Metric and model prediction errors.
14. Feature engineering errors.
15. Error accessing a fitted transformer.

MANDATORY RULES:

1. Fix only CURRENT CELL.

2. Do not modify or rewrite the entire notebook.

3. Do not reference any cell after CURRENT CELL.

4. Do not change:

   - dataset path;
   - target column;
   - problem type;
   - train/test split unnecessarily.

5. Do not create fake data, metrics, predictions, or models.

6. Do not create variables with None or dummy values just to make the error disappear.

7. Only use:

   - variables in AVAILABLE NAMES;
   - object trong PREVIOUS CODE CELLS;
   - variables that are validly defined in CURRENT CELL.

8. If the error is due to inconsistent variable names, prefer using the variable
   that already exists instead of creating a new variable.

9. If the error is related to a dictionary, you must use the correct key
   that was created previously.

10. If the error is related to DataFrame and NumPy:

    - Check the actual data type passing through each transformer.
    - Do not access an ndarray by column name.
    - FunctionTransformer using X["column"] must receive a DataFrame.
    - SimpleImputer, StandardScaler, and ColumnTransformer can
      return a NumPy array.
    - Feature engineering using column names should run before the transformer
      that loses column name information.

11. If the error is related to preprocessing:

    - Fit only on training data.
    - Do not fit_transform on test data.
    - Train and test must use the same fitted preprocessor.
    - Do not create data leakage.

12. If the error is related to the model:

    - The model must be fit before predicting.
    - Use the correct features and target.
    - Keep model names consistent among trained_models, predictions
      and model_results.

13. If the error is related to metrics:

    - Use y_true and y_pred in the correct order.
    - The metric must match the problem type.
    - Do not use the final model's results for every model.
    - RMSE can be calculated using:
      np.sqrt(mean_squared_error(y_true, y_pred)).

14. If the error is caused by a library that is not installed, do not insert pip
    install commands into the notebook. Do not change import to a non-equivalent
    library just to avoid the error.

15. Make the smallest possible change.

16. Keep cell_id unchanged.

17. The returned Source must be pure Python.

18. Do not use Markdown code fence.
"""


def fix_execution_cell_node(
    state: State,
) -> dict:
    """
    Fix the cell causing the error during notebook execution.
    """

    cells = state.get("notebook_cells") or []

    execution_error = (
        state.get("execution_error")
        or {}
    )

    attempts = state.get(
        "execution_fix_attempts",
        0,
    )

    new_attempts = attempts + 1

    if not cells:
        message = (
            "There are no notebook_cells to fix runtime errors."
        )

        return {
            "execution_status": "failed",
            "execution_fix_attempts": new_attempts,
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    cell_id = execution_error.get("cell_id")

    if not cell_id:
        message = (
            "Could not determine the cell_id causing the runtime error."
        )

        return {
            "execution_status": "failed",
            "execution_fix_attempts": new_attempts,
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    cell_index = find_cell_index(
        cells=cells,
        cell_id=cell_id,
    )

    if cell_index is None:
        message = (
            f"Could not find runtime cell `{cell_id}` "
            "trong notebook_cells."
        )

        return {
            "execution_status": "failed",
            "execution_fix_attempts": new_attempts,
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    current_cell = cells[cell_index]

    if current_cell.get("cell_type") != "code":
        message = (
            f"Cell `{cell_id}` is not a code cell."
        )

        return {
            "execution_status": "failed",
            "execution_fix_attempts": new_attempts,
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }

    previous_code_cells = [
        {
            "cell_id": cell.get("cell_id"),
            "section_id": cell.get("section_id"),
            "title": cell.get("title"),
            "purpose": cell.get("purpose"),
            "source": normalize_source(
                cell.get("source")
            ),
        }
        for cell in cells[:cell_index]
        if cell.get("cell_type") == "code"
    ]

    available_names = collect_available_names(
        previous_code_cells
    )

    prompt_context = {
        "target_column": state.get(
            "target_column"
        ),
        "problem_type": state.get(
            "problem_type"
        ),
        "available_names": available_names,
        "previous_code_cells": (
            previous_code_cells
        ),
        "current_cell": {
            "cell_id": current_cell.get(
                "cell_id"
            ),
            "section_id": current_cell.get(
                "section_id"
            ),
            "title": current_cell.get(
                "title"
            ),
            "purpose": current_cell.get(
                "purpose"
            ),
            "source": normalize_source(
                current_cell.get("source")
            ),
        },
        "runtime_error": {
            "error_type": execution_error.get(
                "error_type"
            ),
            "exception_name": (
                execution_error.get(
                    "exception_name"
                )
            ),
            "message": execution_error.get(
                "message"
            ),
            "traceback": execution_error.get(
                "traceback"
            ),
        },
    }

    try:
        result = invoke_structured(runtime_fix_llm, llm, FixedCell,
            [
                SystemMessage(
                    content=(
                        RUNTIME_FIX_SYSTEM_PROMPT
                    )
                ),
                HumanMessage(
                    content=(
                        "Please fix CURRENT CELL based on "
                        "runtime context sau:\n\n"
                        + json.dumps(
                            prompt_context,
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                ),
            ]
        )

        result_dict = result.model_dump()

        returned_cell_id = result_dict.get(
            "cell_id"
        )

        if returned_cell_id != cell_id:
            raise ValueError(
                "LLM changed cell_id: "
                f"{cell_id} -> {returned_cell_id}"
            )

        fixed_source = remove_code_fence(
            result_dict.get("source") or ""
        )


        compile(
            fixed_source,
            filename=cell_id,
            mode="exec",
        )

        updated_cells = deepcopy(cells)
        updated_cells[cell_index]["source"] = (
            fixed_source
        )


        dependency_errors = validate_dependencies(
            updated_cells
        )

        current_cell_errors = [
            error
            for error in dependency_errors
            if error.get("cell_id") == cell_id
        ]

        if current_cell_errors:
            raise ValueError(
                "The runtime fix produced a dependency error: "
                f"{current_cell_errors}"
            )

        return {
            "notebook_cells": updated_cells,


            "validation_cell_status": "pending",
            "validation_cell_errors": None,
            "pipeline_review_status": "pending",
            "pipeline_review_errors": None,


            "build_status": "pending",
            "build_error": None,


            # when the notebook is built and run again.
            "execution_status": "pending",
            "execution_error": None,
            "execution_fix_attempts": new_attempts,

            "fixed_cell_ids": [cell_id],
            "error": None,
            "messages": [
                AIMessage(
                    content=(
                        f"Fixed runtime cell `{cell_id}`. "
                        "Cell will be validated, reviewed, "
                        "built and re-executed."
                    )
                )
            ],
        }

    except Exception as exc:
        message = (
            f"Cannot fix runtime cell "
            f"`{cell_id}`: {exc}"
        )

        old_failures = (
            state.get("fix_cell_failures")
            or []
        )

        return {
            "execution_status": "failed",
            "execution_fix_attempts": new_attempts,
            "fix_cell_failures": [
                *old_failures,
                {
                    "cell_id": cell_id,
                    "error_type": (
                        "runtime_fix_failed"
                    ),
                    "exception_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                },
            ],
            "error": message,
            "messages": [
                AIMessage(content=message)
            ],
        }


def find_cell_index(
    cells: list[dict],
    cell_id: str,
) -> int | None:
    """Find the internal cell position using cell_id."""

    for index, cell in enumerate(cells):
        if cell.get("cell_id") == cell_id:
            return index

    return None


def normalize_source(
    source: str | list | None,
) -> str:
    """Normalize source into a Python string."""

    if source is None:
        return ""

    if isinstance(source, list):
        return "".join(source)

    return str(source)


def remove_code_fence(source: str) -> str:
    """Remove Markdown fence if LLM still returns it."""

    source = source.strip()

    if source.startswith("```python"):
        source = source[len("```python"):]

    elif source.startswith("```"):
        source = source[3:]

    if source.endswith("```"):
        source = source[:-3]

    return source.strip()


def collect_available_names(
    previous_code_cells: list[dict],
) -> list[str]:
    """
    Collect the variables, functions, and imports that have been defined
    in preceding cells.
    """

    available_names = set(BUILTIN_NAMES)

    for cell in previous_code_cells:
        source = normalize_source(
            cell.get("source")
        )

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        analyzer = CellDependencyAnalyzer()
        analyzer.visit(tree)

        available_names.update(
            analyzer.defined_names
        )

    return sorted(
        name
        for name in available_names
        if name not in BUILTIN_NAMES
    )
