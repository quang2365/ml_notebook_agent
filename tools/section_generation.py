import json
import time

from langchain_core.messages import HumanMessage, SystemMessage

from model.model import llm
from schemas.notebook_cell_schema import GeneratedSection
from state import State


section_llm = llm.with_structured_output(
    GeneratedSection,
    method="function_calling",
)


SECTION_SYSTEM_PROMPT = """
You are an expert in Python, Machine Learning, and Jupyter Notebook design.

Your task is to generate notebook cells for ONLY ONE PROVIDED SECTION.

MANDATORY RULES:
1. Generate only cells belonging to the current section.
2. Do not generate content for other sections.
3. Each section should contain 2 to 5 cells.
4. Include at least one Markdown cells describing the section.
5. Python code must be valid and executable sequentially.
6. Do not put Python code inside a Markdown code fence.
7. Do not use variables that were not declared in previous steps.
8. Do not change the dataset path or target column.
9. Do not execute code or create fake outputs or metrics.
10. Use random_state=42 when appropriate and avoid data leakage.
11. Fit preprocessing on the train set when appropriate.
12. The section_id of every cell must match the provided section_id, and cell_id must be unique and start with section_id.
13. Strictly follow the provided VARIABLE CONTRACT.
14. Reuse exactly the variable names, data types, Pipeline step names, and structures in PREVIOUS NOTEBOOK CODE.
15. Do not redefine an existing variable with a different meaning.
16. Use only variables created in the current or previous sections.
17. Every Pipeline must use the standard step names preprocessor and model.
18. Keys in model_results must remain consistent: model, mae, rmse, r2.
19. model_results must always be list[dict] and initialized exactly once.
20. Each model must append metrics immediately after evaluation.
21. The model key must be identical in trained_models, predictions, and model_results.
22. The comparison section only reads model_results and must not reassign it.
23. RMSE must use np.sqrt(mean_squared_error(...)).
"""
NOTEBOOK_VARIABLE_CONTRACT = """
COMMON VARIABLE CONTRACT FOR THE ENTIRE NOTEBOOK:

Use df, target_column, X, y, X_train, X_test, y_train, y_test, preprocessor, trained_models = {}, predictions = {}, and model_results = [] consistently across all sections.
model_results must always be list[dict].
Build comparison tables with pd.DataFrame(model_results).set_index("model") when needed.
Do not reassign model_results to a dict.
Do not rename standard variables or change their data types.
Use Pipeline([("preprocessor", preprocessor), ("model", estimator)]) and access the estimator with pipeline.named_steps["model"].
Use RANDOM_STATE = 42 where supported. Use rmse = np.sqrt(mean_squared_error(y_true, y_pred)).
Do not use variables from later sections.
Always read data with df = pd.read_csv(dataset_path), using the exact provided dataset_path.
"""
MAX_SECTION_RETRIES = 3
RATE_LIMIT_BASE_DELAY = 15
NORMAL_RETRY_DELAY = 3


def build_dataset_context(state: State) -> dict:
    """Build the compact dataset context shared by every section."""
    summary = state.get("summary") or {}
    target_analysis = state.get("target_analysis") or {}

    return {
        "column_names": summary.get("column_names"),
        "numeric_columns": summary.get("numeric_columns"),
        "categorical_columns": summary.get("categorical_columns"),
        "possible_id_columns": summary.get("possible_id_columns"),
        "target_analysis": target_analysis,
    }


def generate_one_section(
    section_id: str,
    section: dict,
    dataset_path: str,
    target_column: str,
    problem_type: str,
    dataset_context: dict,
    previous_code_cells: list[dict],
) -> dict:
    """Generate and validate one section, retrying transient failures."""
    user_prompt = f"""
Create notebook cells for the following section.

SECTION ID:
{section_id}

SECTION PLAN:
{json.dumps(section, ensure_ascii=False, default=str, indent=2)}

DATASET PATH:
{dataset_path}

TARGET COLUMN:
{target_column}

PROBLEM TYPE:
{problem_type}

DATASET CONTEXT:
{json.dumps(dataset_context, ensure_ascii=False, default=str)}

PREVIOUS NOTEBOOK CODE:
{json.dumps(previous_code_cells, ensure_ascii=False, default=str)}

VARIABLE CONTRACT:
{NOTEBOOK_VARIABLE_CONTRACT}
"""
    last_error = None

    for attempt in range(1, MAX_SECTION_RETRIES + 1):
        try:
            generated_section = section_llm.invoke(
                [
                    SystemMessage(content=SECTION_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            generated_dict = generated_section.model_dump()
            returned_section_id = generated_dict.get("section_id")

            if returned_section_id != section_id:
                raise ValueError(
                    "LLM returned the wrong section_id: "
                    f"{returned_section_id}. Expected: {section_id}"
                )

            cells = generated_dict.get("cells") or []
            if not cells:
                raise ValueError(
                    f"LLM generated no cells for section `{section_id}`."
                )

            seen_cell_ids: set[str] = set()
            for cell in cells:
                cell_id = cell.get("cell_id")
                cell_section_id = cell.get("section_id")

                if not cell_id:
                    raise ValueError(
                        f"Section `{section_id}` has a cell without cell_id."
                    )

                if cell_section_id != section_id:
                    raise ValueError(
                        f"Cell `{cell_id}` has section_id "
                        f"`{cell_section_id}` but expected is `{section_id}`."
                    )

                if cell_id in seen_cell_ids:
                    raise ValueError(
                        f"cell_id `{cell_id}` is duplicated in "
                        f"section `{section_id}`."
                    )

                seen_cell_ids.add(cell_id)

            return {
                "status": "success",
                "section_id": section_id,
                "cells": cells,
                "error": None,
            }

        except Exception as exc:
            last_error = exc
            print(
                f"[{section_id}] attempt {attempt}/"
                f"{MAX_SECTION_RETRIES} failed: {exc}"
            )

            if attempt < MAX_SECTION_RETRIES:
                error_text = str(exc)
                if "429" in error_text or "Too Many Requests" in error_text:
                    wait_time = RATE_LIMIT_BASE_DELAY * (2 ** (attempt - 1))
                    print(
                        f"[{section_id}] NVIDIA rate limit. "
                        f"Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    time.sleep(NORMAL_RETRY_DELAY)

    return {
        "status": "failed",
        "section_id": section_id,
        "cells": [],
        "error": str(last_error),
    }
