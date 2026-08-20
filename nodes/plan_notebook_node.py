import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from validators.plan_validator import validate_notebook_plan

from model.model import llm
from model.structured_output import (
    build_structured_llm,
    invoke_structured,
)
from schemas.notebook_plan_schema import NotebookPlan
from state import State


structured_llm = build_structured_llm(llm, NotebookPlan)


def _invoke_plan(messages: list) -> NotebookPlan:
    return invoke_structured(
        runnable=structured_llm,
        llm=llm,
        schema=NotebookPlan,
        messages=messages,
    )



def plan_notebook_node(state: State) -> dict:
    summary = state.get("summary")
    target_analysis = state.get("target_analysis")
    target_column = state.get("target_column")
    problem_type = state.get("problem_type")

    if not summary:
        return {
            "notebook_plan": None,
            "error": "Dataset summary was not found.",
            "messages": [
                AIMessage(
                    content="Unable to create the notebook plan."
                )
            ],
        }

    if not target_analysis:
        return {
            "notebook_plan": None,
            "error": "The target has not been analyzed.",
            "messages": [
                AIMessage(
                    content="Unable to create the notebook plan."
                )
            ],
        }

    system_prompt = """
        You are a Senior Data Scientist specializing in designing
        Jupyter Notebook plans for Machine Learning problems.

        Your task is to create a NOTEBOOK PLAN.
        DO NOT write complete code.

        The notebook must be concise, clearly structured
        and executable sequentially from top to bottom.

        MANDATORY RULES:

        1. The notebook must contain 8 to 10 sections.

        2. section_id values must be sequential:
        section_1
        section_2
        section_3
        ...

        3. Do not create unnecessary sections or split a
        small task into too many sections.

        4. The Machine Learning pipeline order must be logical.

        5. The dataset must be loaded before use.

        6. Basic EDA must occur before modeling.

        7. Features X and target y must be separated.

        8. The train/test split must occur BEFORE any
        preprocessing steps that learn parameters from data, such as:
        - imputation
        - scaling
        - encoding
        - feature selection
        - dimensionality reduction
        - learned transformation

        9. The preprocessor may only be fit on the training set.

        10. Do not fit preprocessing on the entire
            the dataset before the train/test split.

        11. Feature engineering based on fixed formulas
            may be described separately, but every transformation
            that learns parameters must be fit on the training set.

        12. There must be a baseline model.

        13. There must be at least two candidate models to
            compare when appropriate for the problem.

        14. Metrics must match the problem_type.

        15. There must be a model evaluation and comparison section.

        16. There must be a conclusion section.

        17. Each section should contain 1 to 5 tasks.

        18. Do not create fake metrics or results.

        19. Do not execute code.

        20. Do not change the confirmed target_column or
            problem_type that has been confirmed.
        SUGGESTED STRUCTURE:

        section_1:
        - Set up the environment and import libraries.

        section_2:
        - Load the dataset and inspect its basic structure.

        section_3:
        - Exploratory Data Analysis.

        section_4:
        - Define X/y and perform the train/test split.

        section_5:
        - Preprocessing and feature engineering.

        section_6:
        - Baseline model.

        section_7:
        - First candidate model.

        section_8:
        - Second candidate model.

        section_9:
        - Evaluate, compare, and analyze models.

        section_10:
        - Conclusion and future directions.

        Sections may be merged when appropriate,
        but the total number of sections must be 8 to 10.
        """

    analysis_context = (
        summary.get("analysis_context")
        or summary
    )

    user_prompt = f"""
Create a notebook plan for the following problem.

Target:
{target_column}

Problem type:
{problem_type}

Dataset information:
{json.dumps(
    analysis_context,
    ensure_ascii=False,
    default=str,
)}

Return data matching the NotebookPlan schema. Do not include explanations outside the requested structured output.
Target analysis:
{json.dumps(
    target_analysis,
    ensure_ascii=False,
    default=str,
)}
"""

    try:
        plan = _invoke_plan(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        plan_dict = plan.model_dump()
        
        return {
                "notebook_plan": plan_dict,
                "plan_validation_status": "pending",
                "plan_validation_errors": None,
                "error": None,
                "messages": [
                    AIMessage(
                        content=build_plan_message(plan_dict)
                    )
                ],
            }
    except Exception as exc:
        error_message = (
            f"Unable to create the notebook plan: {exc}"
        )

        return {
            "notebook_plan": None,
            "error": error_message,
            "messages": [
                AIMessage(content=error_message)
            ],
        }


def build_plan_message(plan: dict) -> str:
    lines = [
        "# Notebook Plan",
        "",
        f"## {plan.get('notebook_title')}",
        "",
        f"**Objective:** {plan.get('objective')}",
        "",
        f"**Target:** `{plan.get('target_column')}`",
        "",
        f"**Problem type:** `{plan.get('problem_type')}`",
        "",
        "## Planned Models",
        "",
    ]

    for model_name in plan.get(
        "candidate_models",
        [],
    ):
        lines.append(f"- {model_name}")

    lines.extend(
        [
            "",
            "## Evaluation Metrics",
            "",
        ]
    )

    for metric in plan.get(
        "evaluation_metrics",
        [],
    ):
        lines.append(f"- {metric}")

    lines.extend(
        [
            "",
            "## Notebook Sections",
            "",
        ]
    )

    for index, section in enumerate(
        plan.get("sections", []),
        start=1,
    ):
        lines.extend(
            [
                f"### {index}. {section.get('title')}",
                "",
                section.get("objective", ""),
                "",
            ]
        )

        for task in section.get("tasks", []):
            lines.append(f"- {task}")

        lines.append("")

    return "\n".join(lines)
