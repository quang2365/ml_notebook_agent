import json
from uuid import uuid4  #AI

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from rich.console import Console
from rich.markdown import Markdown

from graph import build_graph


console = Console()


def print_json(
    title: str,
    data,
) -> None:
    """In dictionary/list đẹp ra terminal."""

    console.rule(f"[bold cyan]{title}")

    console.print(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def print_last_message(
    result: dict,
) -> None:
    """In message cuối cùng trong State."""

    messages = result.get("messages") or []

    if not messages:
        return

    content = messages[-1].content

    console.rule("[bold green]Last Message")

    if isinstance(content, str):
        console.print(
            Markdown(content)
        )
    else:
        console.print(content)


def main() -> None:
    # =========================================
    # 1. BUILD GRAPH
    # =========================================
    graph = build_graph()

    config = {
        "configurable": {
            "thread_id": f"run-{uuid4()}"  #AI
        },
        "recursion_limit": 50,  #AI
    }

    # =========================================
    # 2. INITIAL STATE
    # =========================================
    initial_state = {
        "messages": [
            HumanMessage(
                content=(
                    "Hãy phân tích dataset này "
                    "và xây dựng notebook machine learning."
                )
            )
        ],

        "dataset_path": "./data/housing.csv",

        "summary": None,
        "summary_llm": None,

        "problem_proposal": None,

        "target_column": None,
        "problem_type": None,

        "approval_status": None,
        "user_feedback": None,

        "target_analysis": None,

        "notebook_plan": None,
        "plan_validation_status": None,  #AI
        "plan_validation_errors": None,  #AI
        "plan_fix_attempts": 0,  #AI
        "notebook_cells": None,
        "notebook_path": "output/test.ipynb",  #AI
        "build_status": "pending",  #AI

        "validation_status": None,
        "validation_errors": None,

        "generation_status": None,

        "fix_attempts": 0,
        "fixed_cell_ids": None,
        "fix_failures": None,

        "error": None,
    }

    # =========================================
    # 3. RUN GRAPH ĐẾN HITL
    # =========================================
    console.rule(
        "[bold yellow]PHASE 1 - RUN TO INTERRUPT"
    )

    result = graph.invoke(
        initial_state,
        config=config,
    )

    print_last_message(result)

    # =========================================
    # 4. KIỂM TRA INTERRUPT
    # =========================================
    interrupts = result.get(
        "__interrupt__",
        [],
    )

    if not interrupts:
        console.print(
            "[red]Graph không dừng tại interrupt.[/red]"
        )

        print_json(
            "Current State",
            result,
        )

        return

    payload = interrupts[0].value

    print_json(
        "Human Review Payload",
        payload,
    )

    # =========================================
    # 5. AUTO APPROVE / EDIT CHO TEST
    # =========================================
    #
    # Với housing.csv, ta chủ động chọn:
    #
    # target = median_house_value
    # problem = regression
    #
    # Sau này có thể thay bằng input() nếu muốn.
    # =========================================

    review_decision = {
        "action": "edit",
        "target_column": (
            "median_house_value"
        ),
        "problem_type": "regression",
        "feedback": (
            "Test tự động: xác nhận "
            "median_house_value làm target."
        ),
    }

    print_json(
        "Review Decision",
        review_decision,
    )

    # =========================================
    # 6. RESUME GRAPH
    # =========================================
    console.rule(
        "[bold yellow]PHASE 2 - RESUME GRAPH"
    )

    final_result = graph.invoke(
        Command(
            resume=review_decision
        ),
        config=config,
    )

    print_last_message(
        final_result
    )

    # =========================================
    # 7. KIỂM TRA TARGET
    # =========================================
    console.rule(
        "[bold cyan]TARGET"
    )

    console.print(
        "Target:",
        final_result.get(
            "target_column"
        ),
    )

    console.print(
        "Problem type:",
        final_result.get(
            "problem_type"
        ),
    )

    # =========================================
    # 8. KIỂM TRA NOTEBOOK PLAN
    # =========================================
    notebook_plan = final_result.get(
        "notebook_plan"
    )

    console.rule(
        "[bold cyan]NOTEBOOK PLAN"
    )

    if notebook_plan:
        console.print(
            "Notebook title:",
            notebook_plan.get(
                "notebook_title"
            ),
        )

        console.print(
            "Sections:",
            len(
                notebook_plan.get(
                    "sections",
                    [],
                )
            ),
        )
    else:
        console.print(
            "[red]Không có notebook_plan[/red]"
        )

    # =========================================
    # 9. KIỂM TRA GENERATED CELLS
    # =========================================
    notebook_cells = final_result.get(
        "notebook_cells"
    ) or []

    console.rule(
        "[bold cyan]GENERATED CELLS"
    )

    console.print(
        "Total cells:",
        len(notebook_cells),
    )

    code_cells = [
        cell
        for cell in notebook_cells
        if cell.get("cell_type")
        == "code"
    ]

    markdown_cells = [
        cell
        for cell in notebook_cells
        if cell.get("cell_type")
        == "markdown"
    ]

    console.print(
        "Code cells:",
        len(code_cells),
    )

    console.print(
        "Markdown cells:",
        len(markdown_cells),
    )

    console.rule("[bold cyan]NOTEBOOK BUILDER")  #AI

    console.print(
        "Build status:",
        final_result.get("build_status"),  #AI
    )

    console.print(
        "Notebook path:",
        final_result.get("notebook_path"),  #AI
    )

    # =========================================
    # 10. VALIDATION RESULT
    # =========================================
    validation_status = (
        final_result.get(
            "validation_status"
        )
    )

    validation_errors = (
        final_result.get(
            "validation_errors"
        )
        or []
    )

    console.rule(
        "[bold cyan]VALIDATION"
    )

    console.print(
        "Status:",
        validation_status,
    )

    console.print(
        "Errors:",
        len(
            validation_errors
        ),
    )

    for error in validation_errors:
        console.print(
            (
                f"[red]"
                f"{error.get('cell_id')}"
                f"[/red]"
                " -> "
                f"{error.get('message')}"
            )
        )

    # =========================================
    # 11. FIX CELLS RESULT
    # =========================================
    console.rule(
        "[bold magenta]FIX CELLS"
    )

    fix_attempts = final_result.get(
        "fix_attempts",
        0,
    )

    fixed_cell_ids = (
        final_result.get(
            "fixed_cell_ids"
        )
        or []
    )

    fix_failures = (
        final_result.get(
            "fix_failures"
        )
        or []
    )

    console.print(
        "Fix attempts:",
        fix_attempts,
    )

    console.print(
        "Fixed cells:",
        len(
            fixed_cell_ids
        ),
    )

    if fixed_cell_ids:
        console.print(
            fixed_cell_ids
        )

    console.print(
        "Fix failures:",
        len(
            fix_failures
        ),
    )

    for failure in fix_failures:
        console.print(
            "\n[red]Cell:[/red]",
            failure.get(
                "cell_id"
            ),
        )

        console.print(
            "[red]Type:[/red]",
            failure.get(
                "exception_type"
            ),
        )

        console.print(
            "[red]Message:[/red]",
            failure.get(
                "message"
            ),
        )

    # =========================================
    # 12. GLOBAL ERROR
    # =========================================
    console.rule(
        "[bold cyan]GRAPH ERROR"
    )

    console.print(
        final_result.get(
            "error"
        )
    )
    console.print(
    "Generation Status  :",
    final_result.get(
        "generation_status"
    ),
)
    # =========================================
    # 13. FINAL SUMMARY
    # =========================================
    console.rule(
        "[bold green]TEST SUMMARY"
    )

    console.print(
        f"""
Target              : {final_result.get("target_column")}
Problem Type        : {final_result.get("problem_type")}
Generation Status   : {final_result.get("generation_status")}
Notebook Cells      : {len(notebook_cells)}
Validation Status   : {validation_status}
Validation Errors   : {len(validation_errors)}
Fix Attempts        : {fix_attempts}
Fixed Cells         : {len(fixed_cell_ids)}
Fix Failures        : {len(fix_failures)}

        """
    )


if __name__ == "__main__":
    main()
