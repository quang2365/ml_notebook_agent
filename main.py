import json
import os  #AI
from uuid import uuid4  #AI

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from rich.console import Console
from rich.markdown import Markdown

console = Console()


def ask_use_deepseek() -> bool:
    """Hỏi người chạy có muốn dùng DeepSeek V4 Flash hay không."""
    answer = input(
        "Sử dụng DeepSeek V4 Flash? [y/N]: "
    ).strip().lower()

    return answer in {
        "y",
        "yes",
        "1",
        "true",
    }


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
    # Phải chọn model trước khi import graph vì các node
    # khởi tạo structured LLM ngay tại thời điểm import.  #AI
    use_deepseek = ask_use_deepseek()  #AI
    os.environ["USE_DEEPSEEK"] = (  #AI
        "true" if use_deepseek else "false"
    )

    from graph import build_graph  #AI
    from model.model import selected_model_name  #AI

    console.print(
        "Model đang sử dụng:",
        selected_model_name(use_deepseek),
    )

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
        "fix_plan_attempts": 0,  #AI
        "notebook_cells": None,
        "section_generation_status": None,
        "section_generation_errors": [],  #AI
        "current_section_index": 0,  #AI
        "generated_section_ids": [],  #AI
        "section_retry_attempts": 0,  #AI

        "notebook_path": "output/test.ipynb",  #AI
        "build_status": "pending",  #AI
        "build_error": None,  #AI

        "execution_status": "pending",  #AI
        "execution_error": None,  #AI

        "validation_cell_status": None,
        "validation_cell_errors": None,

        "pipeline_review_status": "pending",  #AI
        "pipeline_review_errors": [],  #AI

        "fix_cell_attempts": 0,
        "fixed_cell_ids": None,
        "fix_cell_failures": None,

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
    validation_cell_status = (
        final_result.get(
            "validation_cell_status"
        )
    )

    validation_cell_errors = (
        final_result.get(
            "validation_cell_errors"
        )
        or []
    )

    console.rule(
        "[bold cyan]VALIDATION"
    )

    console.print(
        "Status:",
        validation_cell_status,
    )

    console.print(
        "Errors:",
            len(
                validation_cell_errors
            ),
    )

    for error in validation_cell_errors:
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

    fix_cell_attempts = final_result.get(
        "fix_cell_attempts",
        0,
    )

    fixed_cell_ids = (
        final_result.get(
            "fixed_cell_ids"
        )
        or []
    )

    fix_cell_failures = (
        final_result.get(
            "fix_cell_failures"
        )
        or []
    )

    console.print(
        "Fix attempts:",
        fix_cell_attempts,
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
                fix_cell_failures
            ),
    )

    for failure in fix_cell_failures:
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
        "section_generation_status"
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
Generation Status   : {final_result.get("section_generation_status")}
Notebook Cells      : {len(notebook_cells)}
Validation Status   : {validation_cell_status}
Validation Errors   : {len(validation_cell_errors)}
Fix Attempts        : {fix_cell_attempts}
Fixed Cells         : {len(fixed_cell_ids)}
Fix Failures        : {len(fix_cell_failures)}
Pipeline Review     : {final_result.get("pipeline_review_status")}
Pipeline Errors     : {len(final_result.get("pipeline_review_errors") or [])}
Execution Status    : {final_result.get("execution_status")}
Execution Error     : {final_result.get("execution_error")}

        """
    )


if __name__ == "__main__":
    main()
