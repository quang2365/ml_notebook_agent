from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt
from state import State

allow_problem_types = ["regression", "classification"]

def review_problem_node(state: State) -> dict:
    proposal = state.get("problem_proposal")
    summary = state.get("summary")
    decisions = interrupt(
        {
            "title": "Confirm the Machine Learning problem",
            "proposal": proposal,
            "instructions": {
                "approve": {
                    "action": "approve",
                },
                "edit": {
                    "action": "edit",
                    "target_column": "New target name",
                    "problem_type": (
                        "regression or classification"
                    ),
                },
                "reject": {
                    "action": "reject",
                    "feedback": "Reason for rejection",
                },
            },
        }
    )
    decision = decisions.get("action")
    if decision == "approve":
        target_column = proposal.get("target_column")
        problem_type = proposal.get("problem_type")
        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "approval_status": "approved",
            "messages": [HumanMessage(content=f"I confirm the problem as {problem_type}, with target column {target_column}"),AIMessage(content=f"The user confirmed the problem: target is `{target_column}`, problem type is `{problem_type}`")],
        }
    if decision == "edit":
        target_column = decisions.get("target_column")
        problem_type = decisions.get("problem_type")

        dataset_columns = summary.get("column_names", [])

        if target_column not in dataset_columns:
            error_message = f"Target column `{target_column}` does not exist in the dataset."
            return{
                "approval_status": "rejected",
                "error": error_message,
                "messages": [AIMessage(content=error_message)]
            }
        if problem_type not in allow_problem_types:
            error_message = f"Problem type `{problem_type}` is invalid. Please choose one of the valid problem types: {', '.join(allow_problem_types)}."
            return{
                "approval_status": "rejected",
                "error": error_message,
                "messages": [AIMessage(content=error_message)]
            }
        return {
            "target_column": target_column,
            "problem_type": problem_type,
            "approval_status": "approved",
            "messages": [HumanMessage(content=f"I confirm the problem as {problem_type}, with target column {target_column}"),AIMessage(content=f"The user confirmed the problem: target is `{target_column}`, problem type is `{problem_type}`")],
        }
    if decision == "reject":
        return {
            "approval_status": "rejected",
            "messages": [HumanMessage(content="I reject the problem proposal"),AIMessage(content="The user rejected the problem proposal")]
        }
