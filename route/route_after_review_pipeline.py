# route/route_after_review_pipline.py

from state import State


MAX_PIPELINE_FIX_ATTEMPTS = 3


def route_after_review_pipeline(state: State,) -> str:
    status = state.get("pipeline_review_status")

    pipeline_fix_attempts = state.get(
        "pipeline_fix_attempts",
        0,
    )

    if status == "valid":
        return "valid"

    if (
        status == "invalid"
        and pipeline_fix_attempts < MAX_PIPELINE_FIX_ATTEMPTS
    ):
        return "invalid"

    return "failed"
