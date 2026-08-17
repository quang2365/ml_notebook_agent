from state import State


def route_after_stage(state: State) -> str:
    """Dừng stage tuyến tính nếu node vừa trả về lỗi."""

    return "failed" if state.get("error") else "success"
