from state import State


def route_after_generation(
    state: State,
) -> str:

    status = state.get(
        "generation_cell_status"
    )

    cells = (
        state.get("notebook_cells")
        or []
    )

    print(
        "\n========== ROUTE AFTER GENERATION =========="
    )

    print(
        "generation_cell_status =",
        repr(status),
    )

    print(
        "notebook_cells =",
        len(cells),
    )

    if status == "success":
        print(
            "ROUTE = success -> validate_cells"
        )
        return "success"

    print(
        "ROUTE = failed -> END"
    )

    return "failed"
