def validate_notebook_plan(
    plan: dict,
    expected_target: str | None = None,
    expected_problem_type: str | None = None,
) -> list[dict]:

    errors: list[dict] = []

    sections = plan.get("sections") or []

    if not 8 <= len(sections) <= 10:
        errors.append(
            {
                "error_type": "invalid_section_count",
                "location": "sections",
                "message": (
                    "Notebook plan must have 8 to 10 sections. "
                    f"Current count: {len(sections)}."
                ),
            }
        )

    if (
        expected_target
        and plan.get("target_column") != expected_target
    ):
        errors.append(
            {
                "error_type": "target_changed",
                "location": "target_column",
                "message": (
                    f"Target must be `{expected_target}`, "
                    f"but received `{plan.get('target_column')}`."
                ),
            }
        )

    if (
        expected_problem_type
        and plan.get("problem_type")
        != expected_problem_type
    ):
        errors.append(
            {
                "error_type": "problem_type_changed",
                "location": "problem_type",
                "message": (
                    f"Problem type must be `{expected_problem_type}`."
                ),
            }
        )

    seen_ids: set[str] = set()

    for index, section in enumerate(
        sections,
        start=1,
    ):
        section_id = section.get("section_id")
        expected_id = f"section_{index}"
        tasks = section.get("tasks") or []

        if not section_id:
            errors.append(
                {
                    "error_type": "missing_section_id",
                    "location": f"sections[{index - 1}]",
                    "message": "Section has no section_id.",
                }
            )

        elif section_id in seen_ids:
            errors.append(
                {
                    "error_type": "duplicate_section_id",
                    "location": section_id,
                    "message": f"`{section_id}` is duplicated.",
                }
            )

        if section_id != expected_id:
            errors.append(
                {
                    "error_type": "invalid_section_order",
                    "location": section_id or expected_id,
                    "message": (
                        f"Expected `{expected_id}`, "
                        f"received `{section_id}`."
                    ),
                }
            )

        if not 1 <= len(tasks) <= 5:
            errors.append(
                {
                    "error_type": "invalid_task_count",
                    "location": section_id or expected_id,
                    "message": (
                        "Section must have 1 to 5 tasks. "
                        f"Current count: {len(tasks)}."
                    ),
                }
            )

        if section_id:
            seen_ids.add(section_id)

    return errors
