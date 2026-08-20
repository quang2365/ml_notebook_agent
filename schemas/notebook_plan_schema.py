from pydantic import BaseModel, Field


class NotebookSection(BaseModel):
    section_id: str = Field(
        description="Unique identifier of the notebook section."
    )

    title: str = Field(
        description="Notebook section title."
    )

    objective: str = Field(
        description="Objective of this section."
    )

    cell_types: list[str] = Field(
        description=(
            "List of cell types to create, "
            "for example markdown or code."
        )
    )

    tasks: list[str] = Field(
        description="Tasks to complete."
    )


class NotebookPlan(BaseModel):
    notebook_title: str = Field(description="Notebook name.")

    target_column: str = Field(description="Confirmed target column.")

    problem_type: str = Field(description=(
            "Problem type regression "
            "or classification."
        )
    )

    objective: str = Field(description="Overall notebook objective.")

    evaluation_metrics: list[str] = Field(description="Metrics to use.")

    candidate_models: list[str] = Field(description="Candidate models to test.")

    sections: list[NotebookSection] = Field(
    min_length=8,
    max_length=10,
    description=(
        "List of 8 to 10 notebook sections "
        "in the correct order for "
        "performing Machine Learning."
    ),
)
