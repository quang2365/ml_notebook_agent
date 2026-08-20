from typing import Literal

from pydantic import BaseModel, Field


class NotebookCell(BaseModel):
    cell_id: str = Field(
        description="Unique cell identifier."
    )

    section_id: str = Field(
        description="Section containing the cell."
    )
    cell_type: Literal["markdown", "code"] = Field(
        description="Cell type: markdown or code."
    )
    title: str = Field(
        description="Short description of the cell."
    )

    source: str = Field(
        description="Complete cell content."
    )

    purpose: str = Field(
        description="Purpose of the cell."
    )

    expected_output: str | None = Field(
        default=None,
        description=(
            "Description of the expected result when running the code cell. "
            "Markdown cellss may be null."
        ),
    )


class GeneratedSection(BaseModel):
    section_id: str = Field(
        description=(
            "ID of the section currently being generated."
        )
    )

    cells: list[NotebookCell] = Field(
        description=(
            "List of notebook cells "
            "belonging to this section."
        )
    )
