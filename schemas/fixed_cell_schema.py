from pydantic import BaseModel, Field


class FixedCell(BaseModel):
    cell_id: str = Field(description=(
            "ID of the cell being fixed. "
            "Do not change the ID."
        )
    )

    source: str = Field(description=(
            "All Python source code "
            "after it has been fixed."
        )
    )

    changes: str = Field(description=(
            "Brief description of the fixed error."
        )
    )
