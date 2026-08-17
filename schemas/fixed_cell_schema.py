from pydantic import BaseModel, Field


class FixedCell(BaseModel):
    cell_id: str = Field(description=(
            "ID của cell đang được sửa. "
            "Không được thay đổi ID."
        )
    )

    source: str = Field(description=(
            "Toàn bộ Python source code "
            "sau khi đã sửa."
        )
    )

    changes: str = Field(description=(
            "Mô tả ngắn lỗi đã được sửa."
        )
    )