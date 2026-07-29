from pathlib import Path
from typing import Any

import pandas as pd
from langchain.tools import tool

@tool
def inspect_dataset(file_path: str) -> dict[str, Any]:
    """
    Đọc một file CSV và trả về thông tin tổng quan của dataset.

    Args:
        file_path: Đường dẫn đến file CSV cần phân tích.

    Returns:
        Dictionary chứa kích thước, tên cột, kiểu dữ liệu
        và danh sách các cột số/phân loại.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dataset: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Đường dẫn không phải là một file: {file_path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Phiên bản hiện tại chỉ hỗ trợ file CSV."
        )

    try:
        dataframe = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "File CSV không chứa dữ liệu."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            "Không thể phân tích cấu trúc file CSV."
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Không thể đọc encoding của file CSV."
        ) from exc

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = dataframe.select_dtypes(
        exclude="number"
    ).columns.tolist()

    summary: dict[str, Any] = {
        "file_name": path.name,
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": dataframe.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }

    return summary