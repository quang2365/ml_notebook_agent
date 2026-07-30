from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool


def _json_safe(value: Any) -> Any:
    """
    Chuyển dữ liệu pandas/numpy thành kiểu dữ liệu Python
    có thể serialize sang JSON.
    """

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if value is pd.NA:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def _read_csv_with_encoding(
    file_path: Path,
    delimiter: str,
) -> tuple[pd.DataFrame, str]:
    """
    Thử đọc CSV bằng một số encoding thông dụng.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
    ]

    last_error: Exception | None = None

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=delimiter,
                low_memory=False,
            )

            return dataframe, encoding

        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(
        "Không thể xác định encoding của file CSV."
    ) from last_error


def _build_numeric_summary(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Tạo thống kê mô tả cho các cột số.
    """

    result: dict[str, dict[str, Any]] = {}

    for column in numeric_columns:
        series = dataframe[column]

        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        )

        result[column] = {
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "mean": numeric_series.mean(),
            "std": numeric_series.std(),
            "min": numeric_series.min(),
            "q1": numeric_series.quantile(0.25),
            "median": numeric_series.median(),
            "q3": numeric_series.quantile(0.75),
            "max": numeric_series.max(),
            "zero_count": int((numeric_series == 0).sum()),
            "negative_count": int((numeric_series < 0).sum()),
            "infinite_count": int(
                np.isinf(numeric_series).sum()
            ),
        }

    return _json_safe(result)


def _build_categorical_summary(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
    max_categories: int,
) -> dict[str, dict[str, Any]]:
    """
    Tạo thống kê cho các cột phân loại.
    """

    result: dict[str, dict[str, Any]] = {}

    total_rows = len(dataframe)

    for column in categorical_columns:
        series = dataframe[column]

        unique_count = int(
            series.nunique(dropna=True)
        )

        top_values = []

        for value, count in (
            series.value_counts(dropna=True)
            .head(max_categories)
            .items()
        ):
            top_values.append(
                {
                    "value": _json_safe(value),
                    "count": int(count),
                    "percentage": round(
                        count / total_rows * 100,
                        2,
                    )
                    if total_rows > 0
                    else 0.0,
                }
            )

        result[column] = {
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "unique": unique_count,
            "unique_percentage": round(
                unique_count / total_rows * 100,
                2,
            )
            if total_rows > 0
            else 0.0,
            "top_values": top_values,
        }

    return _json_safe(result)


def _find_target_hints(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Chỉ đưa ra gợi ý target dựa trên tên cột.

    Đây không phải kết luận cuối cùng.
    """

    target_keywords = [
        "target",
        "label",
        "class",
        "outcome",
        "response",
        "price",
        "value",
        "default",
        "fraud",
        "churn",
        "sales",
    ]

    candidates = []

    for column in dataframe.columns:
        normalized_name = (
            str(column)
            .lower()
            .strip()
            .replace(" ", "_")
        )

        matched_keywords = [
            keyword
            for keyword in target_keywords
            if keyword in normalized_name
        ]

        if matched_keywords:
            candidates.append(
                {
                    "column": str(column),
                    "matched_keywords": matched_keywords,
                    "dtype": str(dataframe[column].dtype),
                    "unique_values": int(
                        dataframe[column].nunique(
                            dropna=True
                        )
                    ),
                }
            )

    return {
        "name_based_candidates": candidates,
        "last_column": (
            str(dataframe.columns[-1])
            if len(dataframe.columns) > 0
            else None
        ),
        "note": (
            "Đây chỉ là gợi ý dựa trên tên cột. "
            "Cần xác nhận mục tiêu thực tế với người dùng."
        ),
    }


@tool
def inspect_dataset(
    file_path: str,
    preview_rows: int = 5,
    max_categories: int = 10,
    max_profile_columns: int = 50,
    delimiter: str = ",",
) -> dict[str, Any]:
    """
    Đọc và kiểm tra tổng quan một file CSV.

    Tool trả về kích thước dữ liệu, kiểu cột, missing values,
    duplicate rows, thống kê cột số, thống kê cột phân loại,
    dữ liệu xem trước và các cảnh báo chất lượng dữ liệu.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Đường dẫn không phải một file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Hiện tại tool chỉ hỗ trợ file CSV."
        )

    if preview_rows < 0:
        raise ValueError(
            "preview_rows phải lớn hơn hoặc bằng 0."
        )

    if max_categories <= 0:
        raise ValueError(
            "max_categories phải lớn hơn 0."
        )

    if max_profile_columns <= 0:
        raise ValueError(
            "max_profile_columns phải lớn hơn 0."
        )

    try:
        dataframe, encoding_used = (
            _read_csv_with_encoding(
                file_path=path,
                delimiter=delimiter,
            )
        )

    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "File CSV không có dữ liệu."
        ) from exc

    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Không thể phân tích cấu trúc CSV: {exc}"
        ) from exc

    if dataframe.empty:
        raise ValueError(
            "Dataset không có dòng dữ liệu nào."
        )

    rows, columns = dataframe.shape

    numeric_columns = (
        dataframe.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    boolean_columns = (
        dataframe.select_dtypes(
            include=["bool"]
        )
        .columns
        .tolist()
    )

    datetime_columns = (
        dataframe.select_dtypes(
            include=[
                "datetime",
                "datetimetz",
            ]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        dataframe.select_dtypes(
            include=[
                "object",
                "string",
                "category",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    missing_values = (
        dataframe.isna()
        .sum()
        .astype(int)
        .to_dict()
    )

    missing_percentages = (
        dataframe.isna()
        .mean()
        .mul(100)
        .round(2)
        .to_dict()
    )

    unique_values = (
        dataframe.nunique(dropna=True)
        .astype(int)
        .to_dict()
    )

    duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    duplicate_percentage = round(
        duplicate_rows / rows * 100,
        2,
    )

    all_missing_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].isna().all()
    ]

    constant_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].nunique(
            dropna=False
        )
        <= 1
    ]

    possible_id_columns = [
        column
        for column in dataframe.columns
        if (
            dataframe[column].nunique(
                dropna=True
            )
            / rows
            >= 0.98
        )
    ]

    high_cardinality_columns = [
        column
        for column in categorical_columns
        if (
            dataframe[column].nunique(
                dropna=True
            )
            > 100
        )
    ]

    profile_columns = (
        dataframe.columns
        .tolist()[:max_profile_columns]
    )

    profiled_numeric_columns = [
        column
        for column in numeric_columns
        if column in profile_columns
    ]

    profiled_categorical_columns = [
        column
        for column in categorical_columns
        if column in profile_columns
    ]

    numeric_summary = _build_numeric_summary(
        dataframe=dataframe,
        numeric_columns=profiled_numeric_columns,
    )

    categorical_summary = (
        _build_categorical_summary(
            dataframe=dataframe,
            categorical_columns=(
                profiled_categorical_columns
            ),
            max_categories=max_categories,
        )
    )

    warnings: list[str] = []

    missing_columns = [
        column
        for column, count
        in missing_values.items()
        if count > 0
    ]

    if missing_columns:
        warnings.append(
            "Dataset có giá trị thiếu ở các cột: "
            + ", ".join(missing_columns)
        )

    if duplicate_rows > 0:
        warnings.append(
            f"Dataset có {duplicate_rows} dòng trùng lặp."
        )

    if constant_columns:
        warnings.append(
            "Các cột chỉ có một giá trị: "
            + ", ".join(constant_columns)
        )

    if all_missing_columns:
        warnings.append(
            "Các cột bị thiếu toàn bộ dữ liệu: "
            + ", ".join(all_missing_columns)
        )

    if high_cardinality_columns:
        warnings.append(
            "Các cột phân loại có cardinality cao: "
            + ", ".join(high_cardinality_columns)
        )

    if not warnings:
        warnings.append(
            "Chưa phát hiện vấn đề chất lượng dữ liệu "
            "rõ ràng từ bước kiểm tra tổng quan."
        )

    file_size_mb = round(
        path.stat().st_size / (1024**2),
        3,
    )

    memory_usage_mb = round(
        dataframe.memory_usage(
            deep=True
        ).sum()
        / (1024**2),
        3,
    )

    target_hints = _find_target_hints(
        dataframe
    )

    preview = _json_safe(
        dataframe.head(preview_rows).to_dict(
            orient="records"
        )
    )

    # Context ngắn gọn dành cho LLM.
    # Không cần truyền toàn bộ summary nếu dataset có quá nhiều cột.
    analysis_context = {
        "file_name": path.name,
        "rows": rows,
        "columns": columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": {
            column: count
            for column, count
            in missing_values.items()
            if count > 0
        },
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "target_hints": target_hints,
        "quality_warnings": warnings,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }

    result = {
        # Các key cũ được giữ lại để code hiện tại vẫn chạy.
        "file_name": path.name,
        "rows": rows,
        "columns": columns,
        "column_names": dataframe.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype
            in dataframe.dtypes.items()
        },
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,

        # Thông tin mới.
        "boolean_columns": boolean_columns,
        "datetime_columns": datetime_columns,
        "encoding": encoding_used,
        "delimiter": delimiter,
        "file_size_mb": file_size_mb,
        "memory_usage_mb": memory_usage_mb,
        "missing_values": missing_values,
        "missing_percentages": missing_percentages,
        "total_missing_values": int(
            dataframe.isna().sum().sum()
        ),
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": (
            duplicate_percentage
        ),
        "unique_values": unique_values,
        "constant_columns": constant_columns,
        "all_missing_columns": all_missing_columns,
        "possible_id_columns": possible_id_columns,
        "high_cardinality_columns": (
            high_cardinality_columns
        ),
        "numeric_summary": numeric_summary,
        "categorical_summary": (
            categorical_summary
        ),
        "target_hints": target_hints,
        "quality_warnings": warnings,
        "preview": preview,
        "profiled_columns": profile_columns,
        "profile_truncated": (
            columns > max_profile_columns
        ),
        "analysis_context": analysis_context,
    }

    return _json_safe(result)