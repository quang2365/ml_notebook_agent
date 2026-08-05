from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool


# =========================================================
# CÁC HÀM HỖ TRỢ
# =========================================================

CLASSIFICATION_KEYWORDS = {
    "class", "label", "category", "type", "status", "result",
    "outcome", "default", "fraud", "churn", "approved", "target",
}

REGRESSION_KEYWORDS = {
    "price", "value", "amount", "cost", "revenue", "income",
    "salary", "sales", "temperature", "weight", "height", "duration",
}

ID_KEYWORDS = {
    "id", "identifier", "uuid", "index",
}


def infer_problem_type(
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """
    Gợi ý loại bài toán từ kiểu dữ liệu, tên cột và số mức giá trị.

    Kết quả chỉ là đề xuất và vẫn cần người dùng xác nhận.
    """

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Không tìm thấy cột target: {target_column}"
        )

    target = dataframe[target_column].dropna()

    if target.empty:
        return {
            "problem_type": "unknown",
            "confidence": "low",
            "reasons": ["Target không có dữ liệu hợp lệ."],
        }

    unique_count = int(target.nunique())
    unique_ratio = unique_count / len(target)
    normalized_name = (
        target_column.strip().lower().replace(" ", "_")
    )
    name_tokens = set(normalized_name.split("_"))

    base_result = {
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 6),
    }

    def result(
        problem_type: str,
        confidence: str,
        reason: str,
        classification_type: str | None = None,
    ) -> dict[str, Any]:
        output = {
            **base_result,
            "problem_type": problem_type,
            "confidence": confidence,
            "reasons": [reason],
        }

        if classification_type:
            output["classification_type"] = classification_type

        return output

    # Cột gần như duy nhất và có tên giống ID.
    if name_tokens & ID_KEYWORDS and unique_ratio >= 0.98:
        return result(
            "unknown",
            "high",
            "Cột có đặc điểm giống ID.",
        )

    # Boolean hoặc target chỉ có hai giá trị.
    if pd.api.types.is_bool_dtype(target) or unique_count == 2:
        return result(
            "classification",
            "high",
            "Target chỉ có hai giá trị hoặc có kiểu Boolean.",
            "binary",
        )

    # Chuỗi và category thường là nhãn phân loại.
    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
    ):
        return result(
            "classification",
            "high",
            "Target có kiểu chuỗi hoặc phân loại.",
            "multiclass",
        )

    # Ngày giờ cần xử lý riêng.
    if pd.api.types.is_datetime64_any_dtype(target):
        return result(
            "unknown",
            "low",
            "Target có kiểu ngày giờ và cần được xác nhận thêm.",
        )

    if pd.api.types.is_numeric_dtype(target):
        numeric_target = pd.to_numeric(
            target,
            errors="coerce",
        ).dropna()

        if numeric_target.empty:
            return result(
                "unknown",
                "low",
                "Không thể chuyển target thành dữ liệu số hợp lệ.",
            )

        has_class_keyword = any(
            keyword in normalized_name
            for keyword in CLASSIFICATION_KEYWORDS
        )
        has_regression_keyword = any(
            keyword in normalized_name
            for keyword in REGRESSION_KEYWORDS
        )
        integer_like = bool(
            np.allclose(
                numeric_target.to_numpy(dtype=float),
                np.round(numeric_target.to_numpy(dtype=float)),
            )
        )
        class_threshold = min(
            50,
            max(10, int(np.sqrt(len(dataframe)))),
        )

        if has_class_keyword:
            return result(
                "classification",
                "high",
                "Tên target chứa từ khóa thường dùng cho phân loại.",
                "multiclass",
            )

        if (
            integer_like
            and unique_count <= class_threshold
            and unique_ratio <= 0.05
        ):
            return result(
                "classification",
                "medium",
                "Target là số nguyên nhưng chỉ có ít mức giá trị.",
                "multiclass",
            )

        if has_regression_keyword:
            return result(
                "regression",
                "high",
                "Tên target biểu diễn một đại lượng liên tục.",
            )

        return result(
            "regression",
            "medium",
            "Target là dữ liệu số với nhiều giá trị khác nhau.",
        )

    return result(
        "unknown",
        "low",
        "Chưa đủ thông tin để xác định loại bài toán.",
    )



def json_safe(value: Any) -> Any:
    """
    Chuyển dữ liệu pandas/numpy thành kiểu Python
    có thể serialize sang JSON.
    """

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if value is pd.NA:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

        return round(value, 6)

    return value


def read_csv_with_encoding(
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
                sep=delimiter,
                encoding=encoding,
                low_memory=False,
            )

            return dataframe, encoding

        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(
        "Không thể xác định encoding của file CSV."
    ) from last_error


def build_column_overview(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Thống kê tổng quan cho từng cột.
    """

    rows = len(dataframe)
    result: dict[str, dict[str, Any]] = {}

    for column in dataframe.columns:
        missing_count = int(
            dataframe[column].isna().sum()
        )

        unique_count = int(
            dataframe[column].nunique(
                dropna=True
            )
        )

        result[str(column)] = {
            "dtype": str(dataframe[column].dtype),
            "missing_count": missing_count,
            "missing_percentage": (
                round(missing_count / rows * 100, 4)
                if rows > 0
                else 0.0
            ),
            "unique_count": unique_count,
            "unique_percentage": (
                round(unique_count / rows * 100, 4)
                if rows > 0
                else 0.0
            ),
        }

    return result


def build_numeric_profile(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Thống kê mô tả, độ lệch và ngoại lệ
    cho các cột số.
    """

    result: dict[str, dict[str, Any]] = {}

    for column in numeric_columns:
        original_series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        non_null_original = (
            original_series
            .dropna()
            .to_numpy(dtype=float)
        )

        infinite_count = int(
            np.isinf(non_null_original).sum()
        )

        series = original_series.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid_series = series.dropna()

        if valid_series.empty:
            result[column] = {
                "count": 0,
                "missing_count": int(
                    dataframe[column].isna().sum()
                ),
                "infinite_count": infinite_count,
                "unique_count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "q1": None,
                "median": None,
                "q3": None,
                "max": None,
                "skewness": None,
                "zero_count": 0,
                "negative_count": 0,
                "outlier_count_iqr": 0,
                "outlier_percentage_iqr": 0.0,
            }

            continue

        q1 = valid_series.quantile(0.25)
        q3 = valid_series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (
            (valid_series < lower_bound)
            | (valid_series > upper_bound)
        )

        outlier_count = int(
            outlier_mask.sum()
        )

        result[column] = {
            "count": int(valid_series.count()),
            "missing_count": int(
                dataframe[column].isna().sum()
            ),
            "infinite_count": infinite_count,
            "unique_count": int(
                valid_series.nunique()
            ),
            "mean": valid_series.mean(),
            "std": valid_series.std(),
            "min": valid_series.min(),
            "q1": q1,
            "median": valid_series.median(),
            "q3": q3,
            "max": valid_series.max(),
            "skewness": valid_series.skew(),
            "zero_count": int(
                (valid_series == 0).sum()
            ),
            "negative_count": int(
                (valid_series < 0).sum()
            ),
            "iqr": iqr,
            "lower_outlier_bound": lower_bound,
            "upper_outlier_bound": upper_bound,
            "outlier_count_iqr": outlier_count,
            "outlier_percentage_iqr": round(
                outlier_count
                / len(valid_series)
                * 100,
                4,
            ),
        }

    return json_safe(result)


def build_categorical_profile(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
    max_categories: int,
) -> dict[str, dict[str, Any]]:
    """
    Thống kê phân bố cho các cột phân loại.
    """

    result: dict[str, dict[str, Any]] = {}

    for column in categorical_columns:
        series = dataframe[column]
        non_null_series = (
            series.dropna().astype(str)
        )

        value_counts = (
            non_null_series.value_counts()
        )

        total_non_null = len(
            non_null_series
        )

        top_values: list[dict[str, Any]] = []

        for value, count in (
            value_counts
            .head(max_categories)
            .items()
        ):
            top_values.append(
                {
                    "value": str(value)[:200],
                    "count": int(count),
                    "percentage": (
                        round(
                            count
                            / total_non_null
                            * 100,
                            4,
                        )
                        if total_non_null > 0
                        else 0.0
                    ),
                }
            )

        blank_count = int(
            non_null_series
            .str.strip()
            .eq("")
            .sum()
        )

        whitespace_count = int(
            (
                non_null_series
                != non_null_series.str.strip()
            ).sum()
        )

        rare_threshold = max(
            1,
            int(total_non_null * 0.01),
        )

        rare_category_count = int(
            (value_counts <= rare_threshold).sum()
        )

        mode_value = (
            str(value_counts.index[0])
            if not value_counts.empty
            else None
        )

        mode_count = (
            int(value_counts.iloc[0])
            if not value_counts.empty
            else 0
        )

        result[column] = {
            "count": total_non_null,
            "missing_count": int(
                series.isna().sum()
            ),
            "unique_count": int(
                series.nunique(dropna=True)
            ),
            "mode": mode_value,
            "mode_count": mode_count,
            "mode_percentage": (
                round(
                    mode_count
                    / total_non_null
                    * 100,
                    4,
                )
                if total_non_null > 0
                else 0.0
            ),
            "blank_string_count": blank_count,
            "whitespace_value_count": (
                whitespace_count
            ),
            "rare_category_count": (
                rare_category_count
            ),
            "top_values": top_values,
        }

    return json_safe(result)


def build_top_correlations(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    max_pairs: int,
) -> list[dict[str, Any]]:
    """
    Trả về các cặp cột số có tương quan Pearson
    mạnh nhất theo trị tuyệt đối.
    """

    if len(numeric_columns) < 2:
        return []

    numeric_dataframe = (
        dataframe[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    correlation_matrix = (
        numeric_dataframe.corr(
            method="pearson"
        )
    )

    pairs: list[dict[str, Any]] = []

    for first_index, first_column in enumerate(
        numeric_columns
    ):
        for second_column in numeric_columns[
            first_index + 1:
        ]:
            correlation = correlation_matrix.loc[
                first_column,
                second_column,
            ]

            if pd.isna(correlation):
                continue

            pairs.append(
                {
                    "column_1": first_column,
                    "column_2": second_column,
                    "correlation": float(
                        correlation
                    ),
                    "absolute_correlation": abs(
                        float(correlation)
                    ),
                }
            )

    pairs.sort(
        key=lambda item: item[
            "absolute_correlation"
        ],
        reverse=True,
    )

    return json_safe(
        pairs[:max_pairs]
    )


def detect_target_candidates(
    dataframe: pd.DataFrame,
    possible_id_columns: list[str],
) -> list[dict[str, Any]]:
    """
    Gợi ý target dựa trên tên và đặc điểm cột.

    Đây chỉ là gợi ý, không phải kết luận.
    """

    exact_keywords = {
        "target",
        "label",
        "class",
        "outcome",
        "response",
        "y",
    }

    partial_keywords = {
        "price",
        "value",
        "sales",
        "revenue",
        "default",
        "fraud",
        "churn",
        "risk",
        "score",
        "status",
        "result",
    }

    candidates: list[dict[str, Any]] = []
    rows = len(dataframe)

    for index, column in enumerate(
        dataframe.columns
    ):
        column_name = str(column)
        normalized_name = (
            column_name
            .strip()
            .lower()
            .replace(" ", "_")
        )

        unique_count = int(
            dataframe[column].nunique(
                dropna=True
            )
        )

        if unique_count <= 1:
            continue

        score = 0
        reasons: list[str] = []

        if normalized_name in exact_keywords:
            score += 5
            reasons.append(
                "Tên cột trùng từ khóa target."
            )

        matched_partial = [
            keyword
            for keyword in partial_keywords
            if keyword in normalized_name
        ]

        if matched_partial:
            score += 3
            reasons.append(
                "Tên cột chứa từ khóa: "
                + ", ".join(matched_partial)
            )

        if index == len(dataframe.columns) - 1:
            score += 1
            reasons.append(
                "Đây là cột cuối dataset."
            )

        if column_name in possible_id_columns:
            score -= 5
            reasons.append(
                "Cột có đặc điểm giống ID."
            )

        unique_ratio = (
            unique_count / rows
            if rows > 0
            else 0.0
        )

        problem_suggestion = infer_problem_type(
            dataframe=dataframe,
            target_column=column_name,
        )

        if score > 0:
            candidates.append(
                {
                    "column": column_name,
                    "score": score,
                    "dtype": str(
                        dataframe[column].dtype
                    ),
                    "unique_count": unique_count,
                    "unique_ratio": round(
                        unique_ratio,
                        6,
                    ),
                    "suggested_problem_type": (
                        problem_suggestion["problem_type"]
                    ),
                    "problem_type_confidence": (
                        problem_suggestion.get("confidence")
                    ),
                    "problem_type_reasons": (
                        problem_suggestion.get("reasons", [])
                    ),
                    "reasons": reasons,
                }
            )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates[:5]


def build_target_analysis(
    dataframe: pd.DataFrame,
    target_column: str | None,
    max_categories: int,
) -> dict[str, Any] | None:
    """
    Phân tích target khi người dùng đã xác nhận target.
    """

    if target_column is None:
        return None

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Không tìm thấy target column: "
            f"{target_column}"
        )

    target = dataframe[target_column]
    rows = len(dataframe)

    unique_count = int(
        target.nunique(dropna=True)
    )

    missing_count = int(
        target.isna().sum()
    )

    problem_suggestion = infer_problem_type(
        dataframe=dataframe,
        target_column=target_column,
    )
    problem_type = problem_suggestion["problem_type"]

    # target_analysis cần một nhánh cụ thể để tính thống kê.
    # Nếu chưa đủ chắc chắn, dùng số mức giá trị làm fallback.
    if problem_type == "unknown":
        problem_type = (
            "classification"
            if unique_count <= 20
            else "regression"
        )

    result: dict[str, Any] = {
        "target_column": target_column,
        "dtype": str(target.dtype),
        "problem_type": problem_type,
        "problem_type_confidence": (
            problem_suggestion.get("confidence")
        ),
        "problem_type_reasons": (
            problem_suggestion.get("reasons", [])
        ),
        "missing_count": missing_count,
        "missing_percentage": (
            round(
                missing_count / rows * 100,
                4,
            )
            if rows > 0
            else 0.0
        ),
        "unique_count": unique_count,
    }

    if problem_type == "regression":
        numeric_target = (
            pd.to_numeric(
                target,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        result["distribution"] = {
            "mean": numeric_target.mean(),
            "std": numeric_target.std(),
            "min": numeric_target.min(),
            "q1": numeric_target.quantile(0.25),
            "median": numeric_target.median(),
            "q3": numeric_target.quantile(0.75),
            "max": numeric_target.max(),
            "skewness": numeric_target.skew(),
        }

        numeric_columns = (
            dataframe.select_dtypes(
                include=["number"]
            )
            .columns
            .tolist()
        )

        target_correlations: list[
            dict[str, Any]
        ] = []

        if target_column in numeric_columns:
            correlation_series = (
                dataframe[numeric_columns]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .corr()[target_column]
                .drop(
                    labels=[target_column],
                    errors="ignore",
                )
                .dropna()
            )

            sorted_correlations = sorted(
                correlation_series.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )

            for column, correlation in (
                sorted_correlations[:15]
            ):
                target_correlations.append(
                    {
                        "column": column,
                        "correlation": float(
                            correlation
                        ),
                    }
                )

        result["feature_correlations"] = (
            target_correlations
        )

    else:
        class_counts = (
            target
            .astype("string")
            .fillna("<MISSING>")
            .value_counts()
        )

        class_distribution = []

        for value, count in (
            class_counts
            .head(max_categories)
            .items()
        ):
            class_distribution.append(
                {
                    "class": str(value),
                    "count": int(count),
                    "percentage": (
                        round(
                            count
                            / rows
                            * 100,
                            4,
                        )
                        if rows > 0
                        else 0.0
                    ),
                }
            )

        minimum_class_count = (
            int(class_counts.min())
            if not class_counts.empty
            else 0
        )

        maximum_class_count = (
            int(class_counts.max())
            if not class_counts.empty
            else 0
        )

        imbalance_ratio = (
            round(
                maximum_class_count
                / minimum_class_count,
                4,
            )
            if minimum_class_count > 0
            else None
        )

        result["class_distribution"] = (
            class_distribution
        )

        result["imbalance_ratio"] = (
            imbalance_ratio
        )

    return json_safe(result)


# =========================================================
# TOOL CHÍNH
# =========================================================

@tool
def inspect_dataset(
    file_path: str,
    target_column: str | None = None,
    preview_rows: int = 5,
    max_categories: int = 10,
    max_profile_columns: int = 50,
    max_correlation_pairs: int = 15,
    delimiter: str = ",",
) -> dict[str, Any]:
    """
    Đọc và phân tích tổng quan một file CSV.

    Tool trả về:
    - kích thước dataset;
    - kiểu dữ liệu;
    - missing values;
    - duplicate rows;
    - thống kê cột số;
    - thống kê cột phân loại;
    - outlier theo IQR;
    - skewness;
    - tương quan;
    - ứng viên target;
    - phân tích target nếu target_column được cung cấp.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Đường dẫn không phải file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Hiện tại tool chỉ hỗ trợ file CSV."
        )

    if preview_rows < 0:
        raise ValueError(
            "preview_rows phải lớn hơn "
            "hoặc bằng 0."
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
            read_csv_with_encoding(
                file_path=path,
                delimiter=delimiter,
            )
        )

    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            "File CSV không chứa dữ liệu."
        ) from exc

    except pd.errors.ParserError as exc:
        raise ValueError(
            "Không thể phân tích cấu trúc CSV: "
            f"{exc}"
        ) from exc

    if dataframe.empty:
        raise ValueError(
            "Dataset không có dòng dữ liệu."
        )

    rows, columns = dataframe.shape

    numeric_columns = (
        dataframe
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_columns = (
        dataframe
        .select_dtypes(
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

    boolean_columns = (
        dataframe
        .select_dtypes(include=["bool"])
        .columns
        .tolist()
    )

    datetime_columns = (
        dataframe
        .select_dtypes(
            include=[
                "datetime",
                "datetimetz",
            ]
        )
        .columns
        .tolist()
    )

    missing_values = (
        dataframe
        .isna()
        .sum()
        .astype(int)
        .to_dict()
    )

    missing_percentages = (
        dataframe
        .isna()
        .mean()
        .mul(100)
        .round(4)
        .to_dict()
    )

    total_missing_values = int(
        dataframe.isna().sum().sum()
    )

    total_cells = rows * columns

    total_missing_percentage = (
        round(
            total_missing_values
            / total_cells
            * 100,
            4,
        )
        if total_cells > 0
        else 0.0
    )

    duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    duplicate_percentage = round(
        duplicate_rows / rows * 100,
        4,
    )

    constant_columns = [
        str(column)
        for column in dataframe.columns
        if dataframe[column].nunique(
            dropna=False
        )
        <= 1
    ]

    all_missing_columns = [
        str(column)
        for column in dataframe.columns
        if dataframe[column].isna().all()
    ]

    possible_id_columns = [
        str(column)
        for column in dataframe.columns
        if (
            dataframe[column].nunique(
                dropna=True
            )
            / rows
            >= 0.98
        )
    ]

    low_cardinality_numeric_columns = [
        str(column)
        for column in numeric_columns
        if dataframe[column].nunique(
            dropna=True
        )
        <= 20
    ]

    high_cardinality_columns = [
        str(column)
        for column in categorical_columns
        if dataframe[column].nunique(
            dropna=True
        )
        > max(50, int(rows * 0.2))
    ]

    profiled_columns = (
        dataframe.columns
        .tolist()[:max_profile_columns]
    )

    profiled_numeric_columns = [
        column
        for column in numeric_columns
        if column in profiled_columns
    ]

    profiled_categorical_columns = [
        column
        for column in categorical_columns
        if column in profiled_columns
    ]

    column_overview = build_column_overview(
        dataframe
    )

    numeric_profile = build_numeric_profile(
        dataframe=dataframe,
        numeric_columns=(
            profiled_numeric_columns
        ),
    )

    categorical_profile = (
        build_categorical_profile(
            dataframe=dataframe,
            categorical_columns=(
                profiled_categorical_columns
            ),
            max_categories=max_categories,
        )
    )

    top_correlations = (
        build_top_correlations(
            dataframe=dataframe,
            numeric_columns=(
                profiled_numeric_columns
            ),
            max_pairs=max_correlation_pairs,
        )
    )

    target_candidates = (
        detect_target_candidates(
            dataframe=dataframe,
            possible_id_columns=(
                possible_id_columns
            ),
        )
    )

    target_analysis = build_target_analysis(
        dataframe=dataframe,
        target_column=target_column,
        max_categories=max_categories,
    )

    quality_warnings: list[
        dict[str, Any]
    ] = []

    columns_with_missing = {
        column: count
        for column, count
        in missing_values.items()
        if count > 0
    }

    if columns_with_missing:
        quality_warnings.append(
            {
                "type": "missing_values",
                "severity": "warning",
                "message": (
                    "Dataset có giá trị thiếu."
                ),
                "columns": (
                    columns_with_missing
                ),
            }
        )

    if duplicate_rows > 0:
        quality_warnings.append(
            {
                "type": "duplicate_rows",
                "severity": "warning",
                "message": (
                    f"Dataset có "
                    f"{duplicate_rows} dòng trùng."
                ),
            }
        )

    if constant_columns:
        quality_warnings.append(
            {
                "type": "constant_columns",
                "severity": "warning",
                "message": (
                    "Có cột chỉ chứa một giá trị."
                ),
                "columns": constant_columns,
            }
        )

    if all_missing_columns:
        quality_warnings.append(
            {
                "type": "all_missing_columns",
                "severity": "critical",
                "message": (
                    "Có cột bị thiếu toàn bộ."
                ),
                "columns": all_missing_columns,
            }
        )

    if possible_id_columns:
        quality_warnings.append(
            {
                "type": "possible_id_columns",
                "severity": "info",
                "message": (
                    "Một số cột có tỷ lệ giá trị "
                    "duy nhất rất cao và có thể "
                    "là ID."
                ),
                "columns": possible_id_columns,
            }
        )

    if high_cardinality_columns:
        quality_warnings.append(
            {
                "type": "high_cardinality",
                "severity": "warning",
                "message": (
                    "Có cột phân loại có "
                    "cardinality cao."
                ),
                "columns": (
                    high_cardinality_columns
                ),
            }
        )

    high_skew_columns = [
        column
        for column, profile
        in numeric_profile.items()
        if (
            profile.get("skewness") is not None
            and abs(profile["skewness"]) >= 2
        )
    ]

    if high_skew_columns:
        quality_warnings.append(
            {
                "type": "high_skewness",
                "severity": "info",
                "message": (
                    "Một số cột có phân phối "
                    "lệch mạnh."
                ),
                "columns": high_skew_columns,
            }
        )

    high_outlier_columns = [
        column
        for column, profile
        in numeric_profile.items()
        if (
            profile.get(
                "outlier_percentage_iqr"
            )
            is not None
            and profile[
                "outlier_percentage_iqr"
            ]
            >= 5
        )
    ]

    if high_outlier_columns:
        quality_warnings.append(
            {
                "type": "many_iqr_outliers",
                "severity": "info",
                "message": (
                    "Một số cột có ít nhất 5% "
                    "giá trị nằm ngoài ngưỡng IQR."
                ),
                "columns": high_outlier_columns,
            }
        )

    if not quality_warnings:
        quality_warnings.append(
            {
                "type": "no_obvious_issue",
                "severity": "info",
                "message": (
                    "Chưa phát hiện vấn đề rõ "
                    "ràng ở bước kiểm tra tổng quan."
                ),
            }
        )

    preview = dataframe.head(
        preview_rows
    ).to_dict(orient="records")

    file_size_mb = round(
        path.stat().st_size
        / (1024 ** 2),
        4,
    )

    memory_usage_mb = round(
        dataframe.memory_usage(
            deep=True
        ).sum()
        / (1024 ** 2),
        4,
    )

    analysis_context = {
        "dataset_overview": {
            "file_name": path.name,
            "rows": rows,
            "columns": columns,
            "total_cells": total_cells,
            "file_size_mb": file_size_mb,
            "memory_usage_mb": (
                memory_usage_mb
            ),
        },
        "column_types": {
            "numeric_columns": (
                numeric_columns
            ),
            "categorical_columns": (
                categorical_columns
            ),
            "boolean_columns": (
                boolean_columns
            ),
            "datetime_columns": (
                datetime_columns
            ),
        },
        "data_quality": {
            "total_missing_values": (
                total_missing_values
            ),
            "total_missing_percentage": (
                total_missing_percentage
            ),
            "columns_with_missing": (
                columns_with_missing
            ),
            "duplicate_rows": (
                duplicate_rows
            ),
            "constant_columns": (
                constant_columns
            ),
            "possible_id_columns": (
                possible_id_columns
            ),
            "quality_warnings": (
                quality_warnings
            ),
        },
        "numeric_profile": numeric_profile,
        "categorical_profile": (
            categorical_profile
        ),
        "top_correlations": (
            top_correlations
        ),
        "target_candidates": (
            target_candidates
        ),
        "target_analysis": target_analysis,
    }

    result = {
        # Các key cũ được giữ lại
        "file_name": path.name,
        "rows": rows,
        "columns": columns,
        "column_names": (
            dataframe.columns.tolist()
        ),
        "data_types": {
            str(column): str(dtype)
            for column, dtype
            in dataframe.dtypes.items()
        },
        "numeric_columns": numeric_columns,
        "categorical_columns": (
            categorical_columns
        ),

        # Metadata
        "shape": [rows, columns],
        "total_cells": total_cells,
        "encoding": encoding_used,
        "delimiter": delimiter,
        "file_size_mb": file_size_mb,
        "memory_usage_mb": memory_usage_mb,

        # Missing và duplicate
        "missing_values": missing_values,
        "missing_percentages": (
            missing_percentages
        ),
        "total_missing_values": (
            total_missing_values
        ),
        "total_missing_percentage": (
            total_missing_percentage
        ),
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": (
            duplicate_percentage
        ),

        # Kiểu và chất lượng cột
        "boolean_columns": boolean_columns,
        "datetime_columns": datetime_columns,
        "constant_columns": constant_columns,
        "all_missing_columns": (
            all_missing_columns
        ),
        "possible_id_columns": (
            possible_id_columns
        ),
        "low_cardinality_numeric_columns": (
            low_cardinality_numeric_columns
        ),
        "high_cardinality_columns": (
            high_cardinality_columns
        ),

        # Phân tích chi tiết
        "column_overview": column_overview,
        "numeric_profile": numeric_profile,
        "categorical_profile": (
            categorical_profile
        ),
        "top_correlations": (
            top_correlations
        ),

        # Target
        "target_candidates": (
            target_candidates
        ),
        "target_analysis": target_analysis,

        # Cảnh báo và dữ liệu mẫu
        "quality_warnings": (
            quality_warnings
        ),
        "preview": preview,

        # Thông tin profiling
        "profiled_columns": (
            profiled_columns
        ),
        "profile_truncated": (
            columns > max_profile_columns
        ),

        # Context rút gọn cho LLM
        "analysis_context": (
            analysis_context
        ),
    }

    return json_safe(result)