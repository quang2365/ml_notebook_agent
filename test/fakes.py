"""Fake LangChain runnables and reusable test fixtures."""

from __future__ import annotations

from collections import deque
from typing import Any

from schemas.notebook_cell_schema import GeneratedSection
from schemas.notebook_plan_schema import NotebookPlan


class FakeRunnable:
    """Mimic a LangChain Runnable without making network requests."""

    def __init__(self, responses: list[Any]) -> None:
        if not responses:
            raise ValueError("FakeRunnable requires at least one response.")

        self._responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        input: Any,
        config: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            {
                "input": input,
                "config": config,
                "kwargs": kwargs,
            }
        )

        response = self._responses.popleft()
        if isinstance(response, BaseException):
            raise response

        return response


def make_plan(section_count: int = 8) -> NotebookPlan:
    sections = []

    for index in range(1, section_count + 1):
        sections.append(
            {
                "section_id": f"section_{index}",
                "title": f"Section {index}",
                "objective": f"Objective {index}",
                "cell_types": ["markdown", "code"],
                "tasks": [f"Task {index}"],
            }
        )

    return NotebookPlan(
        notebook_title="Offline Housing Test",
        target_column="median_house_value",
        problem_type="regression",
        objective="Build a regression notebook.",
        evaluation_metrics=["MAE", "RMSE"],
        candidate_models=["LinearRegression", "RandomForestRegressor"],
        sections=sections,
    )


def make_generated_section(
    section_id: str,
    source: str,
) -> GeneratedSection:
    return GeneratedSection(
        section_id=section_id,
        cells=[
            {
                "cell_id": f"{section_id}_code_1",
                "section_id": section_id,
                "cell_type": "code",
                "title": f"Code for {section_id}",
                "source": source,
                "purpose": "Offline generated test cell.",
                "expected_output": None,
            }
        ],
    )


def make_agent_cell(
    source: str = "x = 1\nprint(x)",
    cell_id: str = "section_1_code_1",
) -> dict:
    return {
        "cell_id": cell_id,
        "section_id": "section_1",
        "cell_type": "code",
        "title": "Offline test cell",
        "source": source,
        "purpose": "Test notebook cell processing.",
        "expected_output": None,
    }


def make_full_rendered_sections() -> list[GeneratedSection]:
    """Return a realistic 10-section, 30-cell fake LLM render."""

    section_sources = [
        (
            "Setup and configuration",
            "import pandas as pd\n"
            "import numpy as np\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
            "RANDOM_STATE = 42\n"
            "dataset_path = './data/housing.csv'\n"
            "target_column = 'median_house_value'",
        ),
        (
            "Load the housing dataset",
            "df = pd.read_csv(dataset_path)\n"
            "df_head = df.head()",
            "dataset_shape = df.shape\n"
            "column_names = df.columns.tolist()",
        ),
        (
            "Explore data quality and distributions",
            "missing_values = df.isna().sum()\n"
            "duplicate_count = int(df.duplicated().sum())",
            "numeric_summary = df.describe(include='number')\n"
            "categorical_summary = df.describe(include='object')",
        ),
        (
            "Create features, target and holdout split",
            "X = df.drop(columns=[target_column])\n"
            "y = df[target_column]",
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=RANDOM_STATE\n"
            ")",
        ),
        (
            "Build leakage-safe preprocessing",
            "numeric_features = X_train.select_dtypes(include='number').columns.tolist()\n"
            "categorical_features = X_train.select_dtypes(exclude='number').columns.tolist()",
            "numeric_pipeline = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('scaler', StandardScaler()),\n"
            "])\n"
            "categorical_pipeline = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='most_frequent')),\n"
            "    ('encoder', OneHotEncoder(handle_unknown='ignore')),\n"
            "])\n"
            "preprocessor = ColumnTransformer([\n"
            "    ('numeric', numeric_pipeline, numeric_features),\n"
            "    ('categorical', categorical_pipeline, categorical_features),\n"
            "])",
        ),
        (
            "Train a linear-regression baseline",
            "baseline_model = Pipeline([\n"
            "    ('preprocessor', preprocessor),\n"
            "    ('model', LinearRegression()),\n"
            "])",
            "baseline_model.fit(X_train, y_train)\n"
            "trained_models = {'linear_regression': baseline_model}",
        ),
        (
            "Train a random-forest candidate",
            "random_forest_model = Pipeline([\n"
            "    ('preprocessor', preprocessor),\n"
            "    ('model', RandomForestRegressor(\n"
            "        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1\n"
            "    )),\n"
            "])",
            "random_forest_model.fit(X_train, y_train)\n"
            "trained_models['random_forest'] = random_forest_model",
        ),
        (
            "Train a gradient-boosting candidate",
            "gradient_boosting_model = Pipeline([\n"
            "    ('preprocessor', preprocessor),\n"
            "    ('model', GradientBoostingRegressor(random_state=RANDOM_STATE)),\n"
            "])",
            "gradient_boosting_model.fit(X_train, y_train)\n"
            "trained_models['gradient_boosting'] = gradient_boosting_model",
        ),
        (
            "Evaluate and compare all models",
            "predictions = {\n"
            "    name: model.predict(X_test)\n"
            "    for name, model in trained_models.items()\n"
            "}",
            "model_results = []\n"
            "for model_name, y_pred in predictions.items():\n"
            "    model_results.append({\n"
            "        'model': model_name,\n"
            "        'mae': mean_absolute_error(y_test, y_pred),\n"
            "        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),\n"
            "        'r2': r2_score(y_test, y_pred),\n"
            "    })",
        ),
        (
            "Summarize the experiment",
            "results_df = pd.DataFrame(model_results).sort_values('rmse')\n"
            "best_model_name = results_df.iloc[0]['model']",
            "best_model = trained_models[best_model_name]\n"
            "final_summary = {\n"
            "    'best_model': best_model_name,\n"
            "    'test_rows': len(X_test),\n"
            "    'target': target_column,\n"
            "}",
        ),
    ]

    rendered_sections: list[GeneratedSection] = []

    for index, (description, first_code, second_code) in enumerate(
        section_sources,
        start=1,
    ):
        section_id = f"section_{index}"
        rendered_sections.append(
            GeneratedSection(
                section_id=section_id,
                cells=[
                    {
                        "cell_id": f"{section_id}_markdown_1",
                        "section_id": section_id,
                        "cell_type": "markdown",
                        "title": description,
                        "source": f"## {index}. {description}",
                        "purpose": "Explain the section before executing code.",
                        "expected_output": None,
                    },
                    {
                        "cell_id": f"{section_id}_code_1",
                        "section_id": section_id,
                        "cell_type": "code",
                        "title": f"{description} - part 1",
                        "source": first_code,
                        "purpose": "Execute the first task in this section.",
                        "expected_output": None,
                    },
                    {
                        "cell_id": f"{section_id}_code_2",
                        "section_id": section_id,
                        "cell_type": "code",
                        "title": f"{description} - part 2",
                        "source": second_code,
                        "purpose": "Execute the second task in this section.",
                        "expected_output": None,
                    },
                ],
            )
        )

    return rendered_sections
