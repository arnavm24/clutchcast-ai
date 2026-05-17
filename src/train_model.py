from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_pipeline_utils import (
    TARGET_COLUMN,
    apply_terminal_state_overrides,
    compute_probability_metrics,
    load_shared_training_inputs,
)


MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def train_model(train_data: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    X_train = train_data[feature_columns]
    y_train = train_data[TARGET_COLUMN]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Pipeline,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    X_test = test_data[feature_columns]
    probabilities = model.predict_proba(X_test)[:, 1]

    prediction_frame = test_data.copy()
    prediction_frame["home_win_prob"] = probabilities
    prediction_frame = apply_terminal_state_overrides(prediction_frame)

    return compute_probability_metrics(
        y_true=prediction_frame[TARGET_COLUMN],
        probabilities=prediction_frame["home_win_prob"],
        model_key="logistic_regression",
        model_name="Logistic Regression",
        feature_count=len(feature_columns),
        train_data=train_data,
        test_data=test_data,
    )


def save_feature_importance(model: Pipeline, feature_columns: list[str]) -> None:
    classifier = model.named_steps["classifier"]

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": classifier.coef_[0],
            "absolute_importance": abs(classifier.coef_[0]),
        }
    ).sort_values("absolute_importance", ascending=False)

    output_path = REPORTS_DIR / "model_feature_importance.csv"
    importance.to_csv(output_path, index=False)

    print(f"Saved feature importance to: {output_path}")
    print("\nTop feature importance:")
    print(importance.head(15).to_string(index=False))


def save_metrics(metrics: dict) -> None:
    output_path = REPORTS_DIR / "model_metrics.csv"

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path, index=False)

    print(f"Saved model metrics to: {output_path}")
    print("\nModel metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def save_model(model: Pipeline, feature_columns: list[str]) -> None:
    output_path = MODELS_DIR / "win_probability_model.joblib"
    metadata_path = MODELS_DIR / "win_probability_model_features.txt"

    joblib.dump(model, output_path)
    metadata_path.write_text("\n".join(feature_columns), encoding="utf-8")

    print(f"Saved trained model to: {output_path}")
    print(f"Saved model feature list to: {metadata_path}")


def main() -> None:
    train_data, test_data, feature_columns, train_game_ids, test_game_ids = (
        load_shared_training_inputs()
    )

    print("\nDataset summary:")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Train rows: {len(train_data)}")
    print(f"Train games: {len(train_game_ids)}")
    print(f"Test rows: {len(test_data)}")
    print(f"Test games: {len(test_game_ids)}")

    model = train_model(train_data, feature_columns)
    metrics = evaluate_model(model, train_data, test_data, feature_columns)

    save_model(model, feature_columns)
    save_metrics(metrics)
    save_feature_importance(model, feature_columns)

    print("\nSuccess.")
    print("Retrained logistic regression model using the shared game-level split.")


if __name__ == "__main__":
    main()
