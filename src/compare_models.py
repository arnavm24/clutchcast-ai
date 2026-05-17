from pathlib import Path
import argparse
import json

import joblib
import pandas as pd
import torch

from ml_pipeline_utils import (
    TARGET_COLUMN,
    apply_terminal_state_overrides,
    compute_probability_metrics,
    load_shared_training_inputs,
    rank_leaderboard,
)
from train_baseline import baseline_home_win_probability
from train_neural_network import WinProbabilityNeuralNetwork


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_FILES = {
    "baseline": "baseline_predictions_{game_id}.csv",
    "logistic_regression": "ml_predictions_{game_id}.csv",
    "random_forest": "advanced_predictions_{game_id}.csv",
    "pytorch_neural_network": "neural_predictions_{game_id}.csv",
}

MODEL_LABELS = {
    "baseline": "Baseline Rule Model",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "pytorch_neural_network": "PyTorch Neural Network",
}

MODEL_PROBABILITY_COLUMNS = {
    "baseline": "baseline_home_win_prob_pct",
    "logistic_regression": "logistic_regression_home_win_prob_pct",
    "random_forest": "random_forest_home_win_prob_pct",
    "pytorch_neural_network": "pytorch_neural_network_home_win_prob_pct",
}


def get_available_game_ids() -> list[str]:
    game_id_sets = []

    for filename in MODEL_FILES.values():
        prefix = filename.split("{game_id}")[0]
        suffix = filename.split("{game_id}")[1]
        game_ids = {
            file.name.replace(prefix, "").replace(suffix, "")
            for file in PROCESSED_DIR.glob(filename.format(game_id="*"))
        }
        game_id_sets.append(game_ids)

    if not game_id_sets:
        return []

    return sorted(set.intersection(*game_id_sets))


def load_prediction_file(game_id: str, model_key: str) -> pd.DataFrame:
    filename = MODEL_FILES[model_key].format(game_id=game_id)
    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    return pd.read_csv(path, dtype={"game_id": str})


def load_predictions(game_id: str) -> dict[str, pd.DataFrame]:
    return {
        model_key: load_prediction_file(game_id, model_key)
        for model_key in MODEL_FILES
    }


def validate_prediction_lengths(predictions: dict[str, pd.DataFrame]) -> None:
    lengths = {
        model_key: len(df)
        for model_key, df in predictions.items()
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(f"Prediction files have different lengths: {lengths}")


def compare_predictions(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    validate_prediction_lengths(predictions)

    baseline = predictions["baseline"]
    comparison = pd.DataFrame()

    context_columns = [
        "game_id",
        "period",
        "clock",
        "home_score",
        "away_score",
        "score_margin_home",
        "event_team",
        "event_player",
        "event_description",
    ]

    for column in context_columns:
        comparison[column] = baseline[column]

    probability_columns = []

    for model_key, df in predictions.items():
        probability_column = MODEL_PROBABILITY_COLUMNS[model_key]
        comparison[probability_column] = df["home_win_prob_pct"]
        probability_columns.append(probability_column)

    for model_key, probability_column in MODEL_PROBABILITY_COLUMNS.items():
        if model_key == "baseline":
            continue

        diff_column = f"{model_key}_minus_baseline_pct"
        abs_diff_column = f"abs_{model_key}_minus_baseline_pct"

        comparison[diff_column] = (
            comparison[probability_column]
            - comparison[MODEL_PROBABILITY_COLUMNS["baseline"]]
        ).round(2)
        comparison[abs_diff_column] = comparison[diff_column].abs().round(2)

    comparison["max_model_disagreement_pct"] = (
        comparison[probability_columns].max(axis=1)
        - comparison[probability_columns].min(axis=1)
    ).round(2)

    return comparison


def build_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    final_row = comparison.iloc[-1]

    summary = {
        "game_id": str(final_row["game_id"]).zfill(10),
        "rows_compared": len(comparison),
        "max_model_disagreement_pct": round(
            comparison["max_model_disagreement_pct"].max(), 2
        ),
        "final_home_score": int(final_row["home_score"]),
        "final_away_score": int(final_row["away_score"]),
        "final_home_margin": int(final_row["score_margin_home"]),
    }

    for model_key, probability_column in MODEL_PROBABILITY_COLUMNS.items():
        summary[f"{model_key}_final_home_win_prob_pct"] = final_row[probability_column]

        if model_key != "baseline":
            diff_column = f"abs_{model_key}_minus_baseline_pct"
            summary[f"avg_{model_key}_vs_baseline_diff_pct"] = round(
                comparison[diff_column].mean(), 2
            )

    return pd.DataFrame([summary])


def get_biggest_disagreements(
    comparison: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    columns = [
        "period",
        "clock",
        "home_score",
        "away_score",
        "score_margin_home",
        "event_team",
        "event_player",
        "event_description",
        "baseline_home_win_prob_pct",
        "logistic_regression_home_win_prob_pct",
        "random_forest_home_win_prob_pct",
        "pytorch_neural_network_home_win_prob_pct",
        "max_model_disagreement_pct",
    ]

    return (
        comparison.sort_values("max_model_disagreement_pct", ascending=False)
        [columns]
        .head(top_n)
        .reset_index(drop=True)
    )


def evaluate_baseline_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict:
    prediction_frame = test_data.copy()
    prediction_frame["home_win_prob"] = prediction_frame.apply(
        baseline_home_win_probability,
        axis=1,
    )
    prediction_frame = apply_terminal_state_overrides(prediction_frame)

    return compute_probability_metrics(
        y_true=prediction_frame[TARGET_COLUMN],
        probabilities=prediction_frame["home_win_prob"],
        model_key="baseline",
        model_name=MODEL_LABELS["baseline"],
        feature_count=0,
        train_data=train_data,
        test_data=test_data,
    )


def evaluate_sklearn_model(
    model_key: str,
    model_name: str,
    model_path: Path,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_name} model artifact: {model_path}. "
            "Train models before generating the leaderboard."
        )

    model = joblib.load(model_path)
    probabilities = model.predict_proba(test_data[feature_columns])[:, 1]

    prediction_frame = test_data.copy()
    prediction_frame["home_win_prob"] = probabilities
    prediction_frame = apply_terminal_state_overrides(prediction_frame)

    return compute_probability_metrics(
        y_true=prediction_frame[TARGET_COLUMN],
        probabilities=prediction_frame["home_win_prob"],
        model_key=model_key,
        model_name=model_name,
        feature_count=len(feature_columns),
        train_data=train_data,
        test_data=test_data,
    )


def evaluate_pytorch_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    model_path = MODELS_DIR / "pytorch_win_probability_model.pt"
    scaler_path = MODELS_DIR / "pytorch_scaler.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing PyTorch model artifact: {model_path}. "
            "Train models before generating the leaderboard."
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing PyTorch scaler artifact: {scaler_path}. "
            "Train models before generating the leaderboard."
        )

    checkpoint = torch.load(model_path, map_location="cpu")
    input_size = checkpoint.get("input_size", len(feature_columns))

    model = WinProbabilityNeuralNetwork(input_size=input_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path)
    X_test = test_data[feature_columns].astype(float)
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    with torch.no_grad():
        probabilities = model(X_test_tensor).numpy().flatten()

    prediction_frame = test_data.copy()
    prediction_frame["home_win_prob"] = probabilities
    prediction_frame = apply_terminal_state_overrides(prediction_frame)

    return compute_probability_metrics(
        y_true=prediction_frame[TARGET_COLUMN],
        probabilities=prediction_frame["home_win_prob"],
        model_key="pytorch_neural_network",
        model_name=MODEL_LABELS["pytorch_neural_network"],
        feature_count=len(feature_columns),
        train_data=train_data,
        test_data=test_data,
    )


def to_jsonable(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def build_leaderboard() -> pd.DataFrame:
    train_data, test_data, feature_columns, _train_game_ids, _test_game_ids = (
        load_shared_training_inputs()
    )

    metrics = [
        evaluate_baseline_model(train_data, test_data),
        evaluate_sklearn_model(
            model_key="logistic_regression",
            model_name=MODEL_LABELS["logistic_regression"],
            model_path=MODELS_DIR / "win_probability_model.joblib",
            train_data=train_data,
            test_data=test_data,
            feature_columns=feature_columns,
        ),
        evaluate_sklearn_model(
            model_key="random_forest",
            model_name=MODEL_LABELS["random_forest"],
            model_path=MODELS_DIR / "advanced_win_probability_model.joblib",
            train_data=train_data,
            test_data=test_data,
            feature_columns=feature_columns,
        ),
        evaluate_pytorch_model(train_data, test_data, feature_columns),
    ]

    leaderboard = rank_leaderboard(pd.DataFrame(metrics))

    leaderboard_path = REPORTS_DIR / "model_leaderboard.csv"
    champion_path = REPORTS_DIR / "champion_model.json"

    leaderboard.to_csv(leaderboard_path, index=False)

    champion = {
        key: to_jsonable(value)
        for key, value in leaderboard.iloc[0].to_dict().items()
    }

    champion["selection_rule"] = (
        "Lowest Brier score, then lowest log loss, then highest ROC-AUC, "
        "then highest accuracy."
    )

    champion_path.write_text(
        json.dumps(champion, indent=2),
        encoding="utf-8",
    )

    print("\nSuccess.")
    print(f"Saved model leaderboard to: {leaderboard_path}")
    print(f"Saved champion model report to: {champion_path}")
    print("\nModel leaderboard:")
    print(leaderboard.to_string(index=False))

    print("\nChampion model:")
    print(f"{champion['model_name']} ({champion['model_key']})")

    return leaderboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline, logistic regression, random forest, "
            "and PyTorch neural network predictions."
        )
    )

    parser.add_argument(
        "--game-id",
        type=str,
        default=None,
        help=(
            "Specific game ID to compare. "
            "If omitted, uses the latest available shared game."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of biggest disagreement moments to show.",
    )

    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Evaluate all models on the shared test-game split and select a champion.",
    )

    return parser.parse_args()


def run_game_comparison(game_id: str | None, top_n: int) -> None:
    available_game_ids = get_available_game_ids()

    if not available_game_ids:
        raise FileNotFoundError(
            "No games found with all four prediction files. Run:\n"
            "python src/run_pipeline.py --game-id YOUR_GAME_ID --model baseline\n"
            "python src/run_pipeline.py --game-id YOUR_GAME_ID --model ml\n"
            "python src/run_pipeline.py --game-id YOUR_GAME_ID --model advanced\n"
            "python src/run_pipeline.py --game-id YOUR_GAME_ID --model neural"
        )

    if game_id:
        selected_game_id = str(game_id).zfill(10)
    else:
        selected_game_id = available_game_ids[-1]

    if selected_game_id not in available_game_ids:
        raise ValueError(
            f"Game {selected_game_id} does not have all four prediction files.\n"
            f"Available game IDs: {available_game_ids}"
        )

    print(f"Comparing models for game: {selected_game_id}")

    predictions = load_predictions(selected_game_id)
    comparison = compare_predictions(predictions)

    summary = build_summary(comparison)
    disagreements = get_biggest_disagreements(comparison, top_n=top_n)

    comparison_path = REPORTS_DIR / f"model_comparison_{selected_game_id}.csv"
    summary_path = REPORTS_DIR / f"model_comparison_summary_{selected_game_id}.csv"
    disagreements_path = REPORTS_DIR / f"model_disagreements_{selected_game_id}.csv"

    comparison.to_csv(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)
    disagreements.to_csv(disagreements_path, index=False)

    print("\nSuccess.")
    print(f"Saved full comparison to: {comparison_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved biggest disagreements to: {disagreements_path}")

    print("\nModel comparison summary:")
    print(summary.to_string(index=False))

    print("\nBiggest disagreement moments:")
    print(disagreements.to_string(index=False))


def main() -> None:
    args = parse_args()

    if args.leaderboard:
        build_leaderboard()
        return

    run_game_comparison(args.game_id, args.top_n)


if __name__ == "__main__":
    main()
