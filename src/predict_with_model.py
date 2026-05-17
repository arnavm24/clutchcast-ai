from pathlib import Path
import argparse

import joblib
import pandas as pd


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


FEATURE_COLUMNS = [
    "period",
    "seconds_remaining",
    "home_score",
    "away_score",
    "score_margin_home",
    "abs_score_margin",
    "total_score",
    "is_4th_quarter",
    "is_clutch_time",
]


def load_model():
    model_path = MODELS_DIR / "win_probability_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run:\n"
            "python src/train_model.py"
        )

    print(f"Loading model from: {model_path}")
    return joblib.load(model_path)


def load_game_state(game_id: str | None = None) -> pd.DataFrame:
    if game_id:
        game_id = str(game_id).zfill(10)
        input_path = PROCESSED_DIR / f"game_state_{game_id}.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"No game-state file found for game {game_id}. Run:\n"
                f"python src/run_pipeline.py --game-id {game_id} --model ml"
            )

        print(f"Loading game-state file: {input_path}")
        return pd.read_csv(input_path, dtype={"game_id": str})

    files = list(PROCESSED_DIR.glob("game_state_*.csv"))

    if not files:
        raise FileNotFoundError(
            "No game-state file found. Run:\n"
            "python src/game_state.py"
        )

    input_path = files[-1]
    print(f"Loading game-state file: {input_path}")

    return pd.read_csv(input_path, dtype={"game_id": str})


def validate_features(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")


def add_ml_predictions(game_state: pd.DataFrame, model) -> pd.DataFrame:
    output = game_state.copy()

    validate_features(output)

    X = output[FEATURE_COLUMNS]

    home_win_prob = model.predict_proba(X)[:, 1]

    output["home_win_prob"] = home_win_prob
    output["away_win_prob"] = 1 - output["home_win_prob"]

    output["home_win_prob_pct"] = (output["home_win_prob"] * 100).round(1)
    output["away_win_prob_pct"] = (output["away_win_prob"] * 100).round(1)

    output["wp_change"] = output["home_win_prob"].diff().fillna(0)
    output["abs_wp_change"] = output["wp_change"].abs()

    output["prediction_source"] = "logistic_regression_model"

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate logistic regression ML win probability predictions."
    )

    parser.add_argument(
        "--game-id",
        type=str,
        default=None,
        help="Specific NBA game ID to predict, example: 0042300312.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = load_model()
    game_state = load_game_state(args.game_id)

    predictions = add_ml_predictions(game_state, model)

    game_id = str(predictions["game_id"].iloc[0]).zfill(10)
    output_path = PROCESSED_DIR / f"ml_predictions_{game_id}.csv"

    predictions.to_csv(output_path, index=False)

    print("\nSuccess.")
    print(f"Saved ML predictions to: {output_path}")
    print(f"Rows: {len(predictions)}")

    print("\nSample predictions:")
    print(
        predictions[
            [
                "period",
                "clock",
                "home_score",
                "away_score",
                "score_margin_home",
                "home_win_prob_pct",
                "away_win_prob_pct",
                "prediction_source",
            ]
        ].head(10)
    )

    print("\nFinal row:")
    print(
        predictions[
            [
                "period",
                "clock",
                "home_score",
                "away_score",
                "score_margin_home",
                "home_win_prob_pct",
                "away_win_prob_pct",
                "home_won",
            ]
        ].tail(1)
    )


if __name__ == "__main__":
    main()