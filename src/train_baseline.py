from pathlib import Path
import argparse
import math

import pandas as pd

from ml_pipeline_utils import apply_terminal_state_overrides


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x: float) -> float:
    """
    Converts any number into a value between 0 and 1.
    """
    return 1 / (1 + math.exp(-x))


def baseline_home_win_probability(row: pd.Series) -> float:
    """
    Simple rule-based baseline win probability model.

    Logic:
    - Home team leading increases home win probability.
    - A lead matters more when less time is left.
    - Home team gets a small home-court advantage.
    """
    score_margin = row["score_margin_home"]
    seconds_remaining = row["seconds_remaining"]

    total_game_seconds = 48 * 60
    time_elapsed_ratio = 1 - (seconds_remaining / total_game_seconds)
    time_elapsed_ratio = max(0, min(1, time_elapsed_ratio))

    score_weight = 0.08 + 0.35 * time_elapsed_ratio
    home_advantage = 0.15

    raw_score = score_weight * score_margin + home_advantage

    return sigmoid(raw_score)


def add_baseline_predictions(game_state: pd.DataFrame) -> pd.DataFrame:
    """
    Adds baseline home and away win probability columns.
    """
    output = game_state.copy()

    output["home_win_prob"] = output.apply(baseline_home_win_probability, axis=1)
    output = apply_terminal_state_overrides(output)
    output["prediction_source"] = "baseline_rule_model"

    return output


def load_game_state(game_id: str | None = None) -> pd.DataFrame:
    """
    Loads a processed game-state file.

    If game_id is provided, loads that exact game.
    Otherwise, loads the latest available file.
    """
    if game_id:
        game_id = str(game_id).zfill(10)
        input_path = PROCESSED_DIR / f"game_state_{game_id}.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"No game-state file found for game {game_id}. Run:\n"
                f"python src/run_pipeline.py --game-id {game_id} --model baseline"
            )

        print(f"Loading game-state file: {input_path}")
        return pd.read_csv(input_path, dtype={"game_id": str})

    files = list(PROCESSED_DIR.glob("game_state_*.csv"))

    if not files:
        raise FileNotFoundError("No processed game-state files found in data/processed.")

    input_path = files[-1]
    print(f"Loading game-state file: {input_path}")

    return pd.read_csv(input_path, dtype={"game_id": str})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline rule-based win probability predictions."
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

    game_state = load_game_state(args.game_id)
    predictions = add_baseline_predictions(game_state)

    game_id = str(predictions["game_id"].iloc[0]).zfill(10)
    output_path = PROCESSED_DIR / f"baseline_predictions_{game_id}.csv"

    predictions.to_csv(output_path, index=False)

    print("\nSuccess.")
    print(f"Saved baseline predictions to: {output_path}")
    print(f"Rows: {len(predictions)}")

    print("\nSample columns:")
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
                "wp_change",
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
