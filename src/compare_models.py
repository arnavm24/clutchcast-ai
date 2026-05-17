from pathlib import Path
import argparse

import pandas as pd


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_available_game_ids() -> list[str]:
    baseline_ids = {
        file.stem.replace("baseline_predictions_", "")
        for file in PROCESSED_DIR.glob("baseline_predictions_*.csv")
    }

    ml_ids = {
        file.stem.replace("ml_predictions_", "")
        for file in PROCESSED_DIR.glob("ml_predictions_*.csv")
    }

    return sorted(baseline_ids.intersection(ml_ids))


def load_predictions(game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = PROCESSED_DIR / f"baseline_predictions_{game_id}.csv"
    ml_path = PROCESSED_DIR / f"ml_predictions_{game_id}.csv"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline predictions: {baseline_path}")

    if not ml_path.exists():
        raise FileNotFoundError(f"Missing ML predictions: {ml_path}")

    baseline = pd.read_csv(baseline_path, dtype={"game_id": str})
    ml = pd.read_csv(ml_path, dtype={"game_id": str})

    return baseline, ml


def compare_predictions(baseline: pd.DataFrame, ml: pd.DataFrame) -> pd.DataFrame:
    if len(baseline) != len(ml):
        raise ValueError(
            f"Prediction files have different lengths: "
            f"baseline={len(baseline)}, ml={len(ml)}"
        )

    comparison = pd.DataFrame()

    comparison["game_id"] = baseline["game_id"]
    comparison["period"] = baseline["period"]
    comparison["clock"] = baseline["clock"]
    comparison["home_score"] = baseline["home_score"]
    comparison["away_score"] = baseline["away_score"]
    comparison["score_margin_home"] = baseline["score_margin_home"]
    comparison["event_team"] = baseline["event_team"]
    comparison["event_player"] = baseline["event_player"]
    comparison["event_description"] = baseline["event_description"]

    comparison["baseline_home_win_prob_pct"] = baseline["home_win_prob_pct"]
    comparison["ml_home_win_prob_pct"] = ml["home_win_prob_pct"]

    comparison["probability_difference_pct"] = (
        comparison["ml_home_win_prob_pct"]
        - comparison["baseline_home_win_prob_pct"]
    ).round(2)

    comparison["absolute_difference_pct"] = (
        comparison["probability_difference_pct"].abs()
    ).round(2)

    comparison["baseline_wp_change_pct"] = (baseline["wp_change"] * 100).round(2)
    comparison["ml_wp_change_pct"] = (ml["wp_change"] * 100).round(2)

    comparison["wp_change_difference_pct"] = (
        comparison["ml_wp_change_pct"] - comparison["baseline_wp_change_pct"]
    ).round(2)

    return comparison


def build_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    final_row = comparison.iloc[-1]

    summary = {
        "game_id": str(final_row["game_id"]).zfill(10),
        "rows_compared": len(comparison),
        "average_absolute_difference_pct": comparison[
            "absolute_difference_pct"
        ].mean().round(2),
        "maximum_absolute_difference_pct": comparison[
            "absolute_difference_pct"
        ].max().round(2),
        "average_signed_difference_pct": comparison[
            "probability_difference_pct"
        ].mean().round(2),
        "baseline_final_home_win_prob_pct": final_row[
            "baseline_home_win_prob_pct"
        ],
        "ml_final_home_win_prob_pct": final_row["ml_home_win_prob_pct"],
        "final_probability_difference_pct": final_row[
            "probability_difference_pct"
        ],
        "final_home_score": int(final_row["home_score"]),
        "final_away_score": int(final_row["away_score"]),
        "final_home_margin": int(final_row["score_margin_home"]),
    }

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
        "ml_home_win_prob_pct",
        "probability_difference_pct",
        "absolute_difference_pct",
    ]

    return (
        comparison.sort_values("absolute_difference_pct", ascending=False)
        [columns]
        .head(top_n)
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and ML win probability predictions."
    )

    parser.add_argument(
        "--game-id",
        type=str,
        default=None,
        help="Specific game ID to compare. If omitted, uses the latest available shared game.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of biggest disagreement moments to show.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    available_game_ids = get_available_game_ids()

    if not available_game_ids:
        raise FileNotFoundError(
            "No games found with both baseline and ML predictions. Run:\n"
            "python src/run_pipeline.py --game-id YOUR_GAME_ID\n"
            "python src/run_pipeline.py --game-id YOUR_GAME_ID --use-ml"
        )

    if args.game_id:
        game_id = str(args.game_id).zfill(10)
    else:
        game_id = available_game_ids[-1]

    if game_id not in available_game_ids:
        raise ValueError(
            f"Game {game_id} does not have both baseline and ML predictions.\n"
            f"Available game IDs: {available_game_ids}"
        )

    print(f"Comparing models for game: {game_id}")

    baseline, ml = load_predictions(game_id)
    comparison = compare_predictions(baseline, ml)

    summary = build_summary(comparison)
    disagreements = get_biggest_disagreements(comparison, top_n=args.top_n)

    comparison_path = REPORTS_DIR / f"model_comparison_{game_id}.csv"
    summary_path = REPORTS_DIR / f"model_comparison_summary_{game_id}.csv"
    disagreements_path = REPORTS_DIR / f"model_disagreements_{game_id}.csv"

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


if __name__ == "__main__":
    main()