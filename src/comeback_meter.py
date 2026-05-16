from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_file() -> pd.DataFrame:
    """
    Loads the feature file created by src/features.py.

    This file should already contain:
    - win probability
    - score state
    - clutch pressure
    """
    files = list(PROCESSED_DIR.glob("features_*.csv"))

    if not files:
        raise FileNotFoundError(
            "No feature files found. Run src/features.py first."
        )

    input_path = files[0]
    print(f"Loading feature file from: {input_path}")

    return pd.read_csv(input_path, dtype={"game_id": str})


def classify_comeback_status(comeback_probability: float) -> str:
    """
    Converts comeback probability into a readable label.
    """
    if comeback_probability >= 0.40:
        return "Very realistic"
    if comeback_probability >= 0.25:
        return "Possible"
    if comeback_probability >= 0.10:
        return "Difficult"
    if comeback_probability >= 0.03:
        return "Very unlikely"
    return "Nearly impossible"


def calculate_required_scoring_rate(deficit: int, seconds_remaining: int) -> float:
    """
    Estimates how many points per minute the trailing team needs
    just to erase the deficit before the game ends.

    This is not a full basketball model. It is a simple useful context metric.
    """
    if deficit <= 0:
        return 0.0

    minutes_remaining = max(seconds_remaining / 60, 0.01)
    return round(deficit / minutes_remaining, 2)


def add_comeback_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds comeback-related columns to every game event.

    From the home team's perspective:
    - If home is losing, home comeback probability = home_win_prob.
    - If away is losing, away comeback probability = away_win_prob.
    """
    required_columns = [
        "game_id",
        "period",
        "clock",
        "seconds_remaining",
        "home_score",
        "away_score",
        "score_margin_home",
        "home_win_prob",
        "away_win_prob",
        "home_win_prob_pct",
        "away_win_prob_pct",
        "clutch_pressure",
        "pressure_level",
        "event_description",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output = df.copy()

    output["trailing_team"] = "Tie"
    output.loc[output["score_margin_home"] < 0, "trailing_team"] = "Home"
    output.loc[output["score_margin_home"] > 0, "trailing_team"] = "Away"

    output["deficit"] = output["score_margin_home"].abs()

    output["comeback_probability"] = 0.0

    # If home team is trailing, home_win_prob is their comeback chance.
    home_trailing = output["score_margin_home"] < 0
    output.loc[home_trailing, "comeback_probability"] = output.loc[
        home_trailing, "home_win_prob"
    ]

    # If away team is trailing, away_win_prob is their comeback chance.
    away_trailing = output["score_margin_home"] > 0
    output.loc[away_trailing, "comeback_probability"] = output.loc[
        away_trailing, "away_win_prob"
    ]

    # If tied, no comeback is needed.
    tied = output["score_margin_home"] == 0
    output.loc[tied, "comeback_probability"] = 0.5

    output["comeback_probability_pct"] = (
        output["comeback_probability"] * 100
    ).round(1)

    output["comeback_status"] = output["comeback_probability"].apply(
        classify_comeback_status
    )

    output["required_points_per_minute"] = output.apply(
        lambda row: calculate_required_scoring_rate(
            int(row["deficit"]),
            int(row["seconds_remaining"]),
        ),
        axis=1,
    )

    return output


def get_most_interesting_comeback_windows(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Finds moments where a team is trailing but still has a non-trivial comeback chance.
    """
    comeback_rows = df[
        (df["trailing_team"] != "Tie")
        & (df["seconds_remaining"] > 0)
        & (df["deficit"] >= 5)
    ].copy()

    if comeback_rows.empty:
        return pd.DataFrame()

    # Prioritize situations that are both difficult and still alive.
    comeback_rows["interest_score"] = (
        comeback_rows["deficit"] * 0.45
        + comeback_rows["clutch_pressure"] * 0.35
        + comeback_rows["comeback_probability_pct"] * 0.20
    )

    columns = [
        "period",
        "clock",
        "home_score",
        "away_score",
        "trailing_team",
        "deficit",
        "comeback_probability_pct",
        "comeback_status",
        "required_points_per_minute",
        "clutch_pressure",
        "pressure_level",
        "event_description",
    ]

    return (
        comeback_rows.sort_values("interest_score", ascending=False)
        [columns]
        .head(top_n)
        .reset_index(drop=True)
    )


def main() -> None:
    features = load_feature_file()
    game_id = str(features["game_id"].iloc[0]).zfill(10)

    output = add_comeback_metrics(features)

    output_path = PROCESSED_DIR / f"comeback_metrics_{game_id}.csv"
    output.to_csv(output_path, index=False)

    report = get_most_interesting_comeback_windows(output, top_n=10)
    report_path = REPORTS_DIR / f"comeback_report_{game_id}.csv"
    report.to_csv(report_path, index=False)

    print("\nSuccess.")
    print(f"Saved comeback metrics to: {output_path}")
    print(f"Saved comeback report to: {report_path}")

    print("\nMost interesting comeback windows:")
    print(report)


if __name__ == "__main__":
    main()