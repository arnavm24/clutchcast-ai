from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


BASE_FEATURE_COLUMNS = [
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


TARGET_COLUMN = "home_won"


def load_training_dataset() -> pd.DataFrame:
    input_path = PROCESSED_DIR / "training_dataset.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            "Missing training dataset. Run:\n"
            'python src/build_training_dataset.py --season 2023-24 --season-type "Regular Season" --max-games 300'
        )

    print(f"Loading training dataset from: {input_path}")

    data = pd.read_csv(input_path, dtype={"game_id": str})

    if data.empty:
        raise ValueError("Training dataset is empty.")

    return data


def contains_keyword(series: pd.Series, keywords: list[str]) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()

    result = pd.Series(False, index=series.index)

    for keyword in keywords:
        result = result | text.str.contains(keyword, regex=False)

    return result.astype(int)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    total_game_seconds = 48 * 60

    output["time_remaining_fraction"] = (
        output["seconds_remaining"] / total_game_seconds
    ).clip(0, 1)

    output["time_elapsed_fraction"] = (
        1 - output["time_remaining_fraction"]
    ).clip(0, 1)

    output["is_second_half"] = (output["period"] >= 3).astype(int)
    output["is_final_5_minutes"] = (output["seconds_remaining"] <= 5 * 60).astype(int)
    output["is_final_2_minutes"] = (output["seconds_remaining"] <= 2 * 60).astype(int)
    output["is_final_1_minute"] = (output["seconds_remaining"] <= 60).astype(int)

    return output


def add_score_margin_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    output["home_lead"] = (output["score_margin_home"] > 0).astype(int)
    output["away_lead"] = (output["score_margin_home"] < 0).astype(int)
    output["game_tied"] = (output["score_margin_home"] == 0).astype(int)

    output["is_one_possession_game"] = (output["abs_score_margin"] <= 3).astype(int)
    output["is_two_possession_game"] = (output["abs_score_margin"] <= 6).astype(int)
    output["is_blowout_margin"] = (output["abs_score_margin"] >= 20).astype(int)

    output["margin_squared"] = output["score_margin_home"] ** 2

    output["score_margin_time_weighted"] = (
        output["score_margin_home"] * output["time_elapsed_fraction"]
    )

    output["abs_margin_time_weighted"] = (
        output["abs_score_margin"] * output["time_elapsed_fraction"]
    )

    return output


def add_event_type_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    description = output["event_description"]

    output["is_shot"] = contains_keyword(
        description,
        [
            "jump shot",
            "layup",
            "dunk",
            "hook shot",
            "tip shot",
            "floating",
            "driving",
        ],
    )

    output["is_three_pointer"] = contains_keyword(description, ["3pt"])
    output["is_free_throw"] = contains_keyword(description, ["free throw"])
    output["is_missed_shot"] = contains_keyword(description, ["miss"])
    output["is_made_shot"] = (
        ((output["is_shot"] == 1) | (output["is_free_throw"] == 1))
        & (output["is_missed_shot"] == 0)
    ).astype(int)

    output["is_turnover"] = contains_keyword(description, ["turnover"])
    output["is_rebound"] = contains_keyword(description, ["rebound"])
    output["is_offensive_rebound"] = contains_keyword(description, ["rebound (off:"])
    output["is_steal"] = contains_keyword(description, ["steal"])
    output["is_block"] = contains_keyword(description, ["block"])
    output["is_foul"] = contains_keyword(description, ["foul", "p.foul", "s.foul"])
    output["is_timeout"] = contains_keyword(description, ["timeout"])
    output["is_substitution"] = contains_keyword(description, ["sub:"])

    return output


def classify_event_value(description: str) -> int:
    desc = str(description).lower()

    if "timeout" in desc or "sub:" in desc:
        return 0

    if "turnover" in desc:
        return -4

    if "steal" in desc:
        return 4

    if "block" in desc:
        return 3

    if "3pt" in desc and "miss" not in desc:
        return 5

    if "dunk" in desc and "miss" not in desc:
        return 4

    if "layup" in desc and "miss" not in desc:
        return 3

    if "jump shot" in desc and "miss" not in desc:
        return 3

    if "free throw" in desc and "miss" not in desc:
        return 1

    if "miss" in desc:
        return -2

    if "rebound" in desc and "off:" in desc:
        return 2

    if "rebound" in desc:
        return 1

    return 0


def add_event_value_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    output["event_value"] = output["event_description"].apply(classify_event_value)

    return output


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    output = output.sort_values(["game_id", "event_num"]).copy()

    for window in [5, 10]:
        output[f"recent_margin_change_{window}"] = (
            output.groupby("game_id")["score_margin_home"]
            .diff(window)
            .fillna(0)
        )

        output[f"recent_event_value_{window}"] = (
            output.groupby("game_id")["event_value"]
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

        output[f"recent_total_score_change_{window}"] = (
            output.groupby("game_id")["total_score"]
            .diff(window)
            .fillna(0)
        )

    return output


def add_team_event_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    output["event_team"] = output["event_team"].fillna("").astype(str)

    home_team_by_game = (
        output[output["event_team"] != ""]
        .groupby("game_id")["event_team"]
        .agg(lambda teams: teams.mode().iloc[0] if not teams.mode().empty else "")
    )

    output["estimated_home_event_team"] = output["game_id"].map(home_team_by_game).fillna("")

    output["event_by_estimated_home"] = (
        output["event_team"] == output["estimated_home_event_team"]
    ).astype(int)

    output["event_by_estimated_away"] = (
        (output["event_team"] != "")
        & (output["event_team"] != output["estimated_home_event_team"])
    ).astype(int)

    output["signed_event_value_home_perspective"] = output["event_value"]

    output.loc[
        output["event_by_estimated_away"] == 1,
        "signed_event_value_home_perspective",
    ] = -output.loc[
        output["event_by_estimated_away"] == 1,
        "event_value",
    ]

    return output


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    required_columns = BASE_FEATURE_COLUMNS + [
        TARGET_COLUMN,
        "game_id",
        "event_num",
        "event_description",
        "event_team",
    ]

    missing = [col for col in required_columns if col not in output.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output = add_time_features(output)
    output = add_score_margin_features(output)
    output = add_event_type_features(output)
    output = add_event_value_features(output)
    output = add_rolling_features(output)
    output = add_team_event_direction_features(output)

    output = output.fillna(0)

    return output


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "game_id",
        "event_num",
        "clock",
        "event_team",
        "event_player",
        "event_description",
        "event_type",
        "estimated_home_event_team",
        TARGET_COLUMN,
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    return feature_columns


def save_feature_list(feature_columns: list[str]) -> None:
    output_path = PROCESSED_DIR / "model_feature_columns.txt"

    output_path.write_text("\n".join(feature_columns), encoding="utf-8")

    print(f"Saved model feature list to: {output_path}")


def main() -> None:
    data = load_training_dataset()

    model_data = build_model_features(data)

    feature_columns = get_model_feature_columns(model_data)

    output_path = PROCESSED_DIR / "model_training_dataset.csv"
    model_data.to_csv(output_path, index=False)

    save_feature_list(feature_columns)

    print("\nSuccess.")
    print(f"Saved improved model dataset to: {output_path}")
    print(f"Rows: {len(model_data)}")
    print(f"Total columns: {len(model_data.columns)}")
    print(f"Model feature columns: {len(feature_columns)}")

    print("\nFeature columns:")
    for column in feature_columns:
        print(f"- {column}")


if __name__ == "__main__":
    main()