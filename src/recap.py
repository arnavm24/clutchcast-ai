from pathlib import Path
import argparse
import json

import pandas as pd
from nba_api.stats.endpoints import boxscoresummaryv2

from champion_inference import get_prediction_file_prefix, load_champion_metadata


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def format_nba_clock(clock_value) -> str:
    clock = str(clock_value)

    if not clock.startswith("PT"):
        return clock

    clock = clock.replace("PT", "")
    minutes = 0
    seconds = 0.0

    if "M" in clock:
        minutes_part, clock = clock.split("M")
        minutes = int(minutes_part)

    if "S" in clock:
        seconds = float(clock.replace("S", ""))

    if seconds.is_integer():
        return f"{minutes}:{int(seconds):02d}"

    return f"{minutes}:{seconds:04.1f}"


def format_period_clock(period: int, clock_value) -> str:
    label = f"Q{period}" if period <= 4 else f"OT{period - 4}"
    return f"{label}, {format_nba_clock(clock_value)}"


def pluralize_point(value: int) -> str:
    return "point" if abs(value) == 1 else "points"


def get_team_labels(game_id: str) -> tuple[str, str]:
    try:
        summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id, timeout=30)
        try:
            line_score = summary.line_score.get_data_frame()
        except AttributeError:
            line_score = summary.get_data_frames()[5]

        if len(line_score) >= 2:
            away_team = str(line_score.iloc[0]["TEAM_ABBREVIATION"])
            home_team = str(line_score.iloc[1]["TEAM_ABBREVIATION"])
            return home_team, away_team
    except Exception:
        pass

    return "Home", "Away"


def find_file(folder: Path, pattern: str, game_id: str | None = None) -> Path:
    files = list(folder.glob(pattern.format(game_id=game_id or "*")))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {folder / pattern}")
    return sorted(files)[-1]


def load_csv(folder: Path, pattern: str, game_id: str | None = None) -> pd.DataFrame:
    path = find_file(folder, pattern, game_id)
    print(f"Loading: {path}")
    return pd.read_csv(path, dtype={"game_id": str})


def filter_ranked_report(df: pd.DataFrame, require_player_team: bool = False) -> pd.DataFrame:
    if df.empty:
        return df

    keep = pd.Series(True, index=df.index)

    if "seconds_remaining" in df.columns:
        keep = keep & (df["seconds_remaining"] > 0)

    if "event_description" in df.columns:
        description = df["event_description"].fillna("").astype(str).str.lower()
        keep = keep & ~description.str.contains("end of", regex=False)
        keep = keep & ~description.str.contains("instant replay", regex=False)

    if require_player_team:
        for column in ["event_team", "event_player"]:
            if column in df.columns:
                values = df[column].fillna("").astype(str).str.strip()
                keep = keep & (values != "")

    return df[keep].reset_index(drop=True)


def prediction_label_from_key(model_key: str, fallback: str) -> str:
    labels = {
        "baseline": "baseline rule model",
        "logistic_regression": "logistic regression champion model",
        "random_forest": "random forest champion model",
        "pytorch_neural_network": "PyTorch neural network champion model",
    }
    return labels.get(model_key, fallback)


def get_available_prediction_file(game_id: str) -> tuple[Path, str]:
    champion = load_champion_metadata()
    champion_key = champion.get("model_key", "baseline")
    champion_prefix = get_prediction_file_prefix(champion_key)
    champion_path = PROCESSED_DIR / f"{champion_prefix}_{game_id}.csv"

    if champion_path.exists():
        return champion_path, prediction_label_from_key(
            champion_key,
            champion.get("model_name", "champion model"),
        )

    candidates = [
        (PROCESSED_DIR / f"advanced_predictions_{game_id}.csv", "random forest model"),
        (PROCESSED_DIR / f"ml_predictions_{game_id}.csv", "logistic regression model"),
        (PROCESSED_DIR / f"neural_predictions_{game_id}.csv", "PyTorch neural network model"),
        (PROCESSED_DIR / f"baseline_predictions_{game_id}.csv", "baseline rule model"),
    ]

    for path, label in candidates:
        if path.exists():
            return path, label

    raise FileNotFoundError(f"No prediction file found for game {game_id}. Run the pipeline first.")


def load_predictions(game_id: str) -> tuple[pd.DataFrame, str]:
    path, prediction_label = get_available_prediction_file(game_id)
    print(f"Loading predictions from: {path}")
    return pd.read_csv(path, dtype={"game_id": str}), prediction_label


def describe_result(home_team: str, away_team: str, home_score: int, away_score: int) -> str:
    margin = home_score - away_score

    if margin > 0:
        return f"{home_team} beat {away_team} {home_score}-{away_score} by {margin} {pluralize_point(margin)}."

    if margin < 0:
        margin = abs(margin)
        return f"{away_team} beat {home_team} {away_score}-{home_score} by {margin} {pluralize_point(margin)}."

    return f"{home_team} and {away_team} finished tied {home_score}-{away_score}."


def first_or_none(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.iloc[0]


def build_recap(
    game_id: str,
    predictions: pd.DataFrame,
    turning_points: pd.DataFrame,
    player_impact: pd.DataFrame,
    comeback_report: pd.DataFrame,
    momentum_report: pd.DataFrame,
    prediction_label: str,
) -> str:
    home_team, away_team = get_team_labels(game_id)
    final_row = predictions.iloc[-1]
    home_score = int(final_row["home_score"])
    away_score = int(final_row["away_score"])

    result_sentence = describe_result(home_team, away_team, home_score, away_score)
    top_turning_point = first_or_none(filter_ranked_report(turning_points, require_player_team=True))
    top_player = first_or_none(filter_ranked_report(player_impact, require_player_team=True))
    top_comeback = first_or_none(comeback_report)
    top_momentum = first_or_none(filter_ranked_report(momentum_report, require_player_team=False))

    recap_lines = [
        "# ClutchCast AI Post-Game Recap",
        "",
        f"**Game:** {away_team} at {home_team}",
        f"**Game ID:** `{game_id}`",
        f"**Model Used:** {prediction_label}",
        "",
        "## Final Result",
        "",
        result_sentence,
    ]

    if top_turning_point is not None:
        when = format_period_clock(int(top_turning_point["period"]), top_turning_point["clock"])
        turning_before = float(top_turning_point["wp_before_pct"])
        turning_after = float(top_turning_point["wp_after_pct"])
        turning_swing = float(top_turning_point["wp_swing_pct"])
        turning_play = str(top_turning_point["event_description"])
        recap_lines.extend([
            "",
            "## Biggest Turning Point",
            "",
            (
                f"The game's sharpest probability swing came at **{when}**, when home win probability "
                f"moved from **{turning_before:.1f}%** to **{turning_after:.1f}%** "
                f"(**{turning_swing:+.1f} percentage points**)."
            ),
            "",
            f"**Key play:** {turning_play}",
        ])

    if top_player is not None:
        player_name = str(top_player["event_player"])
        player_team = str(top_player["event_team"])
        player_impact_value = float(top_player["total_absolute_swing_pct"])
        player_events = int(top_player["event_count"])
        recap_lines.extend([
            "",
            "## Player Impact",
            "",
            (
                f"**{player_name} ({player_team})** led the player-impact table with "
                f"**{player_impact_value:.1f} total win-probability impact points** "
                f"across **{player_events} tracked events**."
            ),
        ])

    if top_comeback is not None:
        when = format_period_clock(int(top_comeback["period"]), top_comeback["clock"])
        trailing_team = str(top_comeback["trailing_team"])
        deficit = int(top_comeback["deficit"])
        comeback_probability = float(top_comeback["comeback_probability_pct"])
        comeback_status = str(top_comeback["comeback_status"])
        recap_lines.extend([
            "",
            "## Comeback Reality",
            "",
            (
                f"The most interesting comeback window came at **{when}**. **{trailing_team}** "
                f"trailed by **{deficit} {pluralize_point(deficit)}** with an estimated comeback "
                f"chance of **{comeback_probability:.1f}%**, which ClutchCast labeled **{comeback_status}**."
            ),
        ])

    if top_momentum is not None:
        when = format_period_clock(int(top_momentum["period"]), top_momentum["clock"])
        momentum_score = float(top_momentum["hidden_momentum_score"])
        momentum_label = str(top_momentum["momentum_label"])
        momentum_play = str(top_momentum["event_description"])
        recap_lines.extend([
            "",
            "## Hidden Momentum",
            "",
            (
                f"The strongest hidden-momentum reading came at **{when}**, with a score of "
                f"**{momentum_score:.1f}** and a label of **{momentum_label}**."
            ),
            "",
            f"**Momentum play:** {momentum_play}",
        ])

    recap_lines.extend([
        "",
        "## Model Note",
        "",
        (
            "ClutchCast selects its champion model using probability-quality metrics: Brier score first, "
            "then log loss, ROC-AUC, and accuracy. The recap uses the champion prediction file when available."
        ),
        "",
    ])

    return "\n".join(recap_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a polished ClutchCast AI post-game recap.")
    parser.add_argument("--game-id", type=str, default=None, help="Specific NBA game ID to recap, example: 0042300312.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.game_id:
        game_id = str(args.game_id).zfill(10)
    else:
        prediction_files = sorted(PROCESSED_DIR.glob("*_predictions_*.csv"))
        if not prediction_files:
            raise FileNotFoundError("No prediction files found. Run the pipeline first.")
        game_id = prediction_files[-1].stem.split("_")[-1]

    predictions, prediction_label = load_predictions(game_id)
    turning_points = load_csv(REPORTS_DIR, "turning_points_{game_id}.csv", game_id)
    player_impact = load_csv(REPORTS_DIR, "player_impact_{game_id}.csv", game_id)
    comeback_report = load_csv(REPORTS_DIR, "comeback_report_{game_id}.csv", game_id)
    momentum_report = load_csv(REPORTS_DIR, "momentum_report_{game_id}.csv", game_id)

    recap = build_recap(
        game_id=game_id,
        predictions=predictions,
        turning_points=turning_points,
        player_impact=player_impact,
        comeback_report=comeback_report,
        momentum_report=momentum_report,
        prediction_label=prediction_label,
    )

    output_path = REPORTS_DIR / f"post_game_recap_{game_id}.md"
    output_path.write_text(recap, encoding="utf-8")

    print("\nSuccess.")
    print(f"Saved post-game recap to: {output_path}")
    print("\nGenerated recap:")
    print(recap)


if __name__ == "__main__":
    main()
