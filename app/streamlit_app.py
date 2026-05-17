from pathlib import Path
import json
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from nba_api.stats.endpoints import boxscoresummaryv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from champion_inference import get_prediction_file_prefix, load_champion_metadata


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

HOME_COLOR = "#3B82F6"
AWAY_COLOR = "#EF4444"

MODE_LABELS = {
    "baseline": "Baseline Model",
    "logistic_regression": "ML Model",
    "random_forest": "Advanced ML Model",
    "pytorch_neural_network": "Neural Network Model",
}

MODE_FILES = {
    "baseline": "baseline_predictions_{game_id}.csv",
    "logistic_regression": "ml_predictions_{game_id}.csv",
    "random_forest": "advanced_predictions_{game_id}.csv",
    "pytorch_neural_network": "neural_predictions_{game_id}.csv",
}

st.set_page_config(page_title="ClutchCast AI", page_icon="🏀", layout="wide")


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #070A12; color: #F9FAFB; }
        [data-testid="stSidebar"] { background-color: #080B12; border-right: 1px solid #1F2937; }
        div[data-testid="stMetric"] { background: rgba(17,24,39,.9); border: 1px solid #1F2937; padding: 1rem; border-radius: 12px; }
        h1, h2, h3 { letter-spacing: -0.03em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_nba_clock(clock_value) -> str:
    clock = str(clock_value)
    if not clock.startswith("PT"):
        return clock

    clock = clock.replace("PT", "")
    minutes = 0
    seconds = 0

    if "M" in clock:
        minutes_part, clock = clock.split("M")
        minutes = int(minutes_part)

    if "S" in clock:
        seconds = float(clock.replace("S", ""))

    if seconds.is_integer():
        return f"{minutes}:{int(seconds):02d}"

    return f"{minutes}:{seconds:04.1f}"


def get_available_game_ids() -> list[str]:
    game_ids = set()
    for pattern in MODE_FILES.values():
        prefix, suffix = pattern.split("{game_id}")
        for file in PROCESSED_DIR.glob(pattern.format(game_id="*")):
            game_ids.add(file.name.replace(prefix, "").replace(suffix, ""))
    return sorted(game_ids)


def get_available_modes(game_id: str) -> list[str]:
    return [
        mode_key
        for mode_key, pattern in MODE_FILES.items()
        if (PROCESSED_DIR / pattern.format(game_id=game_id)).exists()
    ]


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype={"game_id": str})
    return pd.DataFrame()


def get_prediction_path(game_id: str, model_key: str) -> Path:
    return PROCESSED_DIR / MODE_FILES[model_key].format(game_id=game_id)


@st.cache_data(show_spinner=False)
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


def load_dashboard_data(game_id: str, model_key: str) -> dict:
    prediction_path = get_prediction_path(game_id, model_key)
    if not prediction_path.exists():
        st.error(f"Missing prediction file: `{prediction_path}`")
        st.info(f"Run `python src/run_pipeline.py --game-id {game_id} --model baseline|ml|advanced|neural`")
        st.stop()

    return {
        "predictions": pd.read_csv(prediction_path, dtype={"game_id": str}),
        "comparison_summary": load_csv_if_exists(REPORTS_DIR / f"model_comparison_summary_{game_id}.csv"),
        "model_disagreements": load_csv_if_exists(REPORTS_DIR / f"model_disagreements_{game_id}.csv"),
        "leaderboard": load_csv_if_exists(REPORTS_DIR / "model_leaderboard.csv"),
        "recap": (REPORTS_DIR / f"post_game_recap_{game_id}.md").read_text(encoding="utf-8")
        if (REPORTS_DIR / f"post_game_recap_{game_id}.md").exists()
        else "No recap file found. Run `python src/recap.py --game-id YOUR_GAME_ID`.",
    }


def add_game_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["game_minutes_elapsed"] = ((48 * 60) - output["seconds_remaining"]) / 60
    output["Clock"] = output["clock"].apply(format_nba_clock)
    return output


def clean_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    if "clock" in display.columns:
        display["clock"] = display["clock"].apply(format_nba_clock)

    rename_map = {
        "period": "Quarter",
        "clock": "Clock",
        "home_score": "Home Score",
        "away_score": "Away Score",
        "score_margin_home": "Home Margin",
        "event_team": "Team",
        "event_player": "Player",
        "event_description": "Play Description",
        "home_win_prob_pct": "Home Win Probability",
        "away_win_prob_pct": "Away Win Probability",
        "wp_before_pct": "Win Prob. Before",
        "wp_after_pct": "Win Prob. After",
        "wp_swing_pct": "Win Prob. Swing",
        "rank": "Rank",
        "model_name": "Model",
        "brier_score": "Brier Score",
        "log_loss": "Log Loss",
        "roc_auc": "ROC-AUC",
        "accuracy": "Accuracy",
        "baseline_home_win_prob_pct": "Baseline Home Win Probability",
        "logistic_ml_home_win_prob_pct": "Logistic ML Home Win Probability",
        "advanced_ml_home_win_prob_pct": "Random Forest Home Win Probability",
        "neural_home_win_prob_pct": "Neural Network Home Win Probability",
        "max_model_disagreement_pct": "Max Model Disagreement",
    }
    return display.rename(columns=rename_map)


def show_header(home_team: str, away_team: str, model_label: str, champion_label: str) -> None:
    st.title("ClutchCast AI")
    st.caption("NBA win probability, turning points, player impact, comeback pressure, and model evaluation.")
    st.markdown(f"**{away_team} at {home_team}** · Active view: `{model_label}` · Champion: `{champion_label}`")


def show_game_summary(predictions: pd.DataFrame, home_team: str, away_team: str, model_label: str) -> None:
    final_row = predictions.iloc[-1]
    home_score = int(final_row["home_score"])
    away_score = int(final_row["away_score"])
    margin = home_score - away_score
    result = f"{home_team} by {margin}" if margin > 0 else f"{away_team} by {abs(margin)}" if margin < 0 else "Tie"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Score", f"{home_team} {home_score} - {away_team} {away_score}")
    col2.metric("Result", result)
    col3.metric(f"{home_team} Win Prob.", f"{float(final_row['home_win_prob_pct']):.1f}%")
    col4.metric("Model", model_label)


def show_win_probability_chart(predictions: pd.DataFrame, home_team: str, away_team: str, champion_view: bool) -> None:
    st.subheader("Champion Win Probability Timeline" if champion_view else "Win Probability Timeline")
    chart_data = add_game_time_columns(predictions)
    chart_data_long = chart_data.melt(
        id_vars=["game_minutes_elapsed", "period", "Clock", "home_score", "away_score", "score_margin_home", "event_description"],
        value_vars=["home_win_prob_pct", "away_win_prob_pct"],
        var_name="team",
        value_name="win_probability_pct",
    )
    chart_data_long["team"] = chart_data_long["team"].replace({"home_win_prob_pct": home_team, "away_win_prob_pct": away_team})

    fig = px.line(
        chart_data_long,
        x="game_minutes_elapsed",
        y="win_probability_pct",
        color="team",
        color_discrete_map={home_team: HOME_COLOR, away_team: AWAY_COLOR},
        hover_data=["period", "Clock", "home_score", "away_score", "score_margin_home", "event_description"],
        labels={"game_minutes_elapsed": "Game Time", "win_probability_pct": "Win Probability (%)"},
    )
    fig.update_traces(line=dict(width=3))
    fig.update_yaxes(range=[0, 100], gridcolor="#1F2937")
    fig.update_xaxes(gridcolor="#1F2937")
    fig.add_hline(y=50, line_dash="dot", line_color="#9CA3AF")
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0B1020", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", height=430)
    st.plotly_chart(fig, width="stretch")


def build_turning_points(predictions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    data = predictions.copy()
    data["wp_before_pct"] = (data["home_win_prob"].shift(1).fillna(data["home_win_prob"]) * 100).round(1)
    data["wp_after_pct"] = data["home_win_prob_pct"]
    data["wp_swing_pct"] = (data["wp_after_pct"] - data["wp_before_pct"]).round(1)
    return data.sort_values("abs_wp_change", ascending=False).head(top_n)[[
        "period", "clock", "home_score", "away_score", "score_margin_home", "event_team", "event_player", "event_description", "wp_before_pct", "wp_after_pct", "wp_swing_pct"
    ]]


def build_player_impact(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions[predictions["event_player"].fillna("").astype(str).str.strip() != ""].copy()
    data["positive_swing_pct"] = (data["wp_change"] * 100).round(2)
    data["absolute_swing_pct"] = (data["abs_wp_change"] * 100).round(2)
    grouped = data.groupby(["event_player", "event_team"], as_index=False).agg(
        total_raw_home_wp_swing_pct=("positive_swing_pct", "sum"),
        total_absolute_swing_pct=("absolute_swing_pct", "sum"),
        avg_absolute_swing_pct=("absolute_swing_pct", "mean"),
        event_count=("event_player", "count"),
    )
    grouped = grouped.round(2).sort_values("total_absolute_swing_pct", ascending=False).reset_index(drop=True)
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    return grouped


def calculate_clutch_pressure(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    elapsed_ratio = (1 - (output["seconds_remaining"] / (48 * 60))).clip(0, 1)
    closeness = (1 - (output["abs_score_margin"] / 20)).clip(0, 1)
    uncertainty = 1 - ((output["home_win_prob"] - 0.5).abs() / 0.5).clip(0, 1)
    output["clutch_pressure"] = ((0.40 * closeness + 0.35 * elapsed_ratio + 0.25 * uncertainty) * 100).round(1)
    output["pressure_level"] = pd.cut(output["clutch_pressure"], bins=[-1, 25, 50, 75, 100], labels=["Low", "Medium", "High", "Extreme"])
    return output


def build_comeback_report(predictions: pd.DataFrame) -> pd.DataFrame:
    data = calculate_clutch_pressure(predictions)
    data["trailing_team"] = "Tie"
    data.loc[data["score_margin_home"] < 0, "trailing_team"] = "Home"
    data.loc[data["score_margin_home"] > 0, "trailing_team"] = "Away"
    data["deficit"] = data["score_margin_home"].abs()
    data["comeback_probability"] = 0.5
    data.loc[data["score_margin_home"] < 0, "comeback_probability"] = data["home_win_prob"]
    data.loc[data["score_margin_home"] > 0, "comeback_probability"] = data["away_win_prob"]
    data["comeback_probability_pct"] = (data["comeback_probability"] * 100).round(1)
    data["required_points_per_minute"] = data.apply(lambda row: round(row["deficit"] / max(row["seconds_remaining"] / 60, 0.01), 2) if row["deficit"] > 0 else 0, axis=1)
    data["comeback_status"] = pd.cut(data["comeback_probability"], bins=[-1, .03, .10, .25, .40, 1], labels=["Nearly impossible", "Very unlikely", "Difficult", "Possible", "Very realistic"])
    rows = data[(data["trailing_team"] != "Tie") & (data["seconds_remaining"] > 0) & (data["deficit"] >= 5)].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["interest_score"] = rows["deficit"] * .45 + rows["clutch_pressure"] * .35 + rows["comeback_probability_pct"] * .20
    return rows.sort_values("interest_score", ascending=False).head(10)[[
        "period", "clock", "home_score", "away_score", "trailing_team", "deficit", "comeback_probability_pct", "comeback_status", "required_points_per_minute", "clutch_pressure", "pressure_level", "event_description"
    ]]


def show_model_evaluation(data: dict, champion: dict) -> None:
    st.subheader("Model Evaluation")
    st.caption("Champion selection ranks models by lowest Brier score, then lowest log loss, highest ROC-AUC, and highest accuracy.")

    leaderboard = data["leaderboard"]
    if leaderboard.empty:
        st.warning("Model leaderboard was not found.")
        st.code("python src/compare_models.py --leaderboard", language="powershell")
    else:
        champion_key = champion.get("model_key")
        display = leaderboard.copy()
        display["Champion"] = display["model_key"].eq(champion_key).map({True: "Yes", False: ""})
        st.dataframe(clean_table_columns(display), width="stretch", hide_index=True)

    summary = data["comparison_summary"]
    disagreements = data["model_disagreements"]
    if summary.empty or disagreements.empty:
        st.warning("Per-game four-model comparison files were not found.")
        st.code("python src/compare_models.py --game-id YOUR_GAME_ID", language="powershell")
        return

    row = summary.iloc[0]
    cols = st.columns(4)
    for col, label, field in [
        (cols[0], "Baseline", "baseline_final_home_win_prob_pct"),
        (cols[1], "Logistic", "logistic_ml_final_home_win_prob_pct"),
        (cols[2], "Random Forest", "advanced_ml_final_home_win_prob_pct"),
        (cols[3], "Neural Net", "neural_final_home_win_prob_pct"),
    ]:
        col.metric(label, f"{float(row[field]):.1f}%")

    st.markdown("### Biggest Model Disagreement Moments")
    st.dataframe(clean_table_columns(disagreements), width="stretch", hide_index=True)


def main() -> None:
    apply_custom_css()
    champion = load_champion_metadata()
    champion_key = champion.get("model_key", "baseline")
    champion_label = champion.get("model_name", MODE_LABELS.get(champion_key, "Baseline Model"))

    available_game_ids = get_available_game_ids()
    if not available_game_ids:
        st.error("No analyzed games found.")
        st.code("python src/run_pipeline.py --game-id YOUR_GAME_ID --model baseline", language="powershell")
        st.stop()

    with st.sidebar:
        st.markdown("## ClutchCast AI")
        selected_game_id = st.selectbox("Analyzed game", available_game_ids, index=len(available_game_ids) - 1)
        available_modes = get_available_modes(selected_game_id)
        default_mode = champion_key if champion_key in available_modes else available_modes[-1]
        st.markdown(f"**Champion Model:** {champion_label}")
        st.caption("Main dashboard defaults to the champion when its prediction file exists.")
        advanced = st.checkbox("Inspect another model")
        if advanced:
            model_key = st.selectbox("Model view", available_modes, index=available_modes.index(default_mode), format_func=lambda key: MODE_LABELS[key])
        else:
            model_key = default_mode
        st.divider()
        st.caption("Generate champion report with `python src/compare_models.py --leaderboard`.")

    data = load_dashboard_data(selected_game_id, model_key)
    predictions = data["predictions"]
    home_team, away_team = get_team_labels(selected_game_id)
    model_label = MODE_LABELS.get(model_key, model_key)

    show_header(home_team, away_team, model_label, champion_label)
    show_game_summary(predictions, home_team, away_team, model_label)
    show_win_probability_chart(predictions, home_team, away_team, champion_view=(model_key == champion_key))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Turning Points", "Player Impact", "Clutch Pressure", "Comeback Reality", "Model Evaluation", "Game Recap"
    ])

    with tab1:
        st.dataframe(clean_table_columns(build_turning_points(predictions)), width="stretch", hide_index=True)
    with tab2:
        st.dataframe(clean_table_columns(build_player_impact(predictions)), width="stretch", hide_index=True)
    with tab3:
        st.dataframe(clean_table_columns(calculate_clutch_pressure(predictions).sort_values("clutch_pressure", ascending=False).head(15)), width="stretch", hide_index=True)
    with tab4:
        st.dataframe(clean_table_columns(build_comeback_report(predictions)), width="stretch", hide_index=True)
    with tab5:
        show_model_evaluation(data, champion)
    with tab6:
        st.markdown(data["recap"])


if __name__ == "__main__":
    main()
