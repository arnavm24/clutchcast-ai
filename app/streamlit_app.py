from pathlib import Path
import html
import sys
import textwrap

import pandas as pd
import plotly.express as px
import streamlit as st
from nba_api.stats.endpoints import boxscoresummaryv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from champion_inference import load_champion_metadata


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

DEFAULT_HOME_COLOR = "#3B82F6"
DEFAULT_AWAY_COLOR = "#EF4444"

MODE_LABELS = {
    "baseline": "Baseline Model",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "pytorch_neural_network": "Neural Network",
}

MODE_FILES = {
    "baseline": "baseline_predictions_{game_id}.csv",
    "logistic_regression": "ml_predictions_{game_id}.csv",
    "random_forest": "advanced_predictions_{game_id}.csv",
    "pytorch_neural_network": "neural_predictions_{game_id}.csv",
}

TEAM_IDS = {
    "ATL": "1610612737", "BOS": "1610612738", "BKN": "1610612751", "CHA": "1610612766",
    "CHI": "1610612741", "CLE": "1610612739", "DAL": "1610612742", "DEN": "1610612743",
    "DET": "1610612765", "GSW": "1610612744", "HOU": "1610612745", "IND": "1610612754",
    "LAC": "1610612746", "LAL": "1610612747", "MEM": "1610612763", "MIA": "1610612748",
    "MIL": "1610612749", "MIN": "1610612750", "NOP": "1610612740", "NYK": "1610612752",
    "OKC": "1610612760", "ORL": "1610612753", "PHI": "1610612755", "PHX": "1610612756",
    "POR": "1610612757", "SAC": "1610612758", "SAS": "1610612759", "TOR": "1610612761",
    "UTA": "1610612762", "WAS": "1610612764",
}

TEAM_COLORS = {
    "ATL": "#E03A3E", "BOS": "#007A33", "BKN": "#FFFFFF", "CHA": "#1D1160",
    "CHI": "#CE1141", "CLE": "#860038", "DAL": "#00538C", "DEN": "#FEC524",
    "DET": "#C8102E", "GSW": "#1D428A", "HOU": "#CE1141", "IND": "#FDBB30",
    "LAC": "#C8102E", "LAL": "#FDB927", "MEM": "#5D76A9", "MIA": "#98002E",
    "MIL": "#00471B", "MIN": "#0C2340", "NOP": "#0C2340", "NYK": "#F58426",
    "OKC": "#007AC1", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#E56020",
    "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#C4CED4", "TOR": "#CE1141",
    "UTA": "#002B5C", "WAS": "#002B5C",
}

st.set_page_config(page_title="ClutchCast AI", page_icon="🏀", layout="wide")


def render_html(markup: str) -> None:
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


def apply_custom_css() -> None:
    render_html(
        """
        <style>
        :root { --panel: #101621; --panel-soft: #151D2A; --line: #263244; --text: #F8FAFC; --muted: #94A3B8; }
        .stApp { background: radial-gradient(circle at top left, #162033 0, #070A12 34%, #05070D 100%); color: var(--text); }
        [data-testid="stSidebar"] { background-color: #070A12; border-right: 1px solid #1F2937; }
        .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1500px; }
        h1, h2, h3 { letter-spacing: 0; }
        .eyebrow { color: #8EA0BA; font-size: .72rem; text-transform: uppercase; letter-spacing: .14em; font-weight: 700; }
        .hero-shell { border: 1px solid rgba(148,163,184,.22); background: linear-gradient(135deg, rgba(16,22,34,.96), rgba(6,10,18,.98)); border-radius: 20px; padding: 20px; box-shadow: 0 24px 80px rgba(0,0,0,.42); }
        .scoreboard { display: grid; grid-template-columns: 1fr 190px 1fr; gap: 18px; align-items: center; }
        .team-box { min-height: 188px; border: 1px solid rgba(148,163,184,.18); border-radius: 18px; padding: 18px; background: linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.02)); display: flex; align-items: center; gap: 18px; }
        .team-box.home { flex-direction: row-reverse; text-align: right; }
        .team-logo-wrap { width: 76px; height: 76px; border-radius: 50%; display:flex; align-items:center; justify-content:center; background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); overflow:hidden; flex: 0 0 auto; }
        .team-logo { width: 68px; height: 68px; object-fit: contain; filter: drop-shadow(0 8px 20px rgba(0,0,0,.45)); }
        .team-fallback { width: 76px; height: 76px; border-radius: 50%; display:flex; align-items:center; justify-content:center; font-weight: 900; color:#07111F; background:#E5E7EB; flex: 0 0 auto; }
        .team-name { color: #AAB7CA; font-size: .82rem; text-transform: uppercase; letter-spacing: .11em; font-weight: 800; }
        .team-score { font-size: 4.55rem; line-height: .92; font-weight: 900; color: #F8FAFC; }
        .team-prob-label { margin-top: 12px; color: #8EA0BA; font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 800; }
        .team-prob-big { font-size: 2.5rem; line-height: 1; font-weight: 900; color: #F8FAFC; margin-top: 2px; }
        .clock-card { border-radius: 18px; padding: 22px 14px; text-align:center; background: rgba(5,9,16,.82); border: 1px solid rgba(148,163,184,.18); }
        .clock-value { font-size: 2.05rem; font-weight: 900; color: white; }
        .clock-label { color: #94A3B8; font-size: .78rem; text-transform: uppercase; letter-spacing: .16em; }
        .model-pill { display:inline-block; margin-top: 12px; padding: 6px 10px; border-radius: 999px; background: rgba(59,130,246,.14); border: 1px solid rgba(59,130,246,.32); color: #BFDBFE; font-size: .74rem; font-weight: 700; }
        .wp-wrap { margin-top: 18px; }
        .wp-row { display:flex; justify-content:space-between; color:#DDE7F4; font-weight: 900; margin-bottom: 8px; font-size: 1.15rem; }
        .wp-row strong { font-size: 1.9rem; line-height: 1; }
        .wp-bar { height: 24px; border-radius: 999px; overflow: hidden; display:flex; background:#111827; border:1px solid rgba(148,163,184,.22); }
        .wp-away, .wp-home { height: 100%; }
        .metric-grid { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 16px; }
        .metric-card, .intel-card, .live-card { border: 1px solid rgba(148,163,184,.18); background: rgba(15,23,42,.78); border-radius: 16px; padding: 16px; min-height: 116px; }
        .metric-label { color: #94A3B8; font-size: .74rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 800; }
        .metric-value { color: #F8FAFC; font-size: 1.36rem; line-height: 1.15; font-weight: 900; margin-top: 8px; }
        .metric-detail { color: #A7B4C8; font-size: .86rem; line-height: 1.35; margin-top: 8px; }
        .intel-card { margin-bottom: 12px; }
        .intel-title { color: #E2E8F0; font-weight: 900; margin-bottom: 6px; }
        .intel-body { color: #AEBBD0; line-height: 1.4; font-size: .9rem; }
        .section-card { border: 1px solid rgba(148,163,184,.16); background: rgba(15,23,42,.62); border-radius: 18px; padding: 16px; }
        div[data-testid="stMetric"] { background: rgba(15,23,42,.78); border: 1px solid rgba(148,163,184,.16); padding: 1rem; border-radius: 14px; }
        .stTabs [data-baseweb="tab-list"] { gap: .35rem; }
        .stTabs [data-baseweb="tab"] { background: rgba(15,23,42,.66); border-radius: 999px; color: #CBD5E1; padding: .5rem 1rem; }
        .stTabs [aria-selected="true"] { background: #E5E7EB !important; color: #111827 !important; }
        @media (max-width: 980px) { .scoreboard { grid-template-columns: 1fr; } .metric-grid { grid-template-columns: 1fr 1fr; } .team-box.home { flex-direction: row; text-align:left; } }
        </style>
        """
    )


def esc(value) -> str:
    return html.escape(str(value), quote=True)


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


def format_period(period: int) -> str:
    return f"Q{period}" if period <= 4 else f"OT{period - 4}"


def team_color(team: str, fallback: str) -> str:
    return TEAM_COLORS.get(str(team).upper(), fallback)


def team_logo_url(team: str) -> str | None:
    team_id = TEAM_IDS.get(str(team).upper())
    if not team_id:
        return None
    return f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"


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

    insights_md_path = REPORTS_DIR / f"game_insights_{game_id}.md"

    return {
        "predictions": pd.read_csv(prediction_path, dtype={"game_id": str}),
        "comparison_summary": load_csv_if_exists(REPORTS_DIR / f"model_comparison_summary_{game_id}.csv"),
        "model_disagreements": load_csv_if_exists(REPORTS_DIR / f"model_disagreements_{game_id}.csv"),
        "leaderboard": load_csv_if_exists(REPORTS_DIR / "model_leaderboard.csv"),
        "game_insights": load_csv_if_exists(REPORTS_DIR / f"game_insights_{game_id}.csv"),
        "turning_points": load_csv_if_exists(REPORTS_DIR / f"turning_points_{game_id}.csv"),
        "player_impact": load_csv_if_exists(REPORTS_DIR / f"player_impact_{game_id}.csv"),
        "comeback_report": load_csv_if_exists(REPORTS_DIR / f"comeback_report_{game_id}.csv"),
        "momentum_report": load_csv_if_exists(REPORTS_DIR / f"momentum_report_{game_id}.csv"),
        "game_insights_md": insights_md_path.read_text(encoding="utf-8") if insights_md_path.exists() else "",
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
        "game_id": "Game ID",
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
        "clutch_pressure": "Clutch Pressure",
        "pressure_level": "Pressure Level",
        "trailing_team": "Trailing Team",
        "deficit": "Deficit",
        "comeback_probability_pct": "Comeback Probability",
        "comeback_status": "Comeback Status",
        "required_points_per_minute": "Required Points / Min",
        "hidden_momentum_score": "Hidden Momentum",
        "momentum_label": "Momentum Label",
        "recent_margin_change": "Recent Margin Change",
        "recent_wp_change_pct": "Recent WP Change",
        "recent_event_value": "Recent Event Value",
        "event_value": "Event Value",
        "rank": "Rank",
        "model_key": "Model Key",
        "model_name": "Model",
        "brier_score": "Brier Score",
        "log_loss": "Log Loss",
        "roc_auc": "ROC-AUC",
        "accuracy": "Accuracy",
        "insight": "Insight",
        "value": "Value",
        "details": "Details",
        "total_raw_home_wp_swing_pct": "Total Home WP Swing",
        "total_absolute_swing_pct": "Total Swing Impact",
        "avg_absolute_swing_pct": "Avg Swing Impact",
        "event_count": "Events",
        "baseline_home_win_prob_pct": "Baseline Home Win Probability",
        "logistic_ml_home_win_prob_pct": "Logistic ML Home Win Probability",
        "advanced_ml_home_win_prob_pct": "Random Forest Home Win Probability",
        "neural_home_win_prob_pct": "Neural Network Home Win Probability",
        "max_model_disagreement_pct": "Max Model Disagreement",
    }
    return display.rename(columns=rename_map)


def is_rankable_event(df: pd.DataFrame) -> pd.Series:
    description = df["event_description"].fillna("").astype(str).str.lower()
    event_team = df["event_team"].fillna("").astype(str).str.strip()
    event_player = df["event_player"].fillna("").astype(str).str.strip()
    return (
        (df["seconds_remaining"] > 0)
        & ~description.str.contains("end of", regex=False)
        & ~description.str.contains("instant replay", regex=False)
        & ~description.str.contains("timeout", regex=False)
        & ~description.str.contains("sub:", regex=False)
        & (event_team != "")
        & (event_player != "")
    )


def get_insight(data: dict, name: str, field: str = "details", default: str = "Run game insights to populate this.") -> str:
    insights = data.get("game_insights", pd.DataFrame())
    if insights.empty or "insight" not in insights.columns:
        return default

    rows = insights[insights["insight"] == name]
    if rows.empty or field not in rows.columns:
        return default

    value = rows.iloc[0][field]
    if pd.isna(value) or str(value).strip() == "":
        return default
    return str(value)


def build_turning_points(predictions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    data = predictions.copy()
    data["wp_before_pct"] = (data["home_win_prob"].shift(1).fillna(data["home_win_prob"]) * 100).round(1)
    data["wp_after_pct"] = data["home_win_prob_pct"]
    data["wp_swing_pct"] = (data["wp_after_pct"] - data["wp_before_pct"]).round(1)
    data = data[(data["abs_wp_change"] > 0) & is_rankable_event(data)]
    return data.sort_values("abs_wp_change", ascending=False).head(top_n)[[
        "period", "clock", "home_score", "away_score", "score_margin_home", "event_team", "event_player", "event_description", "wp_before_pct", "wp_after_pct", "wp_swing_pct"
    ]]


def build_player_impact(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions[is_rankable_event(predictions)].copy()
    if data.empty:
        return pd.DataFrame()
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


def render_logo(team: str, color: str) -> str:
    logo_url = team_logo_url(team)
    if logo_url:
        return f'<div class="team-logo-wrap"><img class="team-logo" src="{esc(logo_url)}" alt="{esc(team)} logo"></div>'
    return f'<div class="team-fallback" style="background:{esc(color)};">{esc(team)[:3]}</div>'


def render_team_block(team: str, score: int, prob: float, color: str, side: str) -> str:
    return (
        f'<div class="team-box {esc(side)}" style="border-color:{esc(color)}55;">'
        f'{render_logo(team, color)}'
        '<div>'
        f'<div class="team-name">{esc(team)}</div>'
        f'<div class="team-score">{score}</div>'
        '<div class="team-prob-label">Win Probability</div>'
        f'<div class="team-prob-big">{prob:.1f}%</div>'
        '</div>'
        '</div>'
    )


def render_metric_card(label: str, value: str, detail: str = "") -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{esc(label)}</div>'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-detail">{esc(detail)}</div>'
        '</div>'
    )


def show_scoreboard(predictions: pd.DataFrame, home_team: str, away_team: str, model_label: str, champion_label: str) -> None:
    row = predictions.iloc[-1]
    home_score = int(row["home_score"])
    away_score = int(row["away_score"])
    home_prob = float(row["home_win_prob_pct"])
    away_prob = float(row["away_win_prob_pct"])
    home_color = team_color(home_team, DEFAULT_HOME_COLOR)
    away_color = team_color(away_team, DEFAULT_AWAY_COLOR)
    period_label = format_period(int(row["period"]))
    clock = format_nba_clock(row["clock"])

    render_html(
        f"""
        <div class="hero-shell">
          <div class="eyebrow">ClutchCast AI Game Center</div>
          <div class="scoreboard">
            {render_team_block(away_team, away_score, away_prob, away_color, "away")}
            <div class="clock-card">
              <div class="clock-label">{esc(period_label)}</div>
              <div class="clock-value">{esc(clock)}</div>
              <div class="model-pill">{esc(model_label)} &middot; Champion: {esc(champion_label)}</div>
            </div>
            {render_team_block(home_team, home_score, home_prob, home_color, "home")}
          </div>
          <div class="wp-wrap">
            <div class="wp-row"><span>{esc(away_team)} <strong>{away_prob:.1f}%</strong></span><span>{esc(home_team)} <strong>{home_prob:.1f}%</strong></span></div>
            <div class="wp-bar">
              <div class="wp-away" style="width:{away_prob:.3f}%; background: linear-gradient(90deg, {esc(away_color)}, {esc(away_color)}cc);"></div>
              <div class="wp-home" style="width:{home_prob:.3f}%; background: linear-gradient(90deg, {esc(home_color)}cc, {esc(home_color)});"></div>
            </div>
          </div>
        </div>
        """
    )


def short_text(value: str, max_len: int = 130) -> str:
    value = str(value).strip()
    return value if len(value) <= max_len else value[: max_len - 1].rstrip() + "..."


def show_metric_cards(data: dict, champion_label: str) -> None:
    drama = get_insight(data, "Game Drama Score", field="value", default="Pending")
    drama_detail = get_insight(data, "Game Drama Score", field="details", default="Run game insights to calculate drama score.")
    turning = get_insight(data, "Most Valuable Play", field="details", default="Run game insights to identify the key play.")
    damaging = get_insight(data, "Most Damaging Play", field="details", default="Run game insights to identify the damaging play.")

    render_html(
        f"""
        <div class="metric-grid">
          {render_metric_card("Game Drama", f"{drama}/100" if str(drama).isdigit() else drama, short_text(drama_detail, 120))}
          {render_metric_card("Biggest Swing", "Most Valuable Play", short_text(turning, 120))}
          {render_metric_card("Damaging Play", "Loser WP Swing", short_text(damaging, 120))}
          {render_metric_card("Champion Model", champion_label, "Selected by Brier score, log loss, ROC-AUC, then accuracy.")}
        </div>
        """
    )


def best_row_text(df: pd.DataFrame, columns: list[str], fallback: str) -> str:
    if df.empty:
        return fallback
    row = df.iloc[0]
    pieces = []
    for column in columns:
        if column in row and not pd.isna(row[column]):
            pieces.append(str(row[column]))
    return " · ".join(pieces) if pieces else fallback


def show_game_intelligence_panel(data: dict, predictions: pd.DataFrame) -> None:
    comeback = data["comeback_report"] if not data["comeback_report"].empty else build_comeback_report(predictions)
    player = data["player_impact"] if not data["player_impact"].empty else build_player_impact(predictions)
    momentum = data["momentum_report"]
    key_play = get_insight(data, "Most Valuable Play", field="details", default="Run game insights to populate this panel.")

    comeback_text = best_row_text(
        clean_table_columns(comeback),
        ["Quarter", "Clock", "Trailing Team", "Deficit", "Comeback Probability", "Comeback Status"],
        "No comeback report found. Run the pipeline or game insights command.",
    )
    player_text = best_row_text(
        clean_table_columns(player),
        ["Player", "Team", "Total Swing Impact", "Events"],
        "No player impact report found yet.",
    )
    momentum_text = best_row_text(
        clean_table_columns(momentum),
        ["Quarter", "Clock", "Hidden Momentum", "Momentum Label", "Play Description"],
        "No hidden momentum report found yet.",
    )

    render_html(
        f"""
        <div class="section-card">
          <div class="eyebrow">Game Intelligence</div>
          <div class="intel-card"><div class="intel-title">Comeback Reality</div><div class="intel-body">{esc(short_text(comeback_text, 220))}</div></div>
          <div class="intel-card"><div class="intel-title">Hidden Momentum</div><div class="intel-body">{esc(short_text(momentum_text, 220))}</div></div>
          <div class="intel-card"><div class="intel-title">Top Player Impact</div><div class="intel-body">{esc(short_text(player_text, 220))}</div></div>
          <div class="intel-card"><div class="intel-title">Key Play</div><div class="intel-body">{esc(short_text(key_play, 220))}</div></div>
        </div>
        """
    )


def show_live_mode_panel(game_id: str) -> None:
    render_html(
        f"""
        <div class="live-card">
          <div class="eyebrow">Live Mode MVP</div>
          <div class="intel-body">
            This Streamlit dashboard is for historical/completed game analysis. Live prediction support is available through the local Flask/SocketIO backend MVP.<br><br>
            <strong>Run:</strong> <code>python backend/app.py</code><br>
            <strong>Live endpoint:</strong> <code>/predict/{esc(game_id)}?mode=live</code>
          </div>
        </div>
        """
    )


def show_win_probability_chart(
    predictions: pd.DataFrame,
    home_team: str,
    away_team: str,
    champion_view: bool,
    chart_key: str,
) -> None:
    st.subheader("Champion Win Probability Timeline" if champion_view else "Win Probability Timeline")
    chart_data = add_game_time_columns(predictions)
    chart_data_long = chart_data.melt(
        id_vars=["game_minutes_elapsed", "period", "Clock", "home_score", "away_score", "score_margin_home", "event_description"],
        value_vars=["home_win_prob_pct", "away_win_prob_pct"],
        var_name="team",
        value_name="win_probability_pct",
    )
    chart_data_long["team"] = chart_data_long["team"].replace({"home_win_prob_pct": home_team, "away_win_prob_pct": away_team})

    home_color = team_color(home_team, DEFAULT_HOME_COLOR)
    away_color = team_color(away_team, DEFAULT_AWAY_COLOR)
    fig = px.line(
        chart_data_long,
        x="game_minutes_elapsed",
        y="win_probability_pct",
        color="team",
        color_discrete_map={home_team: home_color, away_team: away_color},
        hover_data=["period", "Clock", "home_score", "away_score", "score_margin_home", "event_description"],
        labels={"game_minutes_elapsed": "Game Time", "win_probability_pct": "Win Probability (%)"},
    )
    fig.update_traces(line=dict(width=3))
    fig.update_yaxes(range=[0, 100], gridcolor="#1F2937")
    fig.update_xaxes(gridcolor="#1F2937")
    fig.add_hline(y=50, line_dash="dot", line_color="#9CA3AF")
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0B1020", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", height=430, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, width="stretch", key=chart_key)


def show_game_overview(data: dict, predictions: pd.DataFrame, game_id: str, home_team: str, away_team: str, model_label: str, champion_label: str, champion_view: bool) -> None:
    show_scoreboard(predictions, home_team, away_team, model_label, champion_label)
    show_metric_cards(data, champion_label)

    left, right = st.columns([2.15, 1], gap="large")
    with left:
        show_win_probability_chart(
            predictions,
            home_team,
            away_team,
            champion_view,
            chart_key="overview_win_probability_chart",
        )
    with right:
        show_game_intelligence_panel(data, predictions)
        show_live_mode_panel(game_id)


def show_game_insights(data: dict, game_id: str) -> None:
    st.subheader("Game Insights")

    if data["game_insights_md"]:
        st.markdown(data["game_insights_md"])
    elif not data["game_insights"].empty:
        st.dataframe(clean_table_columns(data["game_insights"]), width="stretch", hide_index=True)
    else:
        st.warning("Game insights report was not found.")
        st.code(f"python src/game_insights.py --game-id {game_id}", language="powershell")


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
        st.markdown("### Live Backend MVP")
        st.caption("Streamlit shows historical analysis. Live polling runs through Flask/SocketIO.")
        st.code("python backend/app.py", language="powershell")

    data = load_dashboard_data(selected_game_id, model_key)
    predictions = data["predictions"]
    home_team, away_team = get_team_labels(selected_game_id)
    model_label = MODE_LABELS.get(model_key, model_key)
    champion_view = model_key == champion_key

    render_html('<div class="eyebrow">NBA Win Probability Platform</div>')
    st.title("ClutchCast AI")
    st.caption(f"{away_team} at {home_team} · Game ID `{selected_game_id}`")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Game Overview", "Win Probability", "Game Insights", "Turning Points", "Player Impact", "Pressure & Comebacks", "Model Evaluation", "Game Recap"
    ])

    with tab1:
        show_game_overview(data, predictions, selected_game_id, home_team, away_team, model_label, champion_label, champion_view)
    with tab2:
        show_win_probability_chart(
            predictions,
            home_team,
            away_team,
            champion_view,
            chart_key="full_win_probability_chart",
        )
    with tab3:
        show_game_insights(data, selected_game_id)
    with tab4:
        st.dataframe(clean_table_columns(build_turning_points(predictions)), width="stretch", hide_index=True)
    with tab5:
        st.dataframe(clean_table_columns(build_player_impact(predictions)), width="stretch", hide_index=True)
    with tab6:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Clutch Pressure")
            st.dataframe(clean_table_columns(calculate_clutch_pressure(predictions).sort_values("clutch_pressure", ascending=False).head(15)), width="stretch", hide_index=True)
        with col2:
            st.subheader("Comeback Reality")
            st.dataframe(clean_table_columns(build_comeback_report(predictions)), width="stretch", hide_index=True)
    with tab7:
        show_model_evaluation(data, champion)
    with tab8:
        st.markdown(data["recap"])


if __name__ == "__main__":
    main()
