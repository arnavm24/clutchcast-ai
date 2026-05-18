from pathlib import Path
import html
import json
import re
import sys
import textwrap
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from nba_api.stats.endpoints import boxscoresummaryv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from champion_inference import load_champion_metadata


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
BACKEND_BASE_URL = "http://127.0.0.1:5000"

DEFAULT_HOME_COLOR = "#3B82F6"
DEFAULT_AWAY_COLOR = "#EF4444"
CHART_AWAY_COLOR = "#38BDF8"
CHART_HOME_COLOR = "#F43F5E"

CLUTCHCAST_ICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs><radialGradient id="g" cx="30%" cy="25%" r="80%"><stop offset="0" stop-color="#FDBA74"/><stop offset=".42" stop-color="#F97316"/><stop offset="1" stop-color="#1D4ED8"/></radialGradient></defs>
  <rect width="64" height="64" rx="18" fill="#070A12"/>
  <circle cx="32" cy="32" r="23" fill="url(#g)"/>
  <path d="M14 34h36M31 9c6 12 6 34 0 46M13 25c12 5 26 5 38 0M13 43c12-5 26-5 38 0" stroke="rgba(255,255,255,.48)" stroke-width="2.2" fill="none" stroke-linecap="round"/>
  <text x="32" y="39" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="900" fill="#F8FAFC">CC</text>
</svg>
""".strip()
CLUTCHCAST_ICON_DATA_URL = "data:image/svg+xml;charset=utf-8," + quote(CLUTCHCAST_ICON_SVG)

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

st.set_page_config(page_title="ClutchCast AI", page_icon=CLUTCHCAST_ICON_DATA_URL, layout="wide")


def render_html(markup: str) -> None:
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def apply_custom_css() -> None:
    render_html(
        """
        <style>
        :root { --panel:#101621; --line:#263244; --text:#F8FAFC; --muted:#94A3B8; }
        .stApp { background: radial-gradient(circle at top left, #162033 0, #070A12 34%, #05070D 100%); color: var(--text); }
        [data-testid="stSidebar"] { background-color:#070A12; border-right:1px solid #1F2937; }
        .block-container { padding-top:3.15rem; padding-bottom:2rem; max-width:1500px; }
        h1, h2, h3 { letter-spacing:0; }
        .eyebrow { color:#8EA0BA; font-size:.72rem; text-transform:uppercase; letter-spacing:.14em; font-weight:800; }
        .brand-header { display:flex; align-items:center; justify-content:space-between; gap:18px; margin:.8rem 0 18px; }
        .brand-left { display:flex; align-items:center; gap:14px; }
        .brand-mark { width:54px; height:54px; border-radius:18px; position:relative; display:flex; align-items:center; justify-content:center; color:#F8FAFC; font-weight:950; background:radial-gradient(circle at 30% 25%, #FDBA74 0, #F97316 38%, #1D4ED8 100%); box-shadow:0 16px 45px rgba(29,78,216,.32); overflow:hidden; }
        .brand-mark:before { content:""; position:absolute; inset:10px; border:2px solid rgba(255,255,255,.42); border-radius:50%; }
        .brand-mark:after { content:""; position:absolute; width:92px; height:2px; background:rgba(255,255,255,.38); transform:rotate(-28deg); }
        .brand-cc { position:relative; z-index:2; font-size:1.15rem; text-shadow:0 2px 10px rgba(0,0,0,.5); }
        .brand-title { font-size:2.05rem; line-height:1; font-weight:950; color:#F8FAFC; letter-spacing:-.03em; }
        .brand-subtitle { margin-top:5px; color:#9FB0C8; font-size:.88rem; }
        .brand-badge, .status-badge { border:1px solid rgba(148,163,184,.22); background:rgba(15,23,42,.72); border-radius:999px; padding:8px 12px; color:#CBD5E1; font-size:.82rem; font-weight:800; display:inline-block; }
        .status-ok { color:#BBF7D0; border-color:rgba(34,197,94,.35); background:rgba(22,101,52,.24); }
        .status-bad { color:#FECACA; border-color:rgba(239,68,68,.35); background:rgba(127,29,29,.28); }
        .hero-shell { border:1px solid rgba(148,163,184,.22); background:linear-gradient(135deg, rgba(16,22,34,.96), rgba(6,10,18,.98)); border-radius:20px; padding:20px; box-shadow:0 24px 80px rgba(0,0,0,.42); }
        .scoreboard { display:grid; grid-template-columns:1fr 190px 1fr; gap:18px; align-items:center; }
        .team-box { min-height:188px; border:1px solid rgba(148,163,184,.18); border-radius:18px; padding:18px; background:linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.02)); display:flex; align-items:center; gap:18px; }
        .team-box.home { flex-direction:row-reverse; text-align:right; }
        .team-logo-wrap, .team-fallback { width:76px; height:76px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:0 0 auto; }
        .team-logo-wrap { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.18); overflow:hidden; }
        .team-logo { width:68px; height:68px; object-fit:contain; filter:drop-shadow(0 8px 20px rgba(0,0,0,.45)); }
        .team-fallback { font-weight:900; color:#07111F; background:#E5E7EB; }
        .team-name { color:#AAB7CA; font-size:.82rem; text-transform:uppercase; letter-spacing:.11em; font-weight:800; }
        .team-score { font-size:4.55rem; line-height:.92; font-weight:900; color:#F8FAFC; }
        .team-prob-label { margin-top:12px; color:#8EA0BA; font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; font-weight:800; }
        .team-prob-big { font-size:2.5rem; line-height:1; font-weight:900; color:#F8FAFC; margin-top:2px; }
        .clock-card { border-radius:18px; padding:22px 14px; text-align:center; background:rgba(5,9,16,.82); border:1px solid rgba(148,163,184,.18); }
        .clock-value { font-size:2.05rem; font-weight:900; color:white; }
        .clock-label { color:#94A3B8; font-size:.78rem; text-transform:uppercase; letter-spacing:.16em; }
        .model-pill { display:inline-block; margin-top:12px; padding:6px 10px; border-radius:999px; background:rgba(59,130,246,.14); border:1px solid rgba(59,130,246,.32); color:#BFDBFE; font-size:.74rem; font-weight:800; }
        .wp-wrap { margin-top:18px; }
        .wp-row { display:flex; justify-content:space-between; color:#DDE7F4; font-weight:900; margin-bottom:8px; font-size:1.15rem; }
        .wp-row strong { font-size:1.9rem; line-height:1; }
        .wp-bar { height:24px; border-radius:999px; overflow:hidden; display:flex; background:#111827; border:1px solid rgba(148,163,184,.22); }
        .wp-away, .wp-home { height:100%; }
        .metric-grid, .summary-grid, .story-grid, .insight-grid, .recap-grid { display:grid; gap:14px; margin:14px 0 18px; }
        .metric-grid, .summary-grid { grid-template-columns:repeat(4, minmax(0, 1fr)); }
        .story-grid { grid-template-columns:1.35fr 1fr 1fr; }
        .insight-grid, .recap-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
        .metric-card, .intel-card, .live-card, .summary-card, .empty-card, .story-shell, .insight-card { border:1px solid rgba(148,163,184,.18); background:rgba(15,23,42,.78); border-radius:16px; padding:16px; min-height:116px; }
        .metric-label, .summary-label, .card-kicker { color:#94A3B8; font-size:.74rem; text-transform:uppercase; letter-spacing:.12em; font-weight:800; }
        .metric-value, .summary-value, .card-value { color:#F8FAFC; font-size:1.36rem; line-height:1.15; font-weight:900; margin-top:8px; }
        .summary-value.big, .card-value.big { font-size:1.7rem; }
        .metric-detail, .summary-detail, .card-detail, .intel-body { color:#A7B4C8; font-size:.88rem; line-height:1.4; margin-top:8px; }
        .tab-intro { color:#AEBBD0; margin:0 0 14px; max-width:950px; }
        .story-title, .intel-title { color:#E2E8F0; font-weight:900; margin-bottom:6px; }
        .story-lede { color:#D7E3F4; font-size:.98rem; line-height:1.45; margin-bottom:12px; }
        .icon-pill, .avatar { width:42px; height:42px; display:flex; align-items:center; justify-content:center; font-weight:950; color:white; }
        .icon-pill { border-radius:14px; background:rgba(59,130,246,.18); color:#DBEAFE; font-size:1.2rem; margin-bottom:10px; }
        .avatar { border-radius:50%; background:linear-gradient(135deg,#F97316,#2563EB); box-shadow:0 12px 28px rgba(37,99,235,.25); }
        .player-chip { display:flex; align-items:center; gap:12px; }
        .section-card { border:1px solid rgba(148,163,184,.16); background:rgba(15,23,42,.62); border-radius:18px; padding:16px; }
        .right-rail-spacer { height:56px; }
        div[data-testid="stMetric"] { background:rgba(15,23,42,.78); border:1px solid rgba(148,163,184,.16); padding:1rem; border-radius:14px; }
        .stTabs [data-baseweb="tab-list"] { gap:.35rem; margin-top:.35rem; flex-wrap:wrap; }
        .stTabs [data-baseweb="tab"] { background:rgba(15,23,42,.66); border-radius:999px; color:#CBD5E1; padding:.5rem 1rem; }
        .stTabs [aria-selected="true"] { background:#E5E7EB !important; color:#111827 !important; }
        @media (max-width:980px) { .scoreboard, .metric-grid, .summary-grid, .story-grid, .insight-grid, .recap-grid { grid-template-columns:1fr; } .team-box.home { flex-direction:row; text-align:left; } .brand-header { align-items:flex-start; flex-direction:column; } .right-rail-spacer { height:0; } }
        </style>
        """
    )


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


def initials(name: str) -> str:
    parts = [part for part in str(name).replace(".", " ").split() if part]
    if not parts:
        return "--"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_available_game_ids() -> list[str]:
    game_ids = set()
    for pattern in MODE_FILES.values():
        prefix, suffix = pattern.split("{game_id}")
        for file in PROCESSED_DIR.glob(pattern.format(game_id="*")):
            game_ids.add(file.name.replace(prefix, "").replace(suffix, ""))
    return sorted(game_ids)


def get_available_modes(game_id: str) -> list[str]:
    return [mode for mode, pattern in MODE_FILES.items() if (PROCESSED_DIR / pattern.format(game_id=game_id)).exists()]


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
    insights_md_path = REPORTS_DIR / f"game_insights_{game_id}.md"
    recap_path = REPORTS_DIR / f"post_game_recap_{game_id}.md"
    return {
        "predictions": pd.read_csv(get_prediction_path(game_id, model_key), dtype={"game_id": str}),
        "comparison_summary": load_csv_if_exists(REPORTS_DIR / f"model_comparison_summary_{game_id}.csv"),
        "model_disagreements": load_csv_if_exists(REPORTS_DIR / f"model_disagreements_{game_id}.csv"),
        "leaderboard": load_csv_if_exists(REPORTS_DIR / "model_leaderboard.csv"),
        "game_insights": load_csv_if_exists(REPORTS_DIR / f"game_insights_{game_id}.csv"),
        "turning_points": load_csv_if_exists(REPORTS_DIR / f"turning_points_{game_id}.csv"),
        "player_impact": load_csv_if_exists(REPORTS_DIR / f"player_impact_{game_id}.csv"),
        "comeback_report": load_csv_if_exists(REPORTS_DIR / f"comeback_report_{game_id}.csv"),
        "momentum_report": load_csv_if_exists(REPORTS_DIR / f"momentum_report_{game_id}.csv"),
        "game_insights_md": insights_md_path.read_text(encoding="utf-8") if insights_md_path.exists() else "",
        "recap": recap_path.read_text(encoding="utf-8") if recap_path.exists() else "No recap file found.",
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
    return display.rename(columns={
        "game_id": "Game ID", "period": "Quarter", "clock": "Clock", "home_score": "Home Score",
        "away_score": "Away Score", "score_margin_home": "Home Margin", "event_team": "Team",
        "event_player": "Player", "event_description": "Play Description", "home_win_prob_pct": "Home Win Probability",
        "away_win_prob_pct": "Away Win Probability", "wp_before_pct": "Win Prob. Before",
        "wp_after_pct": "Win Prob. After", "wp_swing_pct": "Win Prob. Swing", "clutch_pressure": "Clutch Pressure",
        "pressure_level": "Pressure Level", "trailing_team": "Trailing Team", "deficit": "Deficit",
        "comeback_probability_pct": "Comeback Probability", "comeback_status": "Comeback Status",
        "required_points_per_minute": "Required Points / Min", "hidden_momentum_score": "Hidden Momentum",
        "momentum_label": "Momentum Label", "recent_margin_change": "Recent Margin Change", "recent_wp_change_pct": "Recent WP Change",
        "event_value": "Event Value", "rank": "Rank", "model_key": "Model Key", "model_name": "Model",
        "brier_score": "Brier Score", "log_loss": "Log Loss", "roc_auc": "ROC-AUC", "accuracy": "Accuracy",
        "insight": "Insight", "value": "Value", "details": "Details", "total_raw_home_wp_swing_pct": "Total Home WP Swing",
        "total_absolute_swing_pct": "Total Swing Impact", "avg_absolute_swing_pct": "Avg Swing Impact", "event_count": "Events",
        "max_model_disagreement_pct": "Max Model Disagreement",
    })


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
    return default if pd.isna(value) or str(value).strip() == "" else str(value)


def build_turning_points(predictions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    data = predictions.copy()
    data["wp_before_pct"] = (data["home_win_prob"].shift(1).fillna(data["home_win_prob"]) * 100).round(1)
    data["wp_after_pct"] = data["home_win_prob_pct"]
    data["wp_swing_pct"] = (data["wp_after_pct"] - data["wp_before_pct"]).round(1)
    data = data[(data["abs_wp_change"] > 0) & is_rankable_event(data)]
    cols = ["period", "clock", "home_score", "away_score", "score_margin_home", "event_team", "event_player", "event_description", "wp_before_pct", "wp_after_pct", "wp_swing_pct"]
    return data.sort_values("abs_wp_change", ascending=False).head(top_n)[cols]


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
    cols = ["period", "clock", "home_score", "away_score", "trailing_team", "deficit", "comeback_probability_pct", "comeback_status", "required_points_per_minute", "clutch_pressure", "pressure_level", "event_description"]
    return rows.sort_values("interest_score", ascending=False).head(10)[cols]


def render_logo(team: str, color: str) -> str:
    logo_url = team_logo_url(team)
    if logo_url:
        return f'<div class="team-logo-wrap"><img class="team-logo" src="{esc(logo_url)}" alt="{esc(team)} logo"></div>'
    return f'<div class="team-fallback" style="background:{esc(color)};">{esc(team)[:3]}</div>'


def render_team_block(team: str, score: int, prob: float, color: str, side: str) -> str:
    return (
        f'<div class="team-box {esc(side)}" style="border-color:{esc(color)}55;">'
        f'{render_logo(team, color)}<div><div class="team-name">{esc(team)}</div>'
        f'<div class="team-score">{score}</div><div class="team-prob-label">Win Probability</div>'
        f'<div class="team-prob-big">{prob:.1f}%</div></div></div>'
    )


def render_metric_card(label: str, value: str, detail: str = "") -> str:
    return f'<div class="metric-card"><div class="metric-label">{esc(label)}</div><div class="metric-value">{esc(value)}</div><div class="metric-detail">{esc(detail)}</div></div>'


def render_summary_card(label: str, value: str, detail: str = "", big: bool = False, avatar: str | None = None) -> str:
    value_class = "summary-value big" if big else "summary-value"
    avatar_html = f'<div class="avatar">{esc(avatar)}</div>' if avatar else ""
    return f'<div class="summary-card"><div class="player-chip">{avatar_html}<div><div class="summary-label">{esc(label)}</div><div class="{value_class}">{esc(value)}</div></div></div><div class="summary-detail">{esc(detail)}</div></div>'


def render_info_card(title: str, value: str, detail: str, icon: str = "") -> str:
    icon_html = f'<div class="icon-pill">{esc(icon)}</div>' if icon else ""
    return f'<div class="insight-card">{icon_html}<div class="card-kicker">{esc(title)}</div><div class="card-value">{esc(value)}</div><div class="card-detail">{esc(detail)}</div></div>'


def show_empty_report_card(title: str, command: str) -> None:
    render_html(f'<div class="empty-card"><div class="summary-label">{esc(title)}</div><div class="summary-detail">Report data is not available yet.</div><div class="summary-detail"><code>{esc(command)}</code></div></div>')


def show_scoreboard(predictions: pd.DataFrame, home_team: str, away_team: str, model_label: str, champion_label: str) -> None:
    row = predictions.iloc[-1]
    render_scoreboard(
        home_team=home_team,
        away_team=away_team,
        home_score=as_int(row.get("home_score")),
        away_score=as_int(row.get("away_score")),
        home_prob=as_float(row.get("home_win_prob_pct"), 50.0),
        away_prob=as_float(row.get("away_win_prob_pct"), 50.0),
        period=as_int(row.get("period")),
        clock=format_nba_clock(row.get("clock", "")),
        model_label=model_label,
        champion_label=champion_label,
        eyebrow="ClutchCast AI Game Center",
    )


def render_scoreboard(home_team: str, away_team: str, home_score: int, away_score: int, home_prob: float, away_prob: float, period: int, clock: str, model_label: str, champion_label: str, eyebrow: str) -> None:
    home_color = team_color(home_team, DEFAULT_HOME_COLOR)
    away_color = team_color(away_team, DEFAULT_AWAY_COLOR)
    render_html(
        f"""
        <div class="hero-shell">
          <div class="eyebrow">{esc(eyebrow)}</div>
          <div class="scoreboard">
            {render_team_block(away_team, away_score, away_prob, away_color, "away")}
            <div class="clock-card">
              <div class="clock-label">{esc(format_period(period) if period else "Pregame")}</div>
              <div class="clock-value">{esc(clock or "--")}</div>
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


def best_row_text(df: pd.DataFrame, columns: list[str], fallback: str) -> str:
    if df.empty:
        return fallback
    row = df.iloc[0]
    pieces = [str(row[column]) for column in columns if column in row and not pd.isna(row[column])]
    return " · ".join(pieces) if pieces else fallback


def show_metric_cards(data: dict, champion_label: str) -> None:
    drama = get_insight(data, "Game Drama Score", field="value", default="Pending")
    cards = [
        render_metric_card("Game Drama", f"{drama}/100" if str(drama).isdigit() else drama, short_text(get_insight(data, "Game Drama Score"), 120)),
        render_metric_card("Biggest Swing", "Most Valuable Play", short_text(get_insight(data, "Most Valuable Play"), 120)),
        render_metric_card("Damaging Play", "Loser WP Swing", short_text(get_insight(data, "Most Damaging Play"), 120)),
        render_metric_card("Champion Model", champion_label, "Selected by Brier score, log loss, ROC-AUC, then accuracy."),
    ]
    render_html('<div class="metric-grid">' + "".join(cards) + '</div>')


def show_game_intelligence_panel(data: dict, predictions: pd.DataFrame) -> None:
    comeback = data["comeback_report"] if not data["comeback_report"].empty else build_comeback_report(predictions)
    player = data["player_impact"] if not data["player_impact"].empty else build_player_impact(predictions)
    momentum = data["momentum_report"]
    blocks = [
        ("Comeback Reality", best_row_text(clean_table_columns(comeback), ["Quarter", "Clock", "Trailing Team", "Deficit", "Comeback Probability", "Comeback Status"], "No comeback report found.")),
        ("Hidden Momentum", best_row_text(clean_table_columns(momentum), ["Quarter", "Clock", "Hidden Momentum", "Momentum Label", "Play Description"], "No hidden momentum report found.")),
        ("Top Player Impact", best_row_text(clean_table_columns(player), ["Player", "Team", "Total Swing Impact", "Events"], "No player impact report found.")),
        ("Key Play", get_insight(data, "Most Valuable Play")),
    ]
    render_html('<div class="section-card"><div class="eyebrow">Game Intelligence</div>' + "".join(f'<div class="intel-card"><div class="intel-title">{esc(title)}</div><div class="intel-body">{esc(short_text(text, 220))}</div></div>' for title, text in blocks) + '</div>')


def show_live_mode_panel(game_id: str) -> None:
    render_html(f'<div class="live-card"><div class="eyebrow">Live Backend MVP</div><div class="intel-body">Historical dashboard uses saved reports and CSV files. Live prediction runs through the Flask/SocketIO backend.<br><br><strong>Run:</strong> <code>python backend/app.py</code><br><strong>Endpoint:</strong> <code>/predict/{esc(game_id)}?mode=live</code></div></div>')


def add_quarter_markers(fig, max_elapsed: float) -> None:
    for x_value, label in [(12, "Q1"), (24, "Q2 / Halftime"), (36, "Q3"), (48, "Q4 / End Reg.")]:
        if max_elapsed + 0.5 >= x_value:
            fig.add_vline(x=x_value, line_dash="dash", line_color="rgba(226,232,240,.42)", line_width=1, annotation_text=label, annotation_position="top left", annotation_font_size=11, annotation_font_color="#CBD5E1")
    overtime_end, overtime_number = 53, 1
    while max_elapsed > 48.5 and overtime_end <= max_elapsed + 0.5:
        label = "OT" if overtime_number == 1 else f"{overtime_number}OT"
        fig.add_vline(x=overtime_end, line_dash="dash", line_color="rgba(226,232,240,.34)", line_width=1, annotation_text=label, annotation_position="top left", annotation_font_size=11, annotation_font_color="#CBD5E1")
        overtime_end += 5
        overtime_number += 1


def show_win_probability_chart(predictions: pd.DataFrame, home_team: str, away_team: str, champion_view: bool, chart_key: str, top_spacing_px: int = 0) -> None:
    if top_spacing_px > 0:
        render_html(f'<div style="height:{int(top_spacing_px)}px"></div>')
    st.subheader("Champion Win Probability Timeline" if champion_view else "Win Probability Timeline")
    chart_data = add_game_time_columns(predictions)
    chart_long = chart_data.melt(
        id_vars=["game_minutes_elapsed", "period", "Clock", "home_score", "away_score", "score_margin_home", "event_description"],
        value_vars=["home_win_prob_pct", "away_win_prob_pct"],
        var_name="team",
        value_name="win_probability_pct",
    )
    chart_long["team"] = chart_long["team"].replace({"home_win_prob_pct": home_team, "away_win_prob_pct": away_team})
    chart_long["Quarter"] = chart_long["period"].apply(lambda value: format_period(int(value)))
    chart_long["Score"] = chart_long.apply(lambda row: f"{away_team} {int(row['away_score'])} - {home_team} {int(row['home_score'])}", axis=1)
    chart_long["Play"] = chart_long["event_description"].fillna("No play description")
    fig = px.line(chart_long, x="game_minutes_elapsed", y="win_probability_pct", color="team", color_discrete_map={away_team: CHART_AWAY_COLOR, home_team: CHART_HOME_COLOR}, custom_data=["team", "Quarter", "Clock", "Score", "Play"], labels={"game_minutes_elapsed": "Game Time", "win_probability_pct": "Win Probability (%)"})
    fig.update_traces(line=dict(width=4), hovertemplate="<b>%{customdata[0]}</b><br>Win Probability: %{y:.1f}%<br>Quarter: %{customdata[1]}<br>Clock: %{customdata[2]}<br>Score: %{customdata[3]}<br>Play: %{customdata[4]}<extra></extra>")
    fig.update_yaxes(range=[0, 100], gridcolor="#1F2937")
    fig.update_xaxes(gridcolor="#1F2937")
    fig.add_hline(y=50, line_dash="dot", line_color="#CBD5E1", opacity=.72)
    add_quarter_markers(fig, float(chart_data["game_minutes_elapsed"].max()))
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0B1020", paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified", height=430, margin=dict(l=20, r=132, t=42, b=20), legend_title_text="", legend=dict(orientation="v", yanchor="top", y=.98, xanchor="left", x=1.02, bgcolor="rgba(15,23,42,.78)", bordercolor="#334155", borderwidth=1, font=dict(color="#E5E7EB", size=13)))
    st.plotly_chart(fig, width="stretch", key=chart_key)


def show_brand_header(game_id: str, home_team: str, away_team: str) -> None:
    subtitle = f"NBA Win Probability Platform · {away_team} at {home_team} · Game ID {game_id}" if game_id else "NBA Win Probability Platform · Historical and Live Game Center"
    render_html(f'<div class="brand-header"><div class="brand-left"><div class="brand-mark"><span class="brand-cc">CC</span></div><div><div class="brand-title">ClutchCast AI</div><div class="brand-subtitle">{esc(subtitle)}</div></div></div><div class="brand-badge">Historical Dashboard · Live Backend MVP Available</div></div>')


def build_win_probability_story(data: dict, predictions: pd.DataFrame, home_team: str, away_team: str) -> dict:
    row = predictions.iloc[-1]
    home_score, away_score = as_int(row.get("home_score")), as_int(row.get("away_score"))
    home_prob, away_prob = as_float(row.get("home_win_prob_pct"), 50), as_float(row.get("away_win_prob_pct"), 50)
    favorite = home_team if home_prob >= away_prob else away_team
    favorite_prob = max(home_prob, away_prob)
    seconds_remaining = as_float(row.get("seconds_remaining"), 0)
    turning = build_turning_points(predictions, top_n=1)
    if turning.empty:
        biggest_swing, biggest_detail = "Pending", "Run reports to identify the biggest swing."
    else:
        swing = turning.iloc[0]
        biggest_swing, biggest_detail = f"{float(swing['wp_swing_pct']):+.1f} pts", short_text(str(swing["event_description"]), 120)
    if seconds_remaining == 0:
        winner = home_team if home_score > away_score else away_team if away_score > home_score else "Neither team"
        losing_peak = predictions["away_win_prob_pct"].max() if winner == home_team else predictions["home_win_prob_pct"].max() if winner == away_team else 50.0
        lede = f"Final: {winner} closed out a {away_score}-{home_score} game. The losing side peaked at {losing_peak:.1f}% win probability."
        state = "Final"
    else:
        leader = home_team if home_score > away_score else away_team if away_score > home_score else "Neither team"
        lede = f"Live story: {leader} leads {away_team} {away_score}, {home_team} {home_score}; {favorite} owns the current model edge at {favorite_prob:.1f}%."
        state = f"{format_period(as_int(row.get('period')))}, {format_nba_clock(row.get('clock', ''))}"
    return {"lede": lede, "state": state, "favorite": f"{favorite} · {favorite_prob:.1f}%", "biggest_swing": biggest_swing, "biggest_detail": biggest_detail, "key_play": short_text(get_insight(data, "Most Valuable Play"), 140)}


def show_win_probability_story(data: dict, predictions: pd.DataFrame, home_team: str, away_team: str) -> None:
    story = build_win_probability_story(data, predictions, home_team, away_team)
    cards = [
        render_info_card("Current State", story["state"], story["favorite"], "⏱"),
        render_info_card("Biggest Swing", story["biggest_swing"], story["biggest_detail"], "📈"),
        render_info_card("Key Play", "Game Intelligence", story["key_play"], "🔥"),
    ]
    render_html(f'<div class="story-shell"><div class="story-title">Win Probability Story</div><div class="story-lede">{esc(story["lede"])}</div><div class="story-grid">' + "".join(cards) + '</div></div>')


def show_game_overview(data: dict, predictions: pd.DataFrame, game_id: str, home_team: str, away_team: str, model_label: str, champion_label: str, champion_view: bool) -> None:
    show_scoreboard(predictions, home_team, away_team, model_label, champion_label)
    show_metric_cards(data, champion_label)
    left, right = st.columns([2.15, 1], gap="large")
    with left:
        show_win_probability_chart(predictions, home_team, away_team, champion_view, chart_key="overview_win_probability_chart", top_spacing_px=22)
    with right:
        render_html('<div class="right-rail-spacer"></div>')
        show_game_intelligence_panel(data, predictions)
        show_live_mode_panel(game_id)


def show_game_insights(data: dict, game_id: str) -> None:
    st.subheader("Game Insights")
    render_html('<div class="tab-intro">Game-specific intelligence turns the probability feed into a quick broadcast-style read.</div>')
    insights = data["game_insights"]
    if insights.empty:
        show_empty_report_card("Game Insights", f"python src/game_insights.py --game-id {game_id}")
        return
    cards = [
        render_info_card("Game Drama Score", get_insight(data, "Game Drama Score", "value", "Pending"), short_text(get_insight(data, "Game Drama Score"), 180), "🏀"),
        render_info_card("Most Valuable Play", short_text(get_insight(data, "Most Valuable Play", "value", "Most Valuable Play"), 42), short_text(get_insight(data, "Most Valuable Play"), 180), "📈"),
        render_info_card("Most Damaging Play", short_text(get_insight(data, "Most Damaging Play", "value", "Most Damaging Play"), 42), short_text(get_insight(data, "Most Damaging Play"), 180), "🔥"),
        render_info_card("Clutch-Time Scoring", short_text(get_insight(data, "Clutch-Time Scoring Summary", "value", "Clutch scoring"), 42), short_text(get_insight(data, "Clutch-Time Scoring Summary"), 180), "⏱"),
    ]
    render_html('<div class="insight-grid">' + "".join(cards) + '</div>')
    with st.expander("Detailed game insights table"):
        st.dataframe(clean_table_columns(insights), width="stretch", hide_index=True)


def show_turning_points_tab(predictions: pd.DataFrame, game_id: str) -> None:
    turning = build_turning_points(predictions)
    st.subheader("Turning Points")
    render_html('<div class="tab-intro">The biggest win-probability swings reveal where the game actually bent.</div>')
    if turning.empty:
        show_empty_report_card("Turning points", f"python src/turning_points.py --game-id {game_id}")
        return
    data = turning.copy()
    data["abs_swing"] = data["wp_swing_pct"].abs()
    biggest = data.sort_values("abs_swing", ascending=False).iloc[0]
    quarter_summary = data.groupby("period")["abs_swing"].sum().sort_values(ascending=False)
    volatile_period = int(quarter_summary.index[0])
    cards = [
        render_summary_card("Biggest Swing", f"{float(biggest['wp_swing_pct']):+.1f} pts", short_text(str(biggest["event_description"]), 110), big=True),
        render_summary_card("Total Major Swings", str(int((data["abs_swing"] >= 10).sum())), "Swings of at least 10 win-probability points."),
        render_summary_card("Most Volatile Quarter", format_period(volatile_period), f"{quarter_summary.iloc[0]:.1f} combined swing points."),
        render_summary_card("Key Play", str(biggest["event_player"]), short_text(str(biggest["event_description"]), 110), avatar=initials(str(biggest["event_player"]))),
    ]
    render_html('<div class="summary-grid">' + "".join(cards) + '</div>')
    st.dataframe(clean_table_columns(turning), width="stretch", hide_index=True)


def show_player_impact_tab(predictions: pd.DataFrame, game_id: str) -> None:
    impact = build_player_impact(predictions)
    st.subheader("Player Impact")
    render_html('<div class="tab-intro">Player impact aggregates win-probability movement attached to tracked player events.</div>')
    if impact.empty:
        show_empty_report_card("Player impact", f"python src/player_impact.py --game-id {game_id}")
        return
    highest = impact.sort_values("total_absolute_swing_pct", ascending=False).iloc[0]
    volatile = impact.sort_values("avg_absolute_swing_pct", ascending=False).iloc[0]
    team_impact = impact.groupby("event_team", as_index=False)["total_absolute_swing_pct"].sum().sort_values("total_absolute_swing_pct", ascending=False).iloc[0]
    cards = [
        render_summary_card("Highest Impact Player", str(highest["event_player"]), f"{float(highest['total_absolute_swing_pct']):.1f} total swing pts · {highest['event_team']}", avatar=initials(str(highest["event_player"]))),
        render_summary_card("Most Volatile Player", str(volatile["event_player"]), f"{float(volatile['avg_absolute_swing_pct']):.2f} avg swing pts/event", avatar=initials(str(volatile["event_player"]))),
        render_summary_card("Top Team by Swing", str(team_impact["event_team"]), f"{float(team_impact['total_absolute_swing_pct']):.1f} total swing points."),
        render_summary_card("Tracked Player Events", str(int(impact["event_count"].sum())), "Events with player, team, and valid WP movement context."),
    ]
    render_html('<div class="summary-grid">' + "".join(cards) + '</div>')
    st.dataframe(clean_table_columns(impact), width="stretch", hide_index=True)


def show_pressure_comebacks_tab(predictions: pd.DataFrame, game_id: str) -> None:
    pressure = calculate_clutch_pressure(predictions).sort_values("clutch_pressure", ascending=False).head(15)
    comeback = build_comeback_report(predictions)
    st.subheader("Pressure & Comebacks")
    render_html('<div class="tab-intro">Pressure combines score closeness, game time, and win-probability uncertainty.</div>')
    if pressure.empty:
        show_empty_report_card("Pressure and comeback data", f"python src/features.py --game-id {game_id}")
        return
    peak = pressure.iloc[0]
    highest = None if comeback.empty else comeback.sort_values("comeback_probability_pct", ascending=False).iloc[0]
    biggest = None if comeback.empty else comeback.sort_values("deficit", ascending=False).iloc[0]
    cards = [
        render_summary_card("Peak Clutch Pressure", f"{float(peak['clutch_pressure']):.1f}", f"{format_period(int(peak['period']))} · {format_nba_clock(peak['clock'])} · {short_text(str(peak['event_description']), 80)}", big=True),
        render_summary_card("Highest Comeback Probability", "N/A" if highest is None else f"{float(highest['comeback_probability_pct']):.1f}%", "No comeback window found." if highest is None else f"{highest['trailing_team']} trailing by {int(highest['deficit'])}."),
        render_summary_card("Biggest Deficit With Chance", "N/A" if biggest is None else str(int(biggest["deficit"])), "No eligible comeback deficit found." if biggest is None else f"{biggest['comeback_status']} comeback window."),
        render_summary_card("Most Pressurized Moment", f"{format_period(int(peak['period']))}, {format_nba_clock(peak['clock'])}", short_text(str(peak["event_description"]), 100)),
    ]
    render_html('<div class="summary-grid">' + "".join(cards) + '</div>')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Clutch Pressure Detail")
        st.dataframe(clean_table_columns(pressure), width="stretch", hide_index=True)
    with col2:
        st.markdown("### Comeback Reality Detail")
        st.dataframe(clean_table_columns(comeback), width="stretch", hide_index=True)


def show_model_evaluation(data: dict, champion: dict, game_id: str) -> None:
    st.subheader("Model Evaluation")
    render_html('<div class="tab-intro">The champion is selected by probability quality first, not by model complexity.</div>')
    leaderboard, disagreements = data["leaderboard"], data["model_disagreements"]
    if leaderboard.empty:
        show_empty_report_card("Model leaderboard", "python src/compare_models.py --leaderboard")
    else:
        best_brier = leaderboard.sort_values("brier_score", ascending=True).iloc[0]
        best_auc = leaderboard.sort_values("roc_auc", ascending=False).iloc[0]
        cards = [
            render_summary_card("Champion Model", champion.get("model_name", "Champion unavailable"), "Selected by Brier, log loss, ROC-AUC, then accuracy."),
            render_summary_card("Best Brier Score", f"{float(best_brier['brier_score']):.4f}", str(best_brier["model_name"])),
            render_summary_card("Best ROC-AUC", f"{float(best_auc['roc_auc']):.4f}", str(best_auc["model_name"])),
            render_summary_card("Disagreement Peak", "Pending" if disagreements.empty else f"{float(disagreements['max_model_disagreement_pct'].max()):.1f} pts", f"python src/compare_models.py --game-id {game_id}"),
        ]
        render_html('<div class="summary-grid">' + "".join(cards) + '</div>')
        display = leaderboard.copy()
        display["Champion"] = display["model_key"].eq(champion.get("model_key")).map({True: "Yes", False: ""})
        st.dataframe(clean_table_columns(display), width="stretch", hide_index=True)
    if not disagreements.empty:
        st.markdown("### Biggest Model Disagreement Moments")
        st.dataframe(clean_table_columns(disagreements), width="stretch", hide_index=True)


def extract_recap_section(recap: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\s*$"
    lines = recap.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(pattern, line.strip(), flags=re.IGNORECASE):
            start = index + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return " ".join(line.strip(" -*") for line in lines[start:end] if line.strip()).strip()


def show_game_recap(data: dict, game_id: str) -> None:
    st.subheader("Game Recap")
    recap = data["recap"]
    if recap.startswith("No recap file"):
        show_empty_report_card("Game Recap", f"python src/recap.py --game-id {game_id}")
        return
    sections = {name: extract_recap_section(recap, name) for name in ["Final Result", "Biggest Turning Point", "Player Impact", "Comeback Reality", "Hidden Momentum", "Model Note"]}
    if not any(sections.values()):
        sections["Final Result"] = short_text(recap.replace("#", ""), 360)
    cards = [
        render_info_card("Final Result", "Game Result", short_text(sections.get("Final Result") or "Final result summary not found.", 220), "🏀"),
        render_info_card("Biggest Turning Point", "Key Swing", short_text(sections.get("Biggest Turning Point") or "Turning point summary not found.", 220), "📈"),
        render_info_card("Player Impact", "Top Contributor", short_text(sections.get("Player Impact") or "Player impact summary not found.", 220), "🔥"),
        render_info_card("Comeback Reality", "Pressure Read", short_text(sections.get("Comeback Reality") or "Comeback summary not found.", 220), "⏱"),
        render_info_card("Hidden Momentum", "Flow Signal", short_text(sections.get("Hidden Momentum") or "Momentum summary not found.", 220), "↔"),
        render_info_card("Model Note", "Champion Context", short_text(sections.get("Model Note") or "Model note not found.", 220), "CC"),
    ]
    render_html('<div class="recap-grid">' + "".join(cards) + '</div>')
    with st.expander("Full recap text"):
        st.markdown(recap)


def fetch_backend_json(path: str, timeout: float = 6.0) -> dict:
    url = f"{BACKEND_BASE_URL}{path}"
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {"ok": 200 <= response.status < 300, "status_code": response.status, "data": payload, "url": url}
    except HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except Exception:
            payload = {"error": str(error)}
        return {"ok": False, "status_code": error.code, "data": payload, "url": url}
    except (URLError, TimeoutError, OSError) as error:
        return {"ok": False, "status_code": None, "data": {"error": str(error)}, "url": url}


def show_backend_status(result: dict | None) -> None:
    if result and result.get("ok"):
        render_html('<span class="status-badge status-ok">Backend online</span>')
    else:
        render_html('<span class="status-badge status-bad">Backend offline</span>')
        st.warning("Backend is not running. Start it with: python backend/app.py")


def show_live_prediction(payload: dict, health: dict | None, champion_label: str) -> None:
    game_id = str(payload.get("game_id", "")).zfill(10)
    fallback_home, fallback_away = get_team_labels(game_id) if game_id and game_id != "0000000000" else ("Home", "Away")
    home_team = str(payload.get("home_team") or fallback_home)
    away_team = str(payload.get("away_team") or fallback_away)
    model_name = payload.get("model_name") or payload.get("champion", {}).get("model_name") or champion_label
    champion_name = payload.get("champion", {}).get("model_name", champion_label)
    has_play_by_play = bool(payload.get("has_play_by_play"))
    prediction_source = str(payload.get("prediction_source", "live_prediction"))
    data_source = str(payload.get("data_source", "backend"))
    render_scoreboard(
        home_team=home_team,
        away_team=away_team,
        home_score=as_int(payload.get("home_score")),
        away_score=as_int(payload.get("away_score")),
        home_prob=as_float(payload.get("home_win_prob_pct"), 50.0),
        away_prob=as_float(payload.get("away_win_prob_pct"), 50.0),
        period=as_int(payload.get("period")),
        clock=format_nba_clock(payload.get("clock", "")),
        model_label=str(model_name),
        champion_label=str(champion_name),
        eyebrow="Live Game Center",
    )
    warning = str(payload.get("warning") or "")
    fallback_reason = str(payload.get("fallback_reason") or "")
    if warning:
        st.warning(warning)
    elif not has_play_by_play or prediction_source == "scoreboard_fallback_baseline":
        st.warning("Play-by-play is not available yet, so ClutchCast is using scoreboard fallback.")
    status_text = "Backend online" if health and health.get("ok") else "Backend status unknown"
    cards = [
        render_metric_card("Last Play", short_text(payload.get("last_play", "No play available"), 42), short_text(payload.get("last_play", "No play available"), 160)),
        render_metric_card("Data Source", data_source.replace("_", " ").title(), f"Matched by: {payload.get('matched_by', 'unknown')}"),
        render_metric_card("Play-by-Play", "Available" if has_play_by_play else "Unavailable", f"Rows: {payload.get('play_by_play_rows', 0)}"),
        render_metric_card("Prediction Source", str(model_name), prediction_source.replace("_", " ").title()),
    ]
    render_html('<div class="metric-grid">' + "".join(cards) + '</div>')
    detail_cards = [
        render_metric_card("Backend Status", status_text, f"GET /predict/{esc(game_id)}?mode=live"),
        render_metric_card("Fallback Reason", short_text(fallback_reason or "Full model path is active.", 44), short_text(fallback_reason or "Champion model inference used live play-by-play.", 160)),
        render_metric_card("Champion Model", str(champion_name), "Automatically used once live play-by-play becomes available."),
    ]
    render_html('<div class="summary-grid">' + "".join(detail_cards) + '</div>')


def show_today_games_helper() -> str | None:
    selected_game_id = None
    if st.button("Load Today's Games", key="live_load_today_games"):
        with st.spinner("Fetching today's games from the live backend..."):
            result = fetch_backend_json("/games/today", timeout=10.0)

        if not result["ok"]:
            error = result.get("data", {}).get("error", "Could not load today's games.")
            st.error(error)
        else:
            st.session_state["live_today_games"] = result.get("data", {}).get("games", [])

    games = st.session_state.get("live_today_games", [])
    if not games:
        return None

    def game_label(game: dict) -> str:
        away = game.get("away_team") or "Away"
        home = game.get("home_team") or "Home"
        away_score = game.get("away_score", 0)
        home_score = game.get("home_score", 0)
        period = game.get("period", 0)
        clock = game.get("clock") or game.get("status") or ""
        source = game.get("data_source", "scoreboard")
        return f"{game.get('GAME_ID')} | {away} {away_score} at {home} {home_score} | Q{period} {clock} | {source}"

    display = pd.DataFrame(games)
    selected_index = st.selectbox(
        "Today's games from live scoreboard",
        range(len(games)),
        format_func=lambda index: game_label(games[index]),
        key="live_today_games_select",
    )
    if st.button("Use Selected GAME_ID", key="live_use_today_game"):
        selected_game_id = str(games[selected_index].get("GAME_ID", "")).zfill(10)
        st.session_state["live_game_id"] = selected_game_id
        st.success(f"Selected GAME_ID {selected_game_id}.")
    st.dataframe(clean_table_columns(display), width="stretch", hide_index=True)
    return selected_game_id


def show_live_game_tab(champion_label: str) -> None:
    st.subheader("Live Game")
    render_html('<div class="tab-intro">Streamlit polls the local Flask backend for live NBA updates. Start the backend first with <code>python backend/app.py</code>. Historical tabs use saved CSV/report files; this tab uses <code>/predict/&lt;game_id&gt;?mode=live</code>. If live play-by-play is delayed, the backend falls back to the NBA scoreboard and then switches to champion-model inference when play-by-play appears.</div>')

    col1, col2, col3 = st.columns([1.4, 1, 1])
    with col1:
        game_id = st.text_input("Live GAME_ID", value=st.session_state.get("live_game_id", ""), placeholder="Example: 0042300312", key="live_game_id_input")
        game_id = str(game_id).strip().zfill(10) if str(game_id).strip() else ""
        st.session_state["live_game_id"] = game_id
    with col2:
        auto_refresh = st.checkbox("Auto-refresh every 10 seconds", key="live_auto_refresh")
    with col3:
        st.caption("Live accuracy/update speed depends on NBA API availability and delay.")

    check_status = st.button("Check Backend Status", key="live_check_backend")
    fetch_live = st.button("Fetch Live Prediction", type="primary", key="live_fetch_prediction")
    selected_today_game = show_today_games_helper()
    if selected_today_game:
        game_id = selected_today_game

    if auto_refresh:
        components.html("<script>setTimeout(function(){ window.parent.location.reload(); }, 10000);</script>", height=0)
        st.caption("Auto-refresh is on. This page will reload every 10 seconds while the checkbox remains selected.")

    health = None
    if check_status or fetch_live or auto_refresh:
        health = fetch_backend_json("/health", timeout=2.0)
        show_backend_status(health)

    should_fetch = bool(game_id) and (fetch_live or auto_refresh)
    if should_fetch:
        with st.spinner("Fetching live prediction from backend..."):
            result = fetch_backend_json(f"/predict/{game_id}?mode=live", timeout=20.0)
        if result["ok"]:
            payload = result["data"]
            if isinstance(payload, dict) and payload.get("error"):
                st.error(payload["error"])
            else:
                st.session_state["last_live_payload"] = payload
                show_live_prediction(payload, health, champion_label)
        else:
            error = result.get("data", {}).get("error", "Backend request failed.")
            st.error(error)
            if "play-by-play" in str(error).lower() or "not" in str(error).lower():
                st.info("The game may not have started yet, or NBA API may not have play-by-play data available.")
    elif st.session_state.get("last_live_payload"):
        show_live_prediction(st.session_state["last_live_payload"], health, champion_label)
    else:
        show_empty_report_card("Live Game", "python backend/app.py")

    st.markdown("### Live Backend MVP Note")
    st.info("This is a local-first polling MVP. It does not use a paid feed and should be expected to lag or fail when nba_api is delayed or unavailable.")


def show_missing_historical_tab(command: str) -> None:
    st.info("No historical prediction files are available yet. Live Game can still be used if the backend is running.")
    st.code(command, language="powershell")


def main() -> None:
    apply_custom_css()
    champion = load_champion_metadata()
    champion_key = champion.get("model_key", "baseline")
    champion_label = champion.get("model_name", MODE_LABELS.get(champion_key, "Baseline Model"))

    available_game_ids = get_available_game_ids()
    selected_game_id = ""
    model_key = champion_key
    data = None
    predictions = pd.DataFrame()
    home_team, away_team = "Home", "Away"
    model_label = MODE_LABELS.get(model_key, model_key)
    champion_view = False

    with st.sidebar:
        st.markdown("## ClutchCast AI")
        if available_game_ids:
            selected_game_id = st.selectbox("Analyzed game", available_game_ids, index=len(available_game_ids) - 1)
            available_modes = get_available_modes(selected_game_id)
            default_mode = champion_key if champion_key in available_modes else available_modes[-1]
            st.markdown(f"**Champion Model:** {champion_label}")
            st.caption("Main dashboard defaults to the champion when its prediction file exists.")
            advanced = st.checkbox("Inspect another model")
            model_key = st.selectbox("Model view", available_modes, index=available_modes.index(default_mode), format_func=lambda key: MODE_LABELS[key]) if advanced else default_mode
        else:
            st.warning("No historical games found yet.")
            st.code("python src/run_pipeline.py --game-id YOUR_GAME_ID --model neural", language="powershell")
        st.divider()
        st.markdown("### Live Backend MVP")
        st.caption("Live Game polls Flask/SocketIO backend.")
        st.code("python backend/app.py", language="powershell")

    if selected_game_id:
        data = load_dashboard_data(selected_game_id, model_key)
        predictions = data["predictions"]
        home_team, away_team = get_team_labels(selected_game_id)
        model_label = MODE_LABELS.get(model_key, model_key)
        champion_view = model_key == champion_key

    show_brand_header(selected_game_id, home_team, away_team)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Game Overview", "Win Probability", "Live Game", "Game Insights", "Turning Points", "Player Impact", "Pressure & Comebacks", "Model Evaluation", "Game Recap"
    ])

    missing_command = "python src/run_pipeline.py --game-id YOUR_GAME_ID --model neural"
    with tab1:
        show_game_overview(data, predictions, selected_game_id, home_team, away_team, model_label, champion_label, champion_view) if data is not None else show_missing_historical_tab(missing_command)
    with tab2:
        if data is not None:
            show_win_probability_chart(predictions, home_team, away_team, champion_view, chart_key="full_win_probability_chart")
            show_win_probability_story(data, predictions, home_team, away_team)
        else:
            show_missing_historical_tab(missing_command)
    with tab3:
        show_live_game_tab(champion_label)
    with tab4:
        show_game_insights(data, selected_game_id) if data is not None else show_missing_historical_tab("python src/game_insights.py --game-id YOUR_GAME_ID")
    with tab5:
        show_turning_points_tab(predictions, selected_game_id) if data is not None else show_missing_historical_tab("python src/turning_points.py --game-id YOUR_GAME_ID")
    with tab6:
        show_player_impact_tab(predictions, selected_game_id) if data is not None else show_missing_historical_tab("python src/player_impact.py --game-id YOUR_GAME_ID")
    with tab7:
        show_pressure_comebacks_tab(predictions, selected_game_id) if data is not None else show_missing_historical_tab(missing_command)
    with tab8:
        show_model_evaluation(data, champion, selected_game_id) if data is not None else show_missing_historical_tab("python src/compare_models.py --leaderboard")
    with tab9:
        show_game_recap(data, selected_game_id) if data is not None else show_missing_historical_tab("python src/recap.py --game-id YOUR_GAME_ID")


if __name__ == "__main__":
    main()
