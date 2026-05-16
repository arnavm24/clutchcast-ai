from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from nba_api.stats.endpoints import boxscoresummaryv2


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

HOME_COLOR = "#3B82F6"  # premium blue
AWAY_COLOR = "#EF4444"  # premium red


st.set_page_config(
    page_title="ClutchCast AI",
    page_icon="🏀",
    layout="wide",
)


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #111827 0%, #080B12 40%, #05070D 100%);
            color: #F9FAFB;
        }

        [data-testid="stSidebar"] {
            background-color: #070A12;
            border-right: 1px solid #1F2937;
        }

        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #9CA3AF;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .premium-card {
            background: rgba(17, 24, 39, 0.92);
            border: 1px solid #1F2937;
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.25);
            margin-bottom: 1rem;
        }

        .team-pill-home {
            display: inline-block;
            background: rgba(59, 130, 246, 0.15);
            color: #93C5FD;
            border: 1px solid rgba(59, 130, 246, 0.35);
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-weight: 700;
        }

        .team-pill-away {
            display: inline-block;
            background: rgba(239, 68, 68, 0.15);
            color: #FCA5A5;
            border: 1px solid rgba(239, 68, 68, 0.35);
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.9);
            border: 1px solid #1F2937;
            padding: 1rem;
            border-radius: 16px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
        }

        div[data-testid="stMetricLabel"] {
            color: #9CA3AF;
        }

        div[data-testid="stMetricValue"] {
            color: #F9FAFB;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(17, 24, 39, 0.8);
            border: 1px solid #1F2937;
            border-radius: 999px;
            padding: 0.5rem 1rem;
            color: #D1D5DB;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #2563EB, #7C3AED);
            color: white;
            border: 1px solid transparent;
        }

        h1, h2, h3 {
            letter-spacing: -0.03em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_latest_csv(folder: Path, pattern: str) -> pd.DataFrame:
    files = list(folder.glob(pattern))

    if not files:
        st.error(f"Missing required file: `{folder / pattern}`")
        st.info("Run the project pipeline scripts first, then refresh the dashboard.")
        st.stop()

    return pd.read_csv(files[0], dtype={"game_id": str})


def load_latest_text(folder: Path, pattern: str) -> str:
    files = list(folder.glob(pattern))

    if not files:
        return "No recap file found. Run `python src/recap.py` first."

    return files[0].read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def get_team_labels(game_id: str) -> tuple[str, str]:
    try:
        summary = boxscoresummaryv2.BoxScoreSummaryV2(
            game_id=game_id,
            timeout=30,
        )

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


@st.cache_data
def load_dashboard_data():
    predictions = load_latest_csv(PROCESSED_DIR, "baseline_predictions_*.csv")
    features = load_latest_csv(PROCESSED_DIR, "features_*.csv")
    momentum = load_latest_csv(PROCESSED_DIR, "momentum_*.csv")

    turning_points = load_latest_csv(REPORTS_DIR, "turning_points_*.csv")
    player_impact = load_latest_csv(REPORTS_DIR, "player_impact_*.csv")
    comeback_report = load_latest_csv(REPORTS_DIR, "comeback_report_*.csv")
    momentum_report = load_latest_csv(REPORTS_DIR, "momentum_report_*.csv")

    recap = load_latest_text(REPORTS_DIR, "post_game_recap_*.md")

    return {
        "predictions": predictions,
        "features": features,
        "momentum": momentum,
        "turning_points": turning_points,
        "player_impact": player_impact,
        "comeback_report": comeback_report,
        "momentum_report": momentum_report,
        "recap": recap,
    }


def add_game_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    total_game_seconds = 48 * 60
    output["game_seconds_elapsed"] = total_game_seconds - output["seconds_remaining"]
    output["game_minutes_elapsed"] = output["game_seconds_elapsed"] / 60
    return output


def format_nba_clock(clock_value) -> str:
    """
    Converts NBA API clock format like PT04M19.00S into 4:19.
    """
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
        seconds_part = clock.replace("S", "")
        seconds = int(float(seconds_part))

    return f"{minutes}:{seconds:02d}"


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
        "wp_before_pct": "Win Prob. Before",
        "wp_after_pct": "Win Prob. After",
        "wp_swing_pct": "Win Prob. Swing",
        "rank": "Rank",
        "total_raw_home_wp_swing_pct": "Net Home Win Prob. Swing",
        "total_absolute_swing_pct": "Total Win Prob. Impact",
        "avg_absolute_swing_pct": "Average Impact Per Event",
        "event_count": "Events Tracked",
        "home_win_prob_pct": "Home Win Probability",
        "away_win_prob_pct": "Away Win Probability",
        "clutch_pressure": "Clutch Pressure",
        "pressure_level": "Pressure Level",
        "trailing_team": "Trailing Team",
        "deficit": "Deficit",
        "comeback_probability_pct": "Comeback Probability",
        "comeback_status": "Comeback Status",
        "required_points_per_minute": "Required Points/Min",
        "recent_margin_change": "Recent Margin Change",
        "recent_wp_change_pct": "Recent Win Prob. Change",
        "recent_event_value": "Recent Event Value",
        "hidden_momentum_score": "Momentum Score",
        "momentum_label": "Momentum Label",
    }

    return display.rename(columns=rename_map)


def show_header(home_team: str, away_team: str) -> None:
    st.markdown(
        f"""
        <div class="main-title">ClutchCast AI</div>
        <div class="subtitle">
            Premium NBA win probability, turning-point, momentum, and game-story engine.
        </div>
        <div style="margin-bottom: 1rem;">
            <span class="team-pill-away">{away_team}</span>
            <span style="color:#6B7280; margin: 0 0.5rem;">at</span>
            <span class="team-pill-home">{home_team}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_game_summary(
    predictions: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> None:
    final_row = predictions.iloc[-1]

    game_id = str(final_row["game_id"]).zfill(10)
    home_score = int(final_row["home_score"])
    away_score = int(final_row["away_score"])
    margin = int(final_row["score_margin_home"])
    home_wp = float(final_row["home_win_prob_pct"])
    away_wp = float(final_row["away_win_prob_pct"])

    if margin > 0:
        result = f"{home_team} by {margin}"
    elif margin < 0:
        result = f"{away_team} by {abs(margin)}"
    else:
        result = "Tie"

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Final Score", f"{home_team} {home_score} - {away_team} {away_score}")
    col2.metric("Result", result)
    col3.metric(f"{home_team} Final Win Prob.", f"{home_wp:.1f}%")
    col4.metric("Events Tracked", f"{len(predictions)}")

    st.caption(f"Game ID: {game_id} · {away_team} final win probability: {away_wp:.1f}%")


def show_win_probability_chart(
    predictions: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> None:
    st.subheader("Win Probability Timeline")

    chart_data = add_game_time_columns(predictions)
    chart_data["readable_clock"] = chart_data["clock"].apply(format_nba_clock)

    chart_data_long = chart_data.melt(
        id_vars=[
            "game_minutes_elapsed",
            "period",
            "readable_clock",
            "home_score",
            "away_score",
            "score_margin_home",
            "event_description",
        ],
        value_vars=["home_win_prob_pct", "away_win_prob_pct"],
        var_name="team",
        value_name="win_probability_pct",
    )

    chart_data_long["team"] = chart_data_long["team"].replace(
        {
            "home_win_prob_pct": home_team,
            "away_win_prob_pct": away_team,
        }
    )

    fig = px.line(
        chart_data_long,
        x="game_minutes_elapsed",
        y="win_probability_pct",
        color="team",
        color_discrete_map={
            home_team: HOME_COLOR,
            away_team: AWAY_COLOR,
        },
        hover_data=[
            "period",
            "readable_clock",
            "home_score",
            "away_score",
            "score_margin_home",
            "event_description",
        ],
        labels={
            "game_minutes_elapsed": "Game Time",
            "win_probability_pct": "Win Probability (%)",
            "team": "Team",
            "period": "Quarter",
            "readable_clock": "Clock",
            "home_score": "Home Score",
            "away_score": "Away Score",
            "score_margin_home": "Home Margin",
            "event_description": "Play",
        },
        title=f"{away_team} at {home_team} · Win Probability",
    )

    fig.update_traces(line=dict(width=3))

    fig.update_yaxes(
        range=[0, 100],
        gridcolor="#1F2937",
        zeroline=False,
    )

    fig.update_xaxes(
        gridcolor="#1F2937",
        zeroline=False,
    )

    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="#9CA3AF",
        opacity=0.8,
        annotation_text="50%",
        annotation_position="right",
    )

    for minute, label in [(12, "End Q1"), (24, "Half"), (36, "End Q3"), (48, "Final")]:
        fig.add_vline(
            x=minute,
            line_dash="dash",
            line_color="#6B7280",
            opacity=0.45,
            annotation_text=label,
            annotation_position="top",
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B1020",
        hovermode="x unified",
        legend_title_text="",
        margin=dict(l=20, r=20, t=60, b=20),
        height=430,
    )

    st.plotly_chart(fig, width="stretch")


def show_turning_points(turning_points: pd.DataFrame) -> None:
    st.subheader("Turning Points")
    st.caption("The plays that created the largest win-probability swings.")

    display = clean_table_columns(turning_points)

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
    )


def show_player_impact(
    player_impact: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> None:
    st.subheader("Player Impact")
    st.caption("Players ranked by how much their events moved win probability.")

    top_10 = player_impact.head(10)

    fig = px.bar(
        top_10,
        x="event_player",
        y="total_absolute_swing_pct",
        color="event_team",
        color_discrete_map={
            home_team: HOME_COLOR,
            away_team: AWAY_COLOR,
        },
        hover_data=["event_team", "event_count", "avg_absolute_swing_pct"],
        labels={
            "event_player": "Player",
            "total_absolute_swing_pct": "Total Win Probability Impact",
            "event_team": "Team",
            "event_count": "Events Tracked",
            "avg_absolute_swing_pct": "Average Impact Per Event",
        },
        title="Top Player Win Probability Impact",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B1020",
        margin=dict(l=20, r=20, t=60, b=20),
        height=420,
        legend_title_text="Team",
    )

    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        clean_table_columns(player_impact),
        width="stretch",
        hide_index=True,
    )


def show_clutch_pressure(features: pd.DataFrame) -> None:
    st.subheader("Clutch Pressure")
    st.caption("High-pressure moments based on score closeness, time, and win-probability uncertainty.")

    features_with_time = add_game_time_columns(features)
    features_with_time["readable_clock"] = features_with_time["clock"].apply(format_nba_clock)

    fig = px.scatter(
        features_with_time,
        x="game_minutes_elapsed",
        y="clutch_pressure",
        color="pressure_level",
        size="clutch_pressure",
        hover_data=[
            "period",
            "readable_clock",
            "home_score",
            "away_score",
            "score_margin_home",
            "home_win_prob_pct",
            "event_description",
        ],
        labels={
            "game_minutes_elapsed": "Game Time",
            "clutch_pressure": "Clutch Pressure",
            "pressure_level": "Pressure Level",
            "period": "Quarter",
            "readable_clock": "Clock",
            "home_score": "Home Score",
            "away_score": "Away Score",
            "score_margin_home": "Home Margin",
            "home_win_prob_pct": "Home Win Probability",
            "event_description": "Play",
        },
        title="Clutch Pressure Timeline",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0B1020",
        margin=dict(l=20, r=20, t=60, b=20),
        height=430,
    )

    st.plotly_chart(fig, width="stretch")

    columns = [
        "period",
        "clock",
        "home_score",
        "away_score",
        "score_margin_home",
        "home_win_prob_pct",
        "clutch_pressure",
        "pressure_level",
        "event_description",
    ]

    available_columns = [col for col in columns if col in features.columns]

    display = features.sort_values("clutch_pressure", ascending=False)[available_columns].head(15)

    st.dataframe(
        clean_table_columns(display),
        width="stretch",
        hide_index=True,
    )


def show_comeback_meter(comeback_report: pd.DataFrame) -> None:
    st.subheader("Comeback Reality")
    st.caption("Moments where the trailing team had a comeback scenario worth analyzing.")

    st.dataframe(
        clean_table_columns(comeback_report),
        width="stretch",
        hide_index=True,
    )


def show_hidden_momentum(momentum_report: pd.DataFrame) -> None:
    st.subheader("Hidden Momentum")
    st.caption("Recent-flow score based on score margin changes, win-probability movement, and event value.")

    st.dataframe(
        clean_table_columns(momentum_report),
        width="stretch",
        hide_index=True,
    )


def show_recap(
    recap: str,
    predictions: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> None:
    st.subheader("Auto Game Recap")

    final_row = predictions.iloc[-1]
    home_score = int(final_row["home_score"])
    away_score = int(final_row["away_score"])

    col1, col2 = st.columns([1, 3])

    with col1:
        st.metric(home_team, home_score)
        st.metric(away_team, away_score)

    with col2:
        word_count = len(recap.split())
        read_time = max(1, round(word_count / 200))
        st.caption(f"{word_count} words · ~{read_time} min read")
        st.markdown(
            f"""
            <div class="premium-card">
            {recap}
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    apply_custom_css()

    data = load_dashboard_data()

    predictions = data["predictions"]
    features = data["features"]
    turning_points = data["turning_points"]
    player_impact = data["player_impact"]
    comeback_report = data["comeback_report"]
    momentum_report = data["momentum_report"]
    recap = data["recap"]

    game_id = str(predictions["game_id"].iloc[0]).zfill(10)
    home_team, away_team = get_team_labels(game_id)

    with st.sidebar:
        st.markdown("## 🏀 ClutchCast AI")
        st.caption("Premium NBA game intelligence engine")
        st.divider()
        st.markdown(f"**Matchup:** {away_team} at {home_team}")
        st.markdown(f"**Game ID:** `{game_id}`")
        st.divider()
        st.caption("V1 rule-based analytics engine")
        st.caption("Next upgrade: trained ML model")

    show_header(home_team, away_team)
    show_game_summary(predictions, home_team, away_team)
    show_win_probability_chart(predictions, home_team, away_team)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Turning Points",
            "Player Impact",
            "Clutch Pressure",
            "Comeback Reality",
            "Hidden Momentum",
            "Game Recap",
        ]
    )

    with tab1:
        show_turning_points(turning_points)

    with tab2:
        show_player_impact(player_impact, home_team, away_team)

    with tab3:
        show_clutch_pressure(features)

    with tab4:
        show_comeback_meter(comeback_report)

    with tab5:
        show_hidden_momentum(momentum_report)

    with tab6:
        show_recap(recap, predictions, home_team, away_team)


if __name__ == "__main__":
    main()