from pathlib import Path
import sys
import time

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from champion_inference import latest_prediction_payload, load_champion_metadata, predict_game_state
from game_state import build_game_state
from live_data_provider import get_live_game_snapshot, get_today_games
from load_data import fetch_play_by_play


RAW_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


def load_or_fetch_game_state(game_id: str, force_fetch: bool = False) -> pd.DataFrame:
    game_id = str(game_id).zfill(10)
    game_state_path = PROCESSED_DIR / f"game_state_{game_id}.csv"
    raw_path = RAW_DIR / f"play_by_play_{game_id}.csv"

    if game_state_path.exists() and not force_fetch:
        return pd.read_csv(game_state_path, dtype={"game_id": str})

    pbp_df = fetch_play_by_play(game_id)
    pbp_df.to_csv(raw_path, index=False)

    game_state = build_game_state(raw_path)
    game_state.to_csv(game_state_path, index=False)
    return game_state


def build_game_state_from_live_play_by_play(game_id: str, play_by_play: pd.DataFrame) -> pd.DataFrame:
    game_id = str(game_id).zfill(10)
    raw_path = RAW_DIR / f"play_by_play_{game_id}.csv"
    game_state_path = PROCESSED_DIR / f"game_state_{game_id}.csv"

    play_by_play.to_csv(raw_path, index=False)
    game_state = build_game_state(raw_path)
    game_state.to_csv(game_state_path, index=False)
    return game_state


def scoreboard_baseline_home_probability(snapshot: dict) -> float:
    home_score = float(snapshot.get("home_score") or 0)
    away_score = float(snapshot.get("away_score") or 0)
    margin = home_score - away_score
    status = str(snapshot.get("status") or "").lower()

    if "final" in status:
        if margin > 0:
            return 1.0
        if margin < 0:
            return 0.0
        return 0.5

    period = int(snapshot.get("period") or 0)
    leverage = 0.025
    if period >= 4:
        leverage = 0.04
    elif period >= 2:
        leverage = 0.03

    probability = 0.5 + margin * leverage
    return max(0.02, min(0.98, probability))


def public_snapshot(snapshot: dict) -> dict:
    return {key: value for key, value in snapshot.items() if not key.startswith("_")}


def live_predict_payload(game_id: str) -> dict:
    snapshot = get_live_game_snapshot(game_id)
    champion = load_champion_metadata()

    if snapshot.get("has_play_by_play") and isinstance(snapshot.get("_play_by_play_df"), pd.DataFrame):
        game_state = build_game_state_from_live_play_by_play(game_id, snapshot["_play_by_play_df"])
        predictions = predict_game_state(game_state)
        payload = latest_prediction_payload(predictions)
        payload.update(public_snapshot(snapshot))
        payload["mode"] = "live"
        payload["champion"] = champion
        payload["prediction_source"] = "champion_model_live_play_by_play"
        payload["model_name"] = payload.get("model_name") or champion.get("model_name", "Champion Model")
        payload["play_by_play_rows"] = int(snapshot.get("play_by_play_rows") or len(snapshot["_play_by_play_df"]))
        payload["fallback_reason"] = ""
        payload["warning"] = ""
        return payload

    home_win_prob = scoreboard_baseline_home_probability(snapshot)
    away_win_prob = 1.0 - home_win_prob
    payload = public_snapshot(snapshot)
    payload.update(
        {
            "mode": "live",
            "champion": champion,
            "prediction_source": "scoreboard_fallback_baseline",
            "model_key": "scoreboard_fallback_baseline",
            "model_name": "Scoreboard Fallback Baseline",
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "home_win_prob_pct": round(home_win_prob * 100, 1),
            "away_win_prob_pct": round(away_win_prob * 100, 1),
            "play_by_play_rows": int(snapshot.get("play_by_play_rows") or 0),
            "fallback_reason": snapshot.get("fallback_reason") or "Live play-by-play is unavailable; using scoreboard fallback.",
            "warning": snapshot.get("warning") or "Full champion model could not run because live play-by-play is unavailable.",
        }
    )
    return payload


def predict_payload(game_id: str, mode: str) -> dict:
    if mode == "live":
        return live_predict_payload(game_id)

    game_state = load_or_fetch_game_state(game_id, force_fetch=False)
    predictions = predict_game_state(game_state)
    payload = latest_prediction_payload(predictions)
    payload["mode"] = mode
    payload["champion"] = load_champion_metadata()
    payload["prediction_source"] = "champion_model_historical_game_state"
    return payload


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "clutchcast-ai-backend"})


@app.get("/games/today")
def games_today():
    try:
        games = get_today_games()
        return jsonify({"count": len(games), "games": games})
    except Exception as error:
        return jsonify({"error": str(error), "games": [], "count": 0}), 502


@app.get("/predict/<game_id>")
def predict(game_id: str):
    mode = request.args.get("mode", "historical").lower()

    if mode not in {"historical", "live"}:
        return jsonify({"error": "mode must be 'historical' or 'live'"}), 400

    try:
        return jsonify(predict_payload(game_id, mode))
    except Exception as error:
        return jsonify({"error": str(error), "game_id": str(game_id).zfill(10), "mode": mode}), 502


@socketio.on("subscribe_game")
def subscribe_game(message):
    game_id = str(message.get("game_id", "")).zfill(10)
    mode = str(message.get("mode", "live")).lower()
    updates = int(message.get("updates", 60))

    if not game_id or game_id == "0000000000":
        emit("prediction_error", {"error": "Missing game_id"})
        return

    if mode not in {"historical", "live"}:
        emit("prediction_error", {"error": "mode must be 'historical' or 'live'"})
        return

    for _ in range(max(1, updates)):
        try:
            emit("prediction_update", predict_payload(game_id, mode))
        except Exception as error:
            emit("prediction_error", {"error": str(error), "game_id": game_id, "mode": mode})

        if mode == "historical":
            break

        time.sleep(10)


if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000, debug=True)
