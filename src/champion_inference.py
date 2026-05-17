from pathlib import Path
import json

import joblib
import pandas as pd
import torch

from ml_pipeline_utils import apply_terminal_state_overrides, load_feature_columns
from model_features import build_model_features
from train_baseline import baseline_home_win_probability
from train_neural_network import WinProbabilityNeuralNetwork


MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODEL_CONFIG = {
    "baseline": {
        "name": "Baseline Rule Model",
        "prediction_file_prefix": "baseline_predictions",
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "prediction_file_prefix": "ml_predictions",
        "model_path": MODELS_DIR / "win_probability_model.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "prediction_file_prefix": "advanced_predictions",
        "model_path": MODELS_DIR / "advanced_win_probability_model.joblib",
    },
    "pytorch_neural_network": {
        "name": "PyTorch Neural Network",
        "prediction_file_prefix": "neural_predictions",
        "model_path": MODELS_DIR / "pytorch_win_probability_model.pt",
        "scaler_path": MODELS_DIR / "pytorch_scaler.joblib",
    },
}


def load_champion_metadata() -> dict:
    champion_path = REPORTS_DIR / "champion_model.json"

    if not champion_path.exists():
        return {
            "model_key": "baseline",
            "model_name": MODEL_CONFIG["baseline"]["name"],
            "status": "missing_champion_report",
        }

    metadata = json.loads(champion_path.read_text(encoding="utf-8"))
    metadata.setdefault("status", "ready")
    return metadata


def get_prediction_file_prefix(model_key: str) -> str:
    return MODEL_CONFIG.get(model_key, MODEL_CONFIG["baseline"])["prediction_file_prefix"]


def load_model_bundle(model_key: str | None = None) -> dict:
    metadata = load_champion_metadata()
    selected_model_key = model_key or metadata.get("model_key", "baseline")

    if selected_model_key not in MODEL_CONFIG:
        selected_model_key = "baseline"

    bundle = {
        "model_key": selected_model_key,
        "model_name": MODEL_CONFIG[selected_model_key]["name"],
        "metadata": metadata,
        "feature_columns": [],
        "model": None,
        "scaler": None,
    }

    if selected_model_key == "baseline":
        return bundle

    feature_columns = load_feature_columns()
    bundle["feature_columns"] = feature_columns

    if selected_model_key in {"logistic_regression", "random_forest"}:
        model_path = MODEL_CONFIG[selected_model_key]["model_path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")
        bundle["model"] = joblib.load(model_path)
        return bundle

    model_path = MODEL_CONFIG[selected_model_key]["model_path"]
    scaler_path = MODEL_CONFIG[selected_model_key]["scaler_path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Missing PyTorch model artifact: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing PyTorch scaler artifact: {scaler_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    input_size = checkpoint.get("input_size", len(feature_columns))

    model = WinProbabilityNeuralNetwork(input_size=input_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    bundle["model"] = model
    bundle["scaler"] = joblib.load(scaler_path)
    return bundle


def predict_game_state(game_state: pd.DataFrame, model_key: str | None = None) -> pd.DataFrame:
    bundle = load_model_bundle(model_key)
    output = game_state.copy()

    if bundle["model_key"] == "baseline":
        output["home_win_prob"] = output.apply(baseline_home_win_probability, axis=1)
    elif bundle["model_key"] in {"logistic_regression", "random_forest"}:
        model_ready_data = build_model_features(output)
        X = model_ready_data[bundle["feature_columns"]]
        output["home_win_prob"] = bundle["model"].predict_proba(X)[:, 1]
    else:
        model_ready_data = build_model_features(output)
        X = model_ready_data[bundle["feature_columns"]].astype(float)
        X_scaled = bundle["scaler"].transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            output["home_win_prob"] = bundle["model"](X_tensor).numpy().flatten()

    output = apply_terminal_state_overrides(output)
    output["prediction_source"] = bundle["model_key"]
    output["prediction_model_name"] = bundle["model_name"]

    return output


def latest_prediction_payload(predictions: pd.DataFrame) -> dict:
    row = predictions.iloc[-1]

    return {
        "game_id": str(row.get("game_id", "")).zfill(10),
        "period": int(row.get("period", 0)),
        "clock": str(row.get("clock", "")),
        "home_score": int(row.get("home_score", 0)),
        "away_score": int(row.get("away_score", 0)),
        "home_win_prob": float(row.get("home_win_prob", 0.5)),
        "away_win_prob": float(row.get("away_win_prob", 0.5)),
        "home_win_prob_pct": float(row.get("home_win_prob_pct", 50.0)),
        "away_win_prob_pct": float(row.get("away_win_prob_pct", 50.0)),
        "last_play": str(row.get("event_description", "")),
        "model_key": str(row.get("prediction_source", "")),
        "model_name": str(row.get("prediction_model_name", "")),
    }
