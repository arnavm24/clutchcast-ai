from pathlib import Path
import argparse

import joblib
import pandas as pd
import torch
import torch.nn as nn


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


FEATURE_COLUMNS = [
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


class WinProbabilityNeuralNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def load_model_and_scaler():
    model_path = MODELS_DIR / "pytorch_win_probability_model.pt"
    scaler_path = MODELS_DIR / "pytorch_scaler.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "No PyTorch model found. Run:\n"
            "python src/train_neural_network.py"
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            "No PyTorch scaler found. Run:\n"
            "python src/train_neural_network.py"
        )

    checkpoint = torch.load(model_path, map_location="cpu")

    input_size = checkpoint.get("input_size", len(FEATURE_COLUMNS))

    model = WinProbabilityNeuralNetwork(input_size=input_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path)

    print(f"Loaded PyTorch model from: {model_path}")
    print(f"Loaded PyTorch scaler from: {scaler_path}")

    return model, scaler


def load_game_state(game_id: str | None = None) -> pd.DataFrame:
    if game_id:
        game_id = str(game_id).zfill(10)
        input_path = PROCESSED_DIR / f"game_state_{game_id}.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"No game-state file found for game {game_id}. Run:\n"
                f"python src/run_pipeline.py --game-id {game_id} --model neural"
            )

        print(f"Loading game-state file: {input_path}")
        return pd.read_csv(input_path, dtype={"game_id": str})

    files = list(PROCESSED_DIR.glob("game_state_*.csv"))

    if not files:
        raise FileNotFoundError(
            "No game-state file found. Run:\n"
            "python src/game_state.py"
        )

    input_path = files[-1]
    print(f"Loading game-state file: {input_path}")

    return pd.read_csv(input_path, dtype={"game_id": str})


def validate_features(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")


def add_neural_predictions(
    game_state: pd.DataFrame,
    model: WinProbabilityNeuralNetwork,
    scaler,
) -> pd.DataFrame:
    output = game_state.copy()

    validate_features(output)

    X = output[FEATURE_COLUMNS].astype(float)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        home_win_prob = model(X_tensor).numpy().flatten()

    output["home_win_prob"] = home_win_prob
    output["away_win_prob"] = 1 - output["home_win_prob"]

    output["home_win_prob_pct"] = (output["home_win_prob"] * 100).round(1)
    output["away_win_prob_pct"] = (output["away_win_prob"] * 100).round(1)

    output["wp_change"] = output["home_win_prob"].diff().fillna(0)
    output["abs_wp_change"] = output["wp_change"].abs()

    output["prediction_source"] = "pytorch_neural_network"

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PyTorch neural network win probability predictions."
    )

    parser.add_argument(
        "--game-id",
        type=str,
        default=None,
        help="Specific NBA game ID to predict, example: 0042300312.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, scaler = load_model_and_scaler()
    game_state = load_game_state(args.game_id)

    predictions = add_neural_predictions(
        game_state=game_state,
        model=model,
        scaler=scaler,
    )

    game_id = str(predictions["game_id"].iloc[0]).zfill(10)
    output_path = PROCESSED_DIR / f"neural_predictions_{game_id}.csv"

    predictions.to_csv(output_path, index=False)

    print("\nSuccess.")
    print(f"Saved neural network predictions to: {output_path}")
    print(f"Rows: {len(predictions)}")

    print("\nSample predictions:")
    print(
        predictions[
            [
                "period",
                "clock",
                "home_score",
                "away_score",
                "score_margin_home",
                "home_win_prob_pct",
                "away_win_prob_pct",
                "prediction_source",
            ]
        ].head(10)
    )

    print("\nFinal row:")
    print(
        predictions[
            [
                "period",
                "clock",
                "home_score",
                "away_score",
                "score_margin_home",
                "home_win_prob_pct",
                "away_win_prob_pct",
                "home_won",
            ]
        ].tail(1)
    )


if __name__ == "__main__":
    main()