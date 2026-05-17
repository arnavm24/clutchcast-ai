from pathlib import Path

import copy

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ml_pipeline_utils import (
    TARGET_COLUMN,
    apply_terminal_state_overrides,
    compute_probability_metrics,
    load_shared_training_inputs,
)


MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class WinProbabilityNeuralNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def split_training_games_for_validation(train_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_games = sorted(train_data["game_id"].unique())

    if len(train_games) < 5:
        return train_data.copy(), train_data.copy()

    fit_games, validation_games = train_test_split(
        train_games,
        test_size=0.20,
        random_state=42,
    )

    fit_data = train_data[train_data["game_id"].isin(fit_games)].copy()
    validation_data = train_data[train_data["game_id"].isin(validation_games)].copy()

    return fit_data, validation_data


def prepare_tensors(
    fit_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[TensorDataset, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    X_fit = fit_data[feature_columns].astype(float)
    y_fit = fit_data[TARGET_COLUMN].astype(float)

    X_validation = validation_data[feature_columns].astype(float)
    y_validation = validation_data[TARGET_COLUMN].astype(float)

    X_test = test_data[feature_columns].astype(float)

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    fit_dataset = TensorDataset(
        torch.tensor(X_fit_scaled, dtype=torch.float32),
        torch.tensor(y_fit.values, dtype=torch.float32).view(-1, 1),
    )

    return (
        fit_dataset,
        torch.tensor(X_validation_scaled, dtype=torch.float32),
        torch.tensor(y_validation.values, dtype=torch.float32).view(-1, 1),
        torch.tensor(X_test_scaled, dtype=torch.float32),
        scaler,
    )


def calculate_loss(model: WinProbabilityNeuralNetwork, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    loss_function = nn.BCELoss()

    with torch.no_grad():
        probabilities = model(X)
        return float(loss_function(probabilities, y).item())


def train_neural_network(
    fit_dataset: TensorDataset,
    X_validation: torch.Tensor,
    y_validation: torch.Tensor,
    input_size: int,
    epochs: int = 150,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    patience: int = 12,
) -> tuple[WinProbabilityNeuralNetwork, dict]:
    model = WinProbabilityNeuralNetwork(input_size=input_size)

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0001,
    )

    train_loader = DataLoader(
        fit_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nTraining neural network with validation early stopping...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        validation_loss = calculate_loss(model, X_validation, y_validation)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train loss: {train_loss:.4f} | Validation loss: {validation_loss:.4f}"
            )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    model.load_state_dict(best_state)

    history = {
        "best_epoch": best_epoch,
        "best_validation_loss": round(best_validation_loss, 4),
    }

    return model, history


def evaluate_model(
    model: WinProbabilityNeuralNetwork,
    X_test: torch.Tensor,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    model.eval()

    with torch.no_grad():
        probabilities = model(X_test).numpy().flatten()

    prediction_frame = test_data.copy()
    prediction_frame["home_win_prob"] = probabilities
    prediction_frame = apply_terminal_state_overrides(prediction_frame)

    return compute_probability_metrics(
        y_true=prediction_frame[TARGET_COLUMN],
        probabilities=prediction_frame["home_win_prob"],
        model_key="pytorch_neural_network",
        model_name="PyTorch Neural Network",
        feature_count=len(feature_columns),
        train_data=train_data,
        test_data=test_data,
    )


def save_model_and_scaler(
    model: WinProbabilityNeuralNetwork,
    scaler: StandardScaler,
    feature_columns: list[str],
    training_history: dict,
) -> None:
    model_path = MODELS_DIR / "pytorch_win_probability_model.pt"
    scaler_path = MODELS_DIR / "pytorch_scaler.joblib"
    feature_path = MODELS_DIR / "pytorch_model_features.txt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(feature_columns),
            "feature_columns": feature_columns,
            "training_history": training_history,
        },
        model_path,
    )

    joblib.dump(scaler, scaler_path)
    feature_path.write_text("\n".join(feature_columns), encoding="utf-8")

    print(f"Saved PyTorch model to: {model_path}")
    print(f"Saved PyTorch scaler to: {scaler_path}")
    print(f"Saved PyTorch feature list to: {feature_path}")


def save_metrics(metrics: dict) -> None:
    output_path = REPORTS_DIR / "pytorch_model_metrics.csv"
    pd.DataFrame([metrics]).to_csv(output_path, index=False)

    print(f"Saved PyTorch model metrics to: {output_path}")
    print("\nPyTorch model metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    train_data, test_data, feature_columns, train_game_ids, test_game_ids = (
        load_shared_training_inputs()
    )
    fit_data, validation_data = split_training_games_for_validation(train_data)

    print("\nDataset summary:")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Train rows: {len(train_data)}")
    print(f"Train games: {len(train_game_ids)}")
    print(f"Validation games: {validation_data['game_id'].nunique()}")
    print(f"Test rows: {len(test_data)}")
    print(f"Test games: {len(test_game_ids)}")

    fit_dataset, X_validation, y_validation, X_test, scaler = prepare_tensors(
        fit_data=fit_data,
        validation_data=validation_data,
        test_data=test_data,
        feature_columns=feature_columns,
    )

    model, training_history = train_neural_network(
        fit_dataset=fit_dataset,
        X_validation=X_validation,
        y_validation=y_validation,
        input_size=len(feature_columns),
    )

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        train_data=train_data,
        test_data=test_data,
        feature_columns=feature_columns,
    )
    metrics.update(training_history)

    save_model_and_scaler(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        training_history=training_history,
    )
    save_metrics(metrics)

    print("\nSuccess.")
    print("Retrained PyTorch neural network using shared train/test games and validation early stopping.")


if __name__ == "__main__":
    main()
