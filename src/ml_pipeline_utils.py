from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

MODEL_DATASET_PATH = PROCESSED_DIR / "model_training_dataset.csv"
FEATURE_COLUMNS_PATH = PROCESSED_DIR / "model_feature_columns.txt"
TRAIN_GAME_IDS_PATH = PROCESSED_DIR / "train_game_ids.txt"
TEST_GAME_IDS_PATH = PROCESSED_DIR / "test_game_ids.txt"

TARGET_COLUMN = "home_won"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_model_training_dataset() -> pd.DataFrame:
    if not MODEL_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Missing model training dataset. Run:\n"
            "python src/model_features.py"
        )

    data = pd.read_csv(MODEL_DATASET_PATH, dtype={"game_id": str})

    if data.empty:
        raise ValueError("Model training dataset is empty.")

    data["game_id"] = data["game_id"].astype(str).str.zfill(10)
    return data


def load_feature_columns() -> list[str]:
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "Missing model feature list. Run:\n"
            "python src/model_features.py"
        )

    feature_columns = [
        line.strip()
        for line in FEATURE_COLUMNS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not feature_columns:
        raise ValueError("Feature column list is empty.")

    return feature_columns


def validate_training_inputs(data: pd.DataFrame, feature_columns: list[str]) -> None:
    required_columns = feature_columns + [TARGET_COLUMN, "game_id"]
    missing = [column for column in required_columns if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "Training data only has one target class. "
            "Build a larger dataset with both home wins and home losses."
        )


def read_game_ids(path: Path) -> list[str]:
    return [
        line.strip().zfill(10)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_game_ids(path: Path, game_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(game_ids)) + "\n", encoding="utf-8")


def validate_game_split(
    data: pd.DataFrame,
    train_game_ids: list[str],
    test_game_ids: list[str],
) -> None:
    train_games = set(train_game_ids)
    test_games = set(test_game_ids)
    overlap = train_games.intersection(test_games)

    if overlap:
        raise ValueError(f"Train/test game split overlaps: {sorted(overlap)}")

    dataset_games = set(data["game_id"].astype(str).str.zfill(10).unique())
    missing = (train_games | test_games) - dataset_games

    if missing:
        raise ValueError(
            "Saved train/test split contains game IDs not in the dataset: "
            f"{sorted(missing)}"
        )

    if not train_games or not test_games:
        raise ValueError("Train/test split must include at least one game in each set.")


def create_game_split(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    unique_games = sorted(data["game_id"].astype(str).str.zfill(10).unique())

    if len(unique_games) < 5:
        raise ValueError(
            "Need at least 5 games for a useful train/test split. "
            "Build a larger training dataset first."
        )

    train_game_ids, test_game_ids = train_test_split(
        unique_games,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_game_ids = sorted(train_game_ids)
    test_game_ids = sorted(test_game_ids)

    write_game_ids(TRAIN_GAME_IDS_PATH, train_game_ids)
    write_game_ids(TEST_GAME_IDS_PATH, test_game_ids)

    return train_game_ids, test_game_ids


def get_or_create_game_split(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    if TRAIN_GAME_IDS_PATH.exists() and TEST_GAME_IDS_PATH.exists():
        train_game_ids = read_game_ids(TRAIN_GAME_IDS_PATH)
        test_game_ids = read_game_ids(TEST_GAME_IDS_PATH)
    else:
        train_game_ids, test_game_ids = create_game_split(data)

    validate_game_split(data, train_game_ids, test_game_ids)
    return train_game_ids, test_game_ids


def split_rows_by_game_ids(
    data: pd.DataFrame,
    train_game_ids: list[str],
    test_game_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_games = set(train_game_ids)
    test_games = set(test_game_ids)

    train_data = data[data["game_id"].isin(train_games)].copy()
    test_data = data[data["game_id"].isin(test_games)].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Train/test split produced an empty dataframe.")

    row_overlap = set(train_data["game_id"]).intersection(set(test_data["game_id"]))
    if row_overlap:
        raise ValueError(f"Row-level leakage detected by game_id: {sorted(row_overlap)}")

    return train_data, test_data


def load_shared_training_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    feature_columns = load_feature_columns()
    data = load_model_training_dataset()
    validate_training_inputs(data, feature_columns)

    train_game_ids, test_game_ids = get_or_create_game_split(data)
    train_data, test_data = split_rows_by_game_ids(
        data=data,
        train_game_ids=train_game_ids,
        test_game_ids=test_game_ids,
    )

    return train_data, test_data, feature_columns, train_game_ids, test_game_ids


def compute_probability_metrics(
    y_true,
    probabilities,
    model_key: str,
    model_name: str,
    feature_count: int | None = None,
    train_data: pd.DataFrame | None = None,
    test_data: pd.DataFrame | None = None,
) -> dict:
    y_true = pd.Series(y_true).astype(int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    predicted_labels = (probabilities >= 0.5).astype(int)

    metrics = {
        "model_key": model_key,
        "model_name": model_name,
        "dataset": MODEL_DATASET_PATH.name,
        "feature_count": feature_count,
        "train_rows": len(train_data) if train_data is not None else None,
        "train_games": train_data["game_id"].nunique() if train_data is not None else None,
        "test_rows": len(test_data) if test_data is not None else len(y_true),
        "test_games": test_data["game_id"].nunique() if test_data is not None and "game_id" in test_data else None,
        "brier_score": round(brier_score_loss(y_true, probabilities), 4),
        "log_loss": round(log_loss(y_true, probabilities, labels=[0, 1]), 4),
        "accuracy": round(accuracy_score(y_true, predicted_labels), 4),
    }

    if y_true.nunique() == 2:
        metrics["roc_auc"] = round(roc_auc_score(y_true, probabilities), 4)
    else:
        metrics["roc_auc"] = None

    return metrics


def apply_terminal_state_overrides(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()

    required_columns = [
        "seconds_remaining",
        "home_score",
        "away_score",
        "home_win_prob",
    ]
    missing = [column for column in required_columns if column not in output.columns]

    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    terminal_mask = output["seconds_remaining"] == 0
    home_wins = terminal_mask & (output["home_score"] > output["away_score"])
    away_wins = terminal_mask & (output["home_score"] < output["away_score"])
    ties = terminal_mask & (output["home_score"] == output["away_score"])

    output.loc[home_wins, "home_win_prob"] = 1.0
    output.loc[away_wins, "home_win_prob"] = 0.0
    output.loc[ties, "home_win_prob"] = 0.5

    output["home_win_prob"] = output["home_win_prob"].astype(float).clip(0, 1)
    output["away_win_prob"] = 1 - output["home_win_prob"]
    output["home_win_prob_pct"] = (output["home_win_prob"] * 100).round(1)
    output["away_win_prob_pct"] = (output["away_win_prob"] * 100).round(1)

    if "game_id" in output.columns:
        output["wp_change"] = (
            output.groupby("game_id")["home_win_prob"].diff().fillna(0)
        )
    else:
        output["wp_change"] = output["home_win_prob"].diff().fillna(0)

    output["abs_wp_change"] = output["wp_change"].abs()

    return output


def rank_leaderboard(leaderboard: pd.DataFrame) -> pd.DataFrame:
    output = leaderboard.copy()
    output["roc_auc_sort"] = output["roc_auc"].fillna(-1)

    output = output.sort_values(
        by=["brier_score", "log_loss", "roc_auc_sort", "accuracy"],
        ascending=[True, True, False, False],
    ).drop(columns=["roc_auc_sort"])

    output.insert(0, "rank", range(1, len(output) + 1))
    return output.reset_index(drop=True)
