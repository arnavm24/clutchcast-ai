# ClutchCast AI

ClutchCast AI is an NBA win-probability and game-intelligence platform. It turns NBA play-by-play data into game-state rows, trains multiple probability models fairly, selects a Champion Model by proper probability metrics, and presents the result in a polished Streamlit game-center dashboard.

The project supports two workflows:

- Past/completed game analysis using historical NBA play-by-play data in the Streamlit dashboard.
- Local-first live prediction updates through a Flask + WebSocket MVP that polls `nba_api` and runs the Champion Model.

## Why It Exists

Most sports prediction demos stop at a cool chart. ClutchCast AI is designed to be more technically honest: every model trains on the same data, uses the same features, evaluates on the same held-out games, and competes on probability-quality metrics instead of vibes.

## Tech Stack

- `nba_api` for NBA play-by-play and game metadata
- `pandas` for state building, feature engineering, and reports
- `scikit-learn` for logistic regression, random forest, scaling, and metrics
- `PyTorch` for a tabular neural-network benchmark
- `Streamlit` and `Plotly` for the historical dashboard
- `Flask` and `Flask-SocketIO` for the local live prediction MVP

## Modeling Approach

ClutchCast AI compares four approaches:

1. Baseline rule model: interpretable formula based on score margin, time, and home-court edge.
2. Logistic regression: scaled, interpretable ML benchmark.
3. Random forest: nonlinear tabular model with feature importance.
4. PyTorch neural network: simple tabular MLP with validation early stopping.

All ML models use:

- `data/processed/model_training_dataset.csv`
- `data/processed/model_feature_columns.txt`
- `data/processed/train_game_ids.txt`
- `data/processed/test_game_ids.txt`

The split is by `game_id`, not by row, so events from the same game cannot leak between train and test.

## Champion Model Selection

The Champion Model is selected by `src/compare_models.py --leaderboard` using this ranking order:

1. Lowest Brier score
2. Lowest log loss
3. Highest ROC-AUC
4. Highest accuracy

Outputs:

- `reports/model_leaderboard.csv`
- `reports/champion_model.json`

The Streamlit dashboard defaults to the champion model when its prediction file exists. Other models remain available in the technical Model Evaluation tab.

## Feature Engineering

`src/model_features.py` is the single source of truth for training and inference features. Current features include:

- Time context: remaining/elapsed fractions, quarter indicators, second half, final minutes, overtime.
- Score context: margin, lead flags, possession-size flags, blowout flag, margin-time interactions.
- Event context: shots, makes, misses, threes, free throws, turnovers, rebounds, steals, blocks, fouls, timeouts, substitutions.
- Recent flow: rolling score-margin changes, total-score changes, event value, and home-perspective event value.
- Conservative event direction: event-by-home/away is assigned only when the current event changes the score, avoiding fragile future-derived team inference.

No raw text columns, identifiers, final result leakage, or future score changes are included as model features.

## Dashboard Features

The Streamlit dashboard is a historical game-center experience. It includes:

- Game Overview with hero scoreboard, team win probabilities, model status, and top intelligence cards
- Champion Win Probability Timeline
- Game Insights: drama score, most valuable play, most damaging play, and clutch-time scoring
- Turning Points
- Player Impact
- Clutch Pressure and Comeback Reality
- Game Recap
- Model Evaluation with leaderboard, Brier score, log loss, ROC-AUC, accuracy, final model probabilities, and disagreement moments

Live prediction is not embedded directly into Streamlit yet. The live MVP runs through the Flask/SocketIO backend.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Rebuild Dataset And Features

```powershell
python src/build_training_dataset.py --season 2023-24 --season-type "Regular Season" --max-games 300
python src/model_features.py
```

## Train Models

```powershell
python src/train_model.py
python src/train_advanced_model.py
python src/train_neural_network.py
```

## Select Champion Model

```powershell
python src/compare_models.py --leaderboard
```

## Analyze A Past Game

Replace `YOUR_GAME_ID` with an NBA game ID.

```powershell
python src/run_pipeline.py --game-id YOUR_GAME_ID --model baseline
python src/run_pipeline.py --game-id YOUR_GAME_ID --model ml
python src/run_pipeline.py --game-id YOUR_GAME_ID --model advanced
python src/run_pipeline.py --game-id YOUR_GAME_ID --model neural
python src/compare_models.py --game-id YOUR_GAME_ID
python src/game_insights.py --game-id YOUR_GAME_ID
python src/recap.py --game-id YOUR_GAME_ID
streamlit run app/streamlit_app.py
```

## Live Backend MVP

The backend is intentionally local-first and polling-based. It is not a production sports-data service. Streamlit remains the historical dashboard; live updates are served by Flask/SocketIO.

Run it:

```powershell
python backend/app.py
```

Health check:

```powershell
curl.exe http://127.0.0.1:5000/health
```

Historical prediction endpoint:

```powershell
curl.exe "http://127.0.0.1:5000/predict/YOUR_GAME_ID?mode=historical"
```

Live polling prediction endpoint:

```powershell
curl.exe "http://127.0.0.1:5000/predict/YOUR_GAME_ID?mode=live"
```

WebSocket event:

- Event name: `subscribe_game`
- Payload: `{ "game_id": "YOUR_GAME_ID", "mode": "live" }`
- Emits: `prediction_update`

## Future Live Architecture

```text
nba_api polling -> pandas state builder -> champion model -> Flask API -> WebSocket -> live dashboard
```

The backend already shares the Champion Model inference path with historical analysis through `src/champion_inference.py`.

## Current Limitations

- Accuracy depends heavily on training data size and season coverage.
- `nba_api` is unofficial and can be slow or temporarily unavailable.
- The Streamlit app is historical-first; live mode is available through the backend MVP.
- Possession, lineup, team-strength, rest, injuries, and betting-market context are future improvements.
- Team logos use the public NBA static logo path when the abbreviation is recognized, and fall back gracefully when unavailable.
- The neural network is a benchmark, not assumed to be best. The champion is whatever wins on probability metrics.

## Generated Artifacts

Generated data, model binaries, predictions, and reports are intentionally ignored:

- `data/`
- `models/`
- `reports/`

Do not commit datasets, trained models, prediction CSVs, or leaderboard/report outputs.

## Final Command Checklist

```powershell
python src/build_training_dataset.py --season 2023-24 --season-type "Regular Season" --max-games 300
python src/model_features.py
python src/train_model.py
python src/train_advanced_model.py
python src/train_neural_network.py
python src/compare_models.py --leaderboard
python src/run_pipeline.py --game-id YOUR_GAME_ID --model baseline
python src/run_pipeline.py --game-id YOUR_GAME_ID --model ml
python src/run_pipeline.py --game-id YOUR_GAME_ID --model advanced
python src/run_pipeline.py --game-id YOUR_GAME_ID --model neural
python src/compare_models.py --game-id YOUR_GAME_ID
python src/game_insights.py --game-id YOUR_GAME_ID
python src/recap.py --game-id YOUR_GAME_ID
streamlit run app/streamlit_app.py
python backend/app.py
```
