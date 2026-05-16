import argparse

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


def find_close_games(
    season: str,
    season_type: str,
    max_margin: int,
    top_n: int,
) -> pd.DataFrame:
    print(f"Searching {season} {season_type} games...")

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable=season_type,
        timeout=30,
    )

    games = finder.get_data_frames()[0]

    if games.empty:
        raise ValueError("No games found.")

    # Each game appears twice, once from each team perspective.
    # Keep both rows temporarily so we can infer matchup and score.
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)

    rows = []

    for game_id, group in games.groupby("GAME_ID"):
        if len(group) < 2:
            continue

        team_a = group.iloc[0]
        team_b = group.iloc[1]

        pts_a = int(team_a["PTS"])
        pts_b = int(team_b["PTS"])
        margin = abs(pts_a - pts_b)

        if margin <= max_margin:
            rows.append(
                {
                    "game_id": game_id,
                    "game_date": team_a["GAME_DATE"],
                    "team_1": team_a["TEAM_ABBREVIATION"],
                    "team_1_pts": pts_a,
                    "team_2": team_b["TEAM_ABBREVIATION"],
                    "team_2_pts": pts_b,
                    "margin": margin,
                    "matchup": team_a["MATCHUP"],
                }
            )

    close_games = pd.DataFrame(rows)

    if close_games.empty:
        return close_games

    close_games = close_games.sort_values(
        ["margin", "game_date"],
        ascending=[True, False],
    )

    return close_games.head(top_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find close NBA games for ClutchCast AI demos."
    )

    parser.add_argument(
        "--season",
        type=str,
        default="2023-24",
        help="NBA season, example: 2023-24.",
    )

    parser.add_argument(
        "--season-type",
        type=str,
        default="Regular Season",
        choices=["Regular Season", "Playoffs", "Pre Season", "All Star"],
        help="Season type.",
    )

    parser.add_argument(
        "--max-margin",
        type=int,
        default=3,
        help="Maximum final score margin.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of close games to show.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    close_games = find_close_games(
        season=args.season,
        season_type=args.season_type,
        max_margin=args.max_margin,
        top_n=args.top_n,
    )

    if close_games.empty:
        print("No close games found. Try increasing --max-margin.")
        return

    print("\nClose games found:")
    print(close_games.to_string(index=False))

    print("\nExample command:")
    first_game_id = close_games.iloc[0]["game_id"]
    print(f"python src/run_pipeline.py --game-id {first_game_id}")


if __name__ == "__main__":
    main()