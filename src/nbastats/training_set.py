from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("nbastats")


def build_training_set(
    weighted_stats_dir: str | Path,
    out_path: str | Path,
    start_year: int = 1992,
    end_year: int = 2019,
    min_games_history: int = 20,
) -> Path:
    """Build an ML-ready training dataset from per-game weighted stats.

    For each game, we compute features based on the *previous* N games (default 20) for
    each team and store the feature difference (home - away). Label is score differential.

    Outputs:
      - out_path (pickle)
    """
    weighted_stats_dir = Path(weighted_stats_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for year in range(start_year, end_year + 1):
        # Backwards compatibility: accept either *_sumStatsByGame.pkl or *_weightedStatsByGame.pkl
        p1 = weighted_stats_dir / f"{year}_sumStatsByGame.pkl"
        p2 = weighted_stats_dir / f"{year}_weightedStatsByGame.pkl"
        in_pkl = p1 if p1.exists() else p2
        if not in_pkl.exists():
            logger.warning("Missing weighted stats: %s", in_pkl)
            continue

        logger.info("Building training rows for %d from %s", year, in_pkl.name)
        df = pd.read_pickle(in_pkl)
        games = df.groupby("gameIdx")

        for game_idx, game in games:
            home = game[game["home"] == True]
            away = game[game["home"] == False]
            if home.empty or away.empty:
                continue

            # Use index ordering to define "games to date" consistently
            home_index = home.index[0]
            away_index = away.index[0]
            home_team = home["homeTeam"].unique()[0]
            away_team = away["awayTeam"].unique()[0]

            home_games = df[(df["homeTeam"] == home_team) | (df["awayTeam"] == home_team)]
            away_games = df[(df["homeTeam"] == away_team) | (df["awayTeam"] == away_team)]

            min_index = min(home_index, away_index)
            home_to_date = home_games.loc[:min_index - 1, :]
            away_to_date = away_games.loc[:min_index - 1, :]

            num_home_games = len(home_to_date) / 2.0
            if num_home_games < min_games_history:
                continue

            # Select last N games for each team (as home or away)
            home_last = home_to_date[
                ((home_to_date["homeTeam"] == home_team) & (home_to_date["home"] == True)) |
                ((home_to_date["awayTeam"] == home_team) & (home_to_date["home"] == False))
            ].tail(min_games_history)

            away_last = away_to_date[
                ((away_to_date["homeTeam"] == away_team) & (away_to_date["home"] == True)) |
                ((away_to_date["awayTeam"] == away_team) & (away_to_date["home"] == False))
            ].tail(min_games_history)

            drop_cols = ["PTS", "awayTeam", "gameIdx", "home", "homeTeam"]
            home_stats = home_last.drop(columns=[c for c in drop_cols if c in home_last.columns]).mean(numeric_only=True)
            away_stats = away_last.drop(columns=[c for c in drop_cols if c in away_last.columns]).mean(numeric_only=True)

            game_data = (home_stats - away_stats)
            game_data["score_diff"] = float(home["PTS"].values[0] - away["PTS"].values[0])
            game_data["awayTeam"] = away_team
            game_data["homeTeam"] = home_team
            game_data["gameIdx"] = int(game_idx)
            game_data["year"] = int(year)

            rows.append(game_data)

    training_df = pd.DataFrame(rows)
    training_df.to_pickle(out_path)
    logger.info("Wrote training set: %s (rows=%d, cols=%d)", out_path, len(training_df), training_df.shape[1])
    return out_path
