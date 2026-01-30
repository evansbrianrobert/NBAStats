from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("nbastats")

STAT_COLS = [
    "MP","FG","FGA","FG%","3P","3PA","FT","FTA","FT%","ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS",
    "TS%","eFG%","3PAr","FTr","ORB%","DRB%","TRB%","AST%","STL%","BLK%","TOV%","USG%","ORtg","DRtg"
]


def _team_stats(stats: pd.DataFrame) -> pd.Series:
    """Compute weighted team metrics for a team within a single game."""
    team_stats = pd.Series(dtype=float)

    # Protect against empty stats (rare scrape failures)
    if stats.empty:
        return team_stats

    team_stats["eFG%"] = (stats["FG"].sum() + stats["3P"].sum()/2.0) / max(stats["FGA"].sum(), 1.0)
    team_stats["DRtg"] = np.nansum(stats["DRtg"] * stats["MP"] / 2880.0) / 5.0
    team_stats["ORtg"] = np.nansum(stats["ORtg"] * stats["MP"] / 2880.0) / 5.0
    team_stats["TOV%"] = np.nansum(stats["TOV%"] * stats["MP"] / 2880.0) / 5.0

    team_stats["BLK%"] = np.nansum(stats["BLK%"] * stats["MP"] / 2880.0)
    team_stats["ORB%"] = np.nansum(stats["ORB%"] * stats["MP"] / 2880.0)
    team_stats["DRB%"] = np.nansum(stats["DRB%"] * stats["MP"] / 2880.0)
    team_stats["TRB%"] = np.nansum(stats["TRB%"] * stats["MP"] / 2880.0)
    team_stats["AST%"] = np.nansum(stats["AST%"] * stats["MP"] / 2880.0)
    team_stats["STL%"] = np.nansum(stats["STL%"] * stats["MP"] / 2880.0)

    team_stats["FTr"] = stats["FTA"].sum() / max(stats["FGA"].sum(), 1.0)
    team_stats["3PAr"] = stats["3PA"].sum() / max(stats["FGA"].sum(), 1.0)
    team_stats["TS%"] = stats["PTS"].sum() / max(2.0 * (stats["FGA"].sum() + 0.44 * stats["FTA"].sum()), 1.0)

    # Free Throws can be strings in historical data; coerce carefully.
    ft = pd.to_numeric(stats["FT"], errors="coerce").fillna(0).sum()
    fta = pd.to_numeric(stats["FTA"], errors="coerce").fillna(0).sum()
    team_stats["FT%"] = ft / fta if fta else np.nan

    team_stats["PTS"] = stats["PTS"].sum()
    return team_stats


def game_stats_sum(df_game: pd.DataFrame) -> pd.DataFrame:
    """Compute home/away aggregated stats for a single game."""
    home = df_game[df_game["home"] == True][STAT_COLS]
    away = df_game[df_game["home"] == False][STAT_COLS]

    home_team = df_game[df_game["home"] == True]["Team"].unique()[0]
    away_team = df_game[df_game["home"] == False]["Team"].unique()[0]
    game_idx = int(df_game["gameIdx"].unique()[0])

    home_stats = _team_stats(home)
    home_stats["gameIdx"] = game_idx
    home_stats["homeTeam"] = home_team
    home_stats["awayTeam"] = away_team
    home_stats["home"] = True

    away_stats = _team_stats(away)
    away_stats["gameIdx"] = game_idx
    away_stats["homeTeam"] = home_team
    away_stats["awayTeam"] = away_team
    away_stats["home"] = False

    return pd.DataFrame([home_stats, away_stats])


def build_weighted_stats_by_year(
    master_by_year_dir: str | Path,
    out_dir: str | Path,
    start_year: int = 2008,
    end_year: int = 2019,
    overwrite: bool = False,
) -> None:
    """Build per-year game-level weighted stats from master tables."""
    master_by_year_dir = Path(master_by_year_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year + 1):
        in_pkl = master_by_year_dir / f"{year}_master.pkl"
        if not in_pkl.exists():
            logger.warning("Missing master file: %s", in_pkl)
            continue

        out_pkl = out_dir / f"{year}_weightedStatsByGame.pkl"
        if out_pkl.exists() and not overwrite:
            logger.info("Skipping existing: %s", out_pkl)
            continue

        logger.info("Building weighted stats for %d", year)
        df = pd.read_pickle(in_pkl)

        rows = []
        for game_idx in sorted(df["gameIdx"].unique()):
            rows.append(game_stats_sum(df[df["gameIdx"] == game_idx]))

        game_sum_stats = pd.concat(rows, ignore_index=True)
        game_sum_stats.to_pickle(out_pkl)
        logger.info("Wrote: %s (rows=%d)", out_pkl, len(game_sum_stats))
