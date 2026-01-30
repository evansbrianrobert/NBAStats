from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("nbastats")


def combine_boxscores(boxscores_dir: str | Path, out_path: str | Path) -> Path:
    """Combine year-level boxscore pickles into a single 'AllYears.pkl'."""
    boxscores_dir = Path(boxscores_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pkls = sorted(boxscores_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No year pkls found in {boxscores_dir}")

    df = pd.concat([pd.read_pickle(p) for p in pkls], ignore_index=True)
    df.to_pickle(out_path)
    logger.info("Wrote combined boxscores: %s (rows=%d)", out_path, len(df))
    return out_path


def _build_row(game_idx: int, player: pd.Series, game_player_list: List[pd.DataFrame], home: bool) -> pd.Series:
    tmp = pd.Series(dtype=object)
    tmp["gameIdx"] = game_idx
    tmp["playerIdx"] = player.get("playerIndex", np.nan)

    if home:
        tmp["home"] = True
        tmp["team"] = [ii for ii in game_player_list[0].playerIndex.values]
        tmp["opponent"] = [ii for ii in game_player_list[1].playerIndex.values]
    else:
        tmp["home"] = False
        tmp["team"] = [ii for ii in game_player_list[1].playerIndex.values]
        tmp["opponent"] = [ii for ii in game_player_list[0].playerIndex.values]

    return pd.concat([tmp, player], axis=0)


def _player_data(player: pd.Series, game: pd.Series, home: bool, all_player_data: pd.DataFrame) -> pd.Series:
    if home:
        team = game.Home
        advanced = game.Home_Advanced[game.Home_Advanced.Starters == player.Starters].iloc[:, 2:].iloc[0]
    else:
        team = game.Away
        advanced = game.Away_Advanced[game.Away_Advanced.Starters == player.Starters].iloc[:, 2:].iloc[0]

    name = player.Starters
    # Fix known encoding issue observed in historical data.
    if name == "Peja StojakoviÄ":
        name = "Peja Stojaković"

    info = all_player_data[(all_player_data.Year == game.Season) & (all_player_data.Team == team) & (all_player_data.Name == name)]
    if info.empty:
        info = all_player_data[(all_player_data.Year == game.Season) & (all_player_data.Name == name)]

    if info.empty:
        raise KeyError(f"Could not match player info for name={name} year={game.Season} team={team}")

    info_row = info.iloc[0]
    basic = player.iloc[1:]
    return pd.concat([info_row, basic, advanced], axis=0)


def _home_player_data(row: pd.Series, all_player_data: pd.DataFrame) -> pd.DataFrame:
    box_score = row.Home_Basic.dropna(subset=["FG"])
    return box_score.apply(lambda x: _player_data(x, row, home=True, all_player_data=all_player_data), axis=1)


def _away_player_data(row: pd.Series, all_player_data: pd.DataFrame) -> pd.DataFrame:
    box_score = row.Away_Basic.dropna(subset=["FG"])
    return box_score.apply(lambda x: _player_data(x, row, home=False, all_player_data=all_player_data), axis=1)


def _players_in_game(row: pd.Series, all_player_data: pd.DataFrame) -> List[pd.DataFrame]:
    home = _home_player_data(row, all_player_data)
    away = _away_player_data(row, all_player_data)
    return [home, away]


def build_master_by_year(
    all_years_pkl: str | Path,
    player_data_pkl: str | Path,
    out_dir: str | Path,
    overwrite: bool = False,
) -> None:
    """Build per-year master tables from boxscores + player metadata.

    Inputs:
      - all_years_pkl: combined boxscore dataframe (see combine_boxscores)
      - player_data_pkl: player metadata (height, position, etc.)

    Outputs:
      - {out_dir}/{YEAR}_master.pkl
    """
    all_years_pkl = Path(all_years_pkl)
    player_data_pkl = Path(player_data_pkl)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(all_years_pkl)
    all_player_data = pd.read_pickle(player_data_pkl)

    for year in sorted(df.Season.unique()):
        out_path = out_dir / f"{int(year)}_master.pkl"
        if out_path.exists() and not overwrite:
            logger.info("Skipping existing: %s", out_path)
            continue

        logger.info("Building master for %s", int(year))
        year_df = df[df.Season == year]

        all_players_in_game = year_df.apply(lambda r: _players_in_game(r, all_player_data), axis=1)

        master_rows = []
        for game_idx, game_player_list in enumerate(all_players_in_game):
            home_df, away_df = game_player_list[0], game_player_list[1]
            master_rows.append(home_df.apply(lambda x: _build_row(game_idx, x, game_player_list, True), axis=1))
            master_rows.append(away_df.apply(lambda x: _build_row(game_idx, x, game_player_list, False), axis=1))

            if game_idx % 100 == 0:
                logger.info("...game %d / %d", game_idx, len(all_players_in_game))

        master = pd.concat(master_rows, ignore_index=True)
        master.to_pickle(out_path)
        logger.info("Wrote: %s (rows=%d)", out_path, len(master))
