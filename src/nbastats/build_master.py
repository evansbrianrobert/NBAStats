from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _player_data(
    player: pd.Series,
    game: pd.Series,
    home: bool,
    by_team_name: pd.DataFrame,
    by_name: pd.DataFrame,
) -> pd.Series:
    if home:
        team = game.Home
        adv_df = game.Home_Advanced
    else:
        team = game.Away
        adv_df = game.Away_Advanced

    name = player.Starters
    if name == "Peja StojakoviÄ":
        name = "Peja Stojaković"

    # Advanced row lookup
    advanced = adv_df[adv_df.Starters == player.Starters].iloc[:, 2:].iloc[0]

    # Player metadata lookup (FAST)
    try:
        info = by_team_name.loc[(team, name)]
        info_row = info.iloc[0] if isinstance(info, pd.DataFrame) else info
    except KeyError:
        try:
            info = by_name.loc[name]
            info_row = info.iloc[0] if isinstance(info, pd.DataFrame) else info
        except KeyError:
            raise KeyError(f"Could not match player info for name={name} year={game.Season} team={team}")

    basic = player.iloc[1:]
    return pd.concat([info_row, basic, advanced], axis=0)

def _home_player_data(row: pd.Series, by_team_name: pd.DataFrame, by_name: pd.DataFrame) -> pd.DataFrame:
    box_score = row.Home_Basic.dropna(subset=["FG"])
    return box_score.apply(lambda x: _player_data(x, row, True, by_team_name, by_name), axis=1)

def _away_player_data(row: pd.Series, by_team_name: pd.DataFrame, by_name: pd.DataFrame) -> pd.DataFrame:
    box_score = row.Away_Basic.dropna(subset=["FG"])
    return box_score.apply(lambda x: _player_data(x, row, False, by_team_name, by_name), axis=1)

def _players_in_game(row: pd.Series, by_team_name: pd.DataFrame, by_name: pd.DataFrame) -> List[pd.DataFrame]:
    return [
        _home_player_data(row, by_team_name, by_name),
        _away_player_data(row, by_team_name, by_name),
    ]

def _build_player_lookup(all_player_data_year: pd.DataFrame):
    # Fast lookup by (Team, Name), fallback by Name only
    by_team_name = all_player_data_year.set_index(["Team", "Name"], drop=False)
    by_name = all_player_data_year.set_index(["Name"], drop=False)
    return by_team_name, by_name

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

    logger.info("Loading boxscores: %s", all_years_pkl)
    df = pd.read_pickle(all_years_pkl)

    logger.info("Loading player metadata: %s", player_data_pkl)
    all_player_data = pd.read_pickle(player_data_pkl)

    years = sorted(df.Season.unique())
    logger.info("Found %d seasons: %s", len(years), [int(y) for y in years])

    for year in years:
        year_int = int(year)
        out_path = out_dir / f"{year_int}_master.pkl"

        if out_path.exists() and not overwrite:
            logger.info("Skipping existing: %s", out_path)
            continue

        logger.info("Building master for %d", year_int)

        # Slice year boxscores
        year_df = df[df.Season == year]

        # Slice player metadata to the same year (shrinks lookup table a lot)
        # Your playerData Year column is float (e.g., 1992.0), so compare on float.
        apd_year = all_player_data[all_player_data.Year == float(year)]
        if apd_year.empty:
            logger.warning("No player metadata found for year=%s (float=%s)", year, float(year))

        by_team_name, by_name = _build_player_lookup(apd_year)

        # Build players-in-game (home/away) for each row in the year df.
        # NOTE: This assumes _players_in_game(row, by_team_name, by_name) exists.
        all_players_in_game = year_df.apply(
            lambda r: _players_in_game(r, by_team_name, by_name),
            axis=1,
        )

        master_rows = []
        n_games = len(all_players_in_game)

        for game_idx, game_player_list in enumerate(all_players_in_game):
            home_df, away_df = game_player_list[0], game_player_list[1]

            # Small speed tweak: compute these once per game, not per player row.
            # (Requires _build_row to accept precomputed lists OR you keep your current _build_row.)
            # If you keep your current _build_row signature, ignore this comment.

            master_rows.append(
                home_df.apply(lambda x: _build_row(game_idx, x, game_player_list, True), axis=1)
            )
            master_rows.append(
                away_df.apply(lambda x: _build_row(game_idx, x, game_player_list, False), axis=1)
            )

            if game_idx % 100 == 0:
                logger.info("...game %d / %d", game_idx, n_games)

        master = pd.concat(master_rows, ignore_index=True)
        master.to_pickle(out_path)
        logger.info("Wrote: %s (rows=%d)", out_path, len(master))
