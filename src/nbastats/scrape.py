from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .br_utils import get_soup, get_dates_of_games

logger = logging.getLogger("nbastats")


def _convert_mp_to_seconds(mp: str) -> float:
    try:
        mins, secs = map(int, mp.split(":"))
        return float(mins * 60 + secs)
    except Exception:
        return 0.0


def _convert_basic_row(row: pd.Series) -> pd.Series:
    row = row.copy()
    row["MP"] = _convert_mp_to_seconds(str(row.get("MP", "0:0")))
    for col in ["FG","FGA","FG%","3P","3PA","FTA","FT%","ORB","DRB","TRB","AST","STL","BLK","TOV","PF","PTS"]:
        if col in row:
            try:
                row[col] = float(row[col])
            except Exception:
                row[col] = np.nan
    # 'FT' can be a string sometimes
    if "FT" in row:
        try:
            row["FT"] = float(row["FT"])
        except Exception:
            row["FT"] = np.nan
    return row


def _convert_adv_row(row: pd.Series) -> pd.Series:
    row = row.copy()
    row["MP"] = _convert_mp_to_seconds(str(row.get("MP", "0:0")))
    for col in ["TS%","eFG%","3PAr","FTr","ORB%","DRB%","TRB%","AST%","STL%","BLK%","TOV%","USG%","ORtg","DRtg"]:
        if col in row:
            try:
                row[col] = float(row[col])
            except Exception:
                row[col] = np.nan
    return row


def _edit_table(table: pd.DataFrame, idx: int) -> pd.DataFrame:
    """Drop subtotal rows and coerce types."""
    basic = (idx % 2) == 0
    length = 20 if basic else 16

    # Remove "Reserves" header row (typically row 5) and "Team Totals" last row.
    table_edit = pd.concat([table.iloc[:5, :length], table.iloc[6:-1, :length]], axis=0)

    if basic:
        table_edit = table_edit.apply(_convert_basic_row, axis=1)
    else:
        table_edit = table_edit.apply(_convert_adv_row, axis=1)
    return table_edit


def _get_team_names_from_links(links: List[str]) -> List[str]:
    teams: List[str] = []
    for link in links:
        parts = link[1:].split("/")
        if parts and parts[0] == "teams" and len(parts) > 2:
            teams.append(parts[1])
        if len(teams) == 2:
            break
    return teams


def scrape_boxscores(
    data_dir: str | Path = "data",
    start_year: int = 1992,
    end_year: int = 2022,
    request_delay_s: float = 0.5,
    overwrite: bool = False,
) -> None:
    """Scrape Basketball Reference boxscores into year-level pickles.

    Outputs:
      - {data_dir}/boxscores/{YEAR}.pkl

    Note: This is a lightweight scraper for personal/research use. Please be respectful
    of rate limits; adjust request_delay_s if needed.
    """
    data_dir = Path(data_dir)
    games_path = data_dir / "gamesByYear.pkl"
    boxscores_dir = data_dir / "boxscores"
    boxscores_dir.mkdir(parents=True, exist_ok=True)

    if games_path.exists():
        games_by_year = pd.read_pickle(games_path)
        logger.info("Loaded games index: %s", games_path)
    else:
        logger.info("No games index found; scraping schedules to build it.")
        games_by_year = get_dates_of_games(games_path,start_year,end_year)

    html_prefix = "https://www.basketball-reference.com/boxscores/"

    df_format = pd.DataFrame(columns=[
        "Season","Month","Day","Home","Away",
        "Home_Basic","Home_Advanced","Away_Basic","Away_Advanced"
    ])

    years = [y for y in sorted(games_by_year["Year"].unique()) if start_year <= int(y) <= end_year]
    for year in years:
        out_pkl = boxscores_dir / f"{int(year)}.pkl"
        if out_pkl.exists() and not overwrite:
            logger.info("Skipping existing: %s", out_pkl)
            continue

        group = games_by_year[games_by_year["Year"] == year]
        rows = []
        for _, row in group.iterrows():
            for game in row["Games"]:
                game_url = f"{html_prefix}{game}.html"
                try:
                    tables = pd.read_html(game_url, header=1)
                    idx = [0, int(len(tables)/2 - 1), int(len(tables)/2), -1]
                    tables = [tables[i] for i in idx]
                    _, links = get_soup(game_url)
                    teams = _get_team_names_from_links(links)
                    logger.info("%s %s %s | home=%s away=%s", row["Year"], row["Month"], str(game)[6:8], teams[1], teams[0])
                    tables = [_edit_table(t, i) for i, t in enumerate(tables[:4])]

                    tmp = pd.Series(index=df_format.columns, dtype=object)
                    tmp["Season"] = int(row["Year"])
                    tmp["Month"] = row["Month"]
                    tmp["Day"] = str(game)[6:8]
                    tmp["Home"] = teams[1]
                    tmp["Away"] = teams[0]
                    tmp["Home_Basic"] = tables[0]
                    tmp["Home_Advanced"] = tables[1]
                    tmp["Away_Basic"] = tables[2]
                    tmp["Away_Advanced"] = tables[3]
                except Exception as e:
                    logger.warning("Failed scrape for %s: %s", game_url, e)
                    tmp = pd.Series(index=df_format.columns, dtype=object)
                    tmp["Season"] = int(row["Year"])
                    tmp["Month"] = row["Month"]
                    tmp["Day"] = str(game)[6:8]
                    tmp["Home"] = str(game_url)[-8:-5]
                    tmp["Away"] = np.nan
                    tmp["Home_Basic"] = np.nan
                    tmp["Home_Advanced"] = np.nan
                    tmp["Away_Basic"] = np.nan
                    tmp["Away_Advanced"] = np.nan

                rows.append(tmp)
                time.sleep(request_delay_s)

        master = pd.DataFrame(rows, columns=df_format.columns)
        master.to_pickle(out_pkl)
        logger.info("Wrote: %s (%d games)", out_pkl, len(master))
