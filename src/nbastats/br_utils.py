from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import urllib3

logger = logging.getLogger("nbastats")


def get_soup(url: str) -> Tuple[str, List[str]]:
    """Fetch a URL and return cleaned text + list of href links.

    Note: Basketball Reference pages are HTML; we keep this function small and explicit so it can
    be swapped out later (requests/session reuse, caching, retries, etc.).
    """
    logger.info("Fetching: %s", url)
    req = urllib3.PoolManager()
    res = req.request("GET", url)
    soup = BeautifulSoup(res.data, "html.parser")

    # Remove script/style for cleaner text extraction.
    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    links: List[str] = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href:
            links.append(href)
    return text, links


def get_team_name_abbrevs(out_path: str | Path) -> pd.DataFrame:
    """Scrape team abbreviations by season and save to a pickle."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    teams_by_year: Dict[str, List[str]] = {}
    for year in np.arange(1992, 2023):
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}.html"
        _, links = get_soup(url)

        teams: List[str] = []
        for link in links:
            if len(link) > 3:
                parts = link[1:].split("/")
                # expected: ['teams', 'BOS', '2020.html']
                try:
                    if parts[0] == "teams" and parts[1] and int(parts[2].split(".")[0]) == year:
                        teams.append(parts[1])
                except Exception:
                    continue

        teams_by_year[str(year)] = pd.Series(teams).unique().tolist()

    df = pd.DataFrame({"Year": list(teams_by_year.keys()),
                       "Teams": [teams_by_year[y] for y in teams_by_year.keys()]})
    df.to_pickle(out_path)
    logger.info("Wrote team abbrevs: %s", out_path)
    return df


def get_dates_of_games(out_path: str | Path,
                       start_year: int = 1992,
                       end_year: int = 2022 ,
                       ) -> pd.DataFrame:
    """Scrape Basketball Reference schedule pages to get game ids per month."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    months = [
        "october", #"november", "december", "january", "february", "march", "april", "may", "june"
    ]

    rows = []
    for year in np.arange(start_year, end_year):
        for month in months:
            url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
            text, links = get_soup(url)
            if text == "404":
                continue

            games: List[str] = []
            for link in links:
                parts = link[1:].split("/")
                if len(parts) == 2 and parts[0] == "boxscores" and parts[1].endswith(".html"):
                    games.append(parts[1].split(".")[0])

            rows.append({"Year": year, "Month": month, "Games": games})

    game_df = pd.DataFrame(rows, columns=["Year", "Month", "Games"])
    game_df.to_pickle(out_path)
    logger.info("Wrote games-by-year: %s", out_path)
    return game_df
