from __future__ import annotations

import sys
import argparse
from pathlib import Path

from .logging_config import setup_logging
from .scrape import scrape_boxscores
from .build_master import combine_boxscores, build_master_by_year
from .weighted_stats import build_weighted_stats_by_year
from .training_set import build_training_set
from .modeling import train_baseline_classifier


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nbastats",
        description="NBAStats: scraping → dataset build → baseline modeling",
    )
    p.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    p.add_argument("--log-file", default=None, help="Optional log file path")

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scrape", help="Scrape Basketball Reference boxscores into data/boxscores")
    s.add_argument("--data-dir", default="data", help="Base data directory")
    s.add_argument("--start-year", type=int, default=1992)
    s.add_argument("--end-year", type=int, default=2022)
    s.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    s.add_argument("--overwrite", action="store_true")

    c = sub.add_parser("combine-boxscores", help="Combine year pickles into one AllYears.pkl")
    c.add_argument("--boxscores-dir", default="data/boxscores")
    c.add_argument("--out", default="data/AllYears.pkl")

    m = sub.add_parser("build-master", help="Build per-year master tables (requires playerData.pkl)")
    m.add_argument("--all-years", default="data/AllYears.pkl")
    m.add_argument("--player-data", default="data/playerData.pkl")
    m.add_argument("--out-dir", default="data/master_by_year")
    m.add_argument("--overwrite", action="store_true")

    w = sub.add_parser("weighted-stats", help="Build per-year game-level weighted stats from master tables")
    w.add_argument("--master-dir", default="data/master_by_year")
    w.add_argument("--out-dir", default="data/weighted_stats_by_year")
    w.add_argument("--start-year", type=int, default=2008)
    w.add_argument("--end-year", type=int, default=2019)
    w.add_argument("--overwrite", action="store_true")

    t = sub.add_parser("make-training", help="Build trainingData.pkl from weighted stats")
    t.add_argument("--weighted-dir", default="data/weighted_stats_by_year")
    t.add_argument("--out", default="data/trainingData.pkl")
    t.add_argument("--start-year", type=int, default=1992)
    t.add_argument("--end-year", type=int, default=2019)
    t.add_argument("--min-history", type=int, default=20, help="Min games history per team")

    b = sub.add_parser("train-baseline", help="Train a baseline classifier on trainingData.pkl")
    b.add_argument("--training", default="data/trainingData.pkl")
    b.add_argument("--out-dir", default="outputs")
    b.add_argument("--test-frac", type=float, default=0.2)
    b.add_argument("--seed", type=int, default=0)

    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    setup_logging(args.log_level, args.log_file)

    if args.cmd == "scrape":
        scrape_boxscores(
            data_dir=args.data_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            request_delay_s=args.delay,
            overwrite=args.overwrite,
        )
    elif args.cmd == "combine-boxscores":
        combine_boxscores(args.boxscores_dir, args.out)
    elif args.cmd == "build-master":
        build_master_by_year(args.all_years, args.player_data, args.out_dir, overwrite=args.overwrite)
    elif args.cmd == "weighted-stats":
        build_weighted_stats_by_year(
            args.master_dir, args.out_dir, start_year=args.start_year, end_year=args.end_year, overwrite=args.overwrite
        )
    elif args.cmd == "make-training":
        build_training_set(
            args.weighted_dir, args.out, start_year=args.start_year, end_year=args.end_year, min_games_history=args.min_history
        )
    elif args.cmd == "train-baseline":
        train_baseline_classifier(args.training, out_dir=args.out_dir, test_frac=args.test_frac, random_seed=args.seed)
    else:
        p.error("Unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
