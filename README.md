# NBAStats — End-to-End NBA Data Pipeline (Scraping → Dataset → Baseline ML)

This repo is a compact, reviewable example of how I build **repeatable scientific data workflows**:
- scrape messy web data,
- normalize and organize it into structured datasets, and
- run baseline ML experiments with clear, inspectable outputs.

It’s intentionally lightweight (plain Python + pandas), with an emphasis on:
**logging, repeatability, readable code, and “what runs first / what runs next.”**

---

## What a reviewer should look at

If you’re reviewing this as a code sample, the “story” is:

1. **`src/nbastats/scrape.py`** — scrapes game boxscores and stores raw year-level pickles  
2. **`src/nbastats/build_master.py`** — builds a per-player, per-game master table (requires a player metadata table)  
3. **`src/nbastats/weighted_stats.py`** — aggregates player stats into **game-level features**  
4. **`src/nbastats/training_set.py`** — produces an ML-ready dataset using a “last N games” rolling history  
5. **`src/nbastats/modeling.py`** — trains a simple baseline classifier (logistic regression)

Each stage can be re-run independently; intermediate artifacts are saved to disk.

---

## Repository layout

```
.
├── src/
│   └── nbastats/
│       ├── cli.py              # argparse CLI with subcommands
│       ├── logging_config.py   # consistent logging setup
│       ├── br_utils.py         # Basketball Reference helpers (fetching, schedules)
│       ├── scrape.py           # boxscore scraping
│       ├── build_master.py     # combine + master dataset build
│       ├── weighted_stats.py   # game-level aggregation
│       ├── training_set.py     # training dataset construction
│       └── modeling.py         # baseline ML training + metrics
├── scripts/
│   └── nbastats                # convenience entrypoint
└── README.md
```

---

## Setup

### 1) Create an environment
```bash
conda env create -f environment.yml
conda activate nba-stats
export PYTHONPATH=$PWD/src
./scripts/nbastats --help
```
## Quickstart (end-to-end)

> **Note:** scraping can take time; the code includes a request delay by default.

### 1) Scrape boxscores
Writes year-level pickles to `data/boxscores/`.

```bash
./scripts/nbastats scrape --data-dir data --start-year 2018 --end-year 2019 --delay 0.75
```

Outputs:
- `data/gamesByYear.pkl`
- `data/boxscores/2018.pkl`, `data/boxscores/2019.pkl`, ...

### 2) Combine year pickles
```bash
./scripts/nbastats combine-boxscores --boxscores-dir data/boxscores --out data/AllYears.pkl
```

Output:
- `data/AllYears.pkl`

### 3) Build the per-year master tables
This stage expects a `playerData.pkl` (player metadata). In the original version of this project,
player metadata was scraped separately. For review purposes, the important part is the workflow
and dataset joins.

```bash
./scripts/nbastats build-master --all-years data/AllYears.pkl --player-data data/playerData.pkl --out-dir data/master_by_year
```

Outputs:
- `data/master_by_year/2018_master.pkl`, etc.

### 4) Build game-level weighted stats
```bash
./scripts/nbastats weighted-stats --master-dir data/master_by_year --out-dir data/weighted_stats_by_year --start-year 2018 --end-year 2019
```

Outputs:
- `data/weighted_stats_by_year/2018_weightedStatsByGame.pkl`, etc.

### 5) Build the training dataset
By default, each game is represented by the **difference between the home team and away team**
over the previous **20 games**.

```bash
./scripts/nbastats make-training --weighted-dir data/weighted_stats_by_year --out data/trainingData.pkl --start-year 2018 --end-year 2019 --min-history 20
```

Output:
- `data/trainingData.pkl`

### 6) Train a baseline model
Predicts whether the home team wins (`score_diff > 0`) using a logistic regression baseline.

```bash
./scripts/nbastats train-baseline --training data/trainingData.pkl --out-dir outputs --test-frac 0.2 --seed 0
```

Outputs:
- `outputs/baseline_logreg.pkl`
- logs include accuracy + ROC-AUC

---

## Logging

All commands support:
- `--log-level DEBUG|INFO|WARNING|ERROR`
- `--log-file path/to/file.log`

Example:
```bash
./scripts/nbastats scrape --data-dir data --start-year 2019 --end-year 2019 --log-level INFO --log-file outputs/run.log
```

---

## Notes / limitations

- This scraper is intended for personal/research use. Please be respectful with request rates.
- The modeling script is intentionally baseline (clean workflow > best accuracy).
- Some historical Basketball Reference tables contain inconsistent formatting; the pipeline is designed to fail loudly and log clearly.

---

## Why this is a useful code sample

This repo demonstrates:
- applied scientific computing workflows (scrape → normalize → aggregate → model)
- repeatable dataset construction with intermediate artifacts
- basic performance hygiene (vectorized pandas operations, avoid manual busywork)
- professional ergonomics (argparse entrypoints + structured logging)

