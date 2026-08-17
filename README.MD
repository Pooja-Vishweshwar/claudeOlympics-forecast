# RetailCast India — 28-day demand forecast

Forecasts `d_1914`–`d_1941` for 60 product-store series.

## Reproduce

```bash
pip install -r requirements.txt          # via JFrog Artifactory (see note below)
python3 pipeline.py --data data --out submission.csv
```

`pipeline.py`'s `--data` defaults to `../data` relative to this file, but the `data/`
directory ships *inside* this repo (see Layout below), so pass `--data data` (or
`./data`) explicitly rather than relying on the default. Override `--out` too if your
layout differs (`--data /path/to/data`).

Reproduce the audit findings and the model-selection table:

```bash
python3 run_audit.py --data data     # reproduces the evidence for every data verdict
python3 back_test.py --data data     # rolling-origin backtest (model-selection table)
```

Deterministic (fixed seed); regenerates an identical `submission.csv`. Targets Python
3.11+ (`pandas==3.0.2` requires it) — if your system `python3` resolves to something
older, create a venv against a newer interpreter, e.g. `python3.12 -m venv .venv`.

**Dependency surface is deliberately small.** The pipeline uses only `pd.read_csv`,
`to_csv`, `set_index`, `reindex`, `to_numpy`, `insert`, `DataFrame` and `Categorical` from
pandas — no `groupby`, `merge`, `pivot`, or `axis=1` filling, all of which have shifted
semantics across versions. Row-wise NaN filling is done in numpy (`_fill_rows`) and price
lookup via a plain dict, so neither depends on pandas indexing behaviour. Runs clean under
`-W error::FutureWarning -W error::DeprecationWarning`. Before writing, the script asserts
shape, finiteness and non-negativity, so a bad environment fails loudly rather than
producing an invalid submission.

> **Package note.** This repo is built inside Lilly, where packages are pulled from JFrog
> Artifactory rather than public PyPI. Set your index before installing, e.g.
> `export PIP_INDEX_URL=https://elilillyco.jfrog.io/artifactory/api/pypi/<pypi-repo>/simple`
> (get the exact repo path and credentials from the Developer Platform front door).
>
> **Estimator note.** The primary estimator is LightGBM with `objective="poisson"`.
> `pipeline.py` falls back to `sklearn`'s `HistGradientBoostingRegressor(loss="poisson")`
> if lightgbm is unavailable, and prints which one it used. Any lightgbm import failure —
> including `OSError` from a missing OpenMP runtime, not just `ImportError` — degrades to
> the fallback so the run always produces a submission. On macOS, `brew install libomp`
> fixes the OpenMP case. Run `audit/backtest.py` to measure whichever estimator is active.

## What the pipeline does

1. **Excludes `market_signal.csv`** — target-derived leak, no horizon coverage.
2. **Excludes `vendor_signal.csv`** as a feature — it is `full-history mean × day-of-week`
   and regime-blind; retained only as a benchmark (mean RMSSE 0.912 on `d_1886`–`d_1913`).
3. **Structural exclusions** — pre-launch zeros; the CABLE dead window `d_961`–`d_1440`;
   launch dates for DETERGENT (`d_484`) and AGARBATTI (`d_1324`).
4. **Level** — mean of trailing 14/28/56-day windows, which tracks regime breaks
   automatically without hardcoded dates.
5. **Seasonality** — per-series day-of-week factors shrunk 50% toward the item-level pooled
   estimate; pooled event multipliers (Ram Navami and Eid al-Fitr fall in the horizon).
6. **Price frozen at the last observed week** — the one deep historical promo produced no
   measurable lift, so the pipeline refuses to extrapolate elasticity.
7. **GBM** — LightGBM, Poisson objective (intermittent counts; 14 of 60 series exceed 70%
   zero-days), direct per-horizon-step, with explicit regime-shift ratio features.
8. **Blend** — `0.7 × baseline + 0.3 × GBM`, except series with a detected recent break,
   which use the baseline alone — the GBM still reverts those toward their pre-break level.

## Layout

```
pipeline.py              # single entry point -> submission.csv
forecast.py              # forecasting logic used by pipeline.py
retailcast_config.json   # config + recorded data verdicts
requirements.txt
run_audit.py             # reproduces the evidence for every data verdict
back_test.py             # rolling-origin backtest (model selection)
data/                    # sales_train.csv, calendar.csv, sell_prices.csv, etc.
submission.csv
```

The shared data ships inside this repo at `data/`, not alongside it — pass
`--data data` explicitly (see Reproduce above).
`retailcast_config.json` is the challenge run manifest and is left exactly as issued.