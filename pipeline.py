#!/usr/bin/env python3
"""pipeline.py - RetailCast India 28-day forecast (d_1914..d_1941).

Design — each choice traces to a check in audit/run_audit.py:
  - market_signal.csv EXCLUDED: target-derived leak, no horizon coverage.
  - vendor_signal.csv EXCLUDED as feature: = full-history-mean x DOW, regime-blind.
  - Level from blended short trailing windows (14/28/56) -> auto-adapts to regime breaks.
  - Structural exclusions: pre-launch zeros, CABLE dead window, item launch dates.
  - DOW multipliers shrunk 50% toward item-level pooled estimate.
  - Pooled event multipliers (Ram Navami, Eid al-Fitr fall in horizon).
  - GBM: LightGBM, Poisson objective (intermittent counts), + regime-shift features.
  - Final = 0.7*baseline + 0.3*GBM, EXCEPT series with a detected recent regime break,
    which use the baseline alone. Rationale: backtest showed the GBM reverts broken
    series toward their pre-break level, undoing the main data correction.

Usage:
    python3 pipeline.py --data ../data --out submission.csv
"""
import argparse, os
import numpy as np, pandas as pd

# LightGBM is the intended estimator. sklearn's HistGradientBoosting is a drop-in
# fallback for environments where lightgbm isn't installed (both are histogram GBDTs
# and both support Poisson loss, so results are close but NOT identical).
# NB: catch Exception, not ImportError. lightgbm imports fine but raises OSError from
# ctypes if the OpenMP runtime (libomp) is missing -- common on macOS and on slim
# containers. A crash there would produce no submission at all, so any import failure
# must degrade to the fallback rather than propagate.
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception as _e:  # noqa: BLE001
    _LGB_ERR = _e
    HAVE_LGB = False
if not HAVE_LGB:
    from sklearn.ensemble import HistGradientBoostingRegressor

H, LAST = 28, 1913
DCOLS = [f"d_{i}" for i in range(1, LAST + 1)]

def load(d):
    s = pd.read_csv(os.path.join(d, "sales_train.csv"))
    c = pd.read_csv(os.path.join(d, "calendar.csv"))
    p = pd.read_csv(os.path.join(d, "sell_prices.csv"))
    return s, c, p

def _fill_rows(A):
    """Row-wise forward then backward fill of NaNs, in numpy.

    Replaces `pd.DataFrame(A).ffill(axis=1).bfill(axis=1)`: axis=1 filling has carried
    deprecation warnings and changing semantics across pandas versions, and this is one
    of the few places the pipeline depended on them. numpy has no such churn.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[1]
    valid = ~np.isnan(A)
    # forward: carry the index of the last valid column
    idx = np.where(valid, np.arange(n)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(A, idx, axis=1)
    # backward: same trick on the reversed array, for leading NaNs
    validr = ~np.isnan(out[:, ::-1])
    idxr = np.where(validr, np.arange(n)[None, :], 0)
    np.maximum.accumulate(idxr, axis=1, out=idxr)
    out = np.take_along_axis(out[:, ::-1], idxr, axis=1)[:, ::-1]
    return out


def prep(sales, cal, prices):
    X = sales[DCOLS].to_numpy(float)
    item = sales["item_id"].to_numpy(); store = sales["store_id"].to_numpy()
    alld = DCOLS + [f"d_{i}" for i in range(LAST + 1, LAST + H + 1)]
    wd = cal.set_index("d")["wday"].reindex(alld).to_numpy()
    ev = cal.set_index("d")["event_name_1"].reindex(alld).to_numpy()
    wk = cal.set_index("d")["wm_yr_wk"].reindex(alld).to_numpy()
    # Plain dict lookup rather than MultiIndex .loc: no reliance on .loc returning a
    # scalar (it returns a Series if the key is duplicated) and no pandas indexing
    # semantics to drift between versions.
    pk = dict(zip(zip(prices["store_id"], prices["item_id"], prices["wm_yr_wk"]),
                  prices["sell_price"]))
    PR = np.full((len(sales), len(alld)), np.nan)
    for i in range(len(sales)):
        key_si = (store[i], item[i])
        for j, w in enumerate(wk):
            v = pk.get((key_si[0], key_si[1], w))
            if v is not None: PR[i, j] = v
    PR = _fill_rows(PR)
    return X, item, store, wd, ev, PR

def struct_start(X, item, i):
    """First usable training day (1-based). Structural, evidence-based exclusions only."""
    nz = np.nonzero(X[i])[0]
    s = nz[0] + 1 if len(nz) else 1          # pre-launch zeros are not demand
    if item[i] == "ELECTRONICS_1_CABLE":  s = max(s, 1441)   # dead window d_961-1440
    if item[i] == "HOMECARE_2_AGARBATTI": s = max(s, 1324)   # launch
    if item[i] == "HOMECARE_1_DETERGENT": s = max(s, 484)    # launch
    return s

def dow_factors(X, item, SS, end, win=364, shrink=0.5):
    n = len(X); F = np.ones((n, 7))
    for i in range(n):
        s = max(SS[i], end - win + 1)
        if end - s < 60: s = max(SS[i], end - 364 + 1)
        idx = np.arange(s - 1, end); x = X[i, idx]; w = WD_TRAIN[idx]
        m = x.mean()
        if m <= 0: continue
        for k in range(7):
            v = x[w == k + 1]
            if len(v) >= 4: F[i, k] = v.mean() / m
    for it in np.unique(item):
        sel = item == it; g = F[sel].mean(axis=0)
        F[sel] = (1 - shrink) * F[sel] + shrink * g
    return F / F.mean(axis=1, keepdims=True)

def event_mult(X, ev, end):
    tot = X.sum(axis=0); M = {}
    for name in set(e for e in ev[:end] if isinstance(e, str)):
        days = [d for d in range(1, end + 1) if ev[d - 1] == name]
        on, bs = [], []
        for d in days:
            if d < 40 or d + 20 > end: continue
            on.append(tot[d - 1]); bs.append(np.r_[tot[d - 21:d - 1], tot[d:d + 20]].mean())
        if len(on) >= 3: M[name] = float(np.clip(np.mean(on) / np.mean(bs), 0.8, 1.6))
    return M

def baseline(X, item, SS, ev, end, hz_wd):
    L = sum(np.array([X[i, max(SS[i], end - w + 1) - 1:end].mean() for i in range(len(X))])
            for w in (14, 28, 56)) / 3
    F = dow_factors(X, item, SS, end)
    P = L[:, None] * F[np.arange(len(X))[:, None], hz_wd - 1]
    M = event_mult(X, ev, end)
    for k in range(H):
        e = ev[end + k]
        if isinstance(e, str) and e in M: P[:, k] *= M[e]
    return P

FEATURE_NAMES = ["series", "item", "store", "horizon_step", "wday",
                 "mean_7", "mean_28", "mean_56", "mean_182", "sd_28", "zero_frac_56",
                 "dow_mean_182", "dow_ratio", "is_event", "price", "price_ratio",
                 "ratio_28_182", "ratio_56_182", "ratio_7_56", "break_tstat", "break_signed"]
CAT_IDX = [1, 2]

def feats(X, itc, stc, PR, wd, ev, o, h, freeze_price=True):
    n = len(X); t = o + h; k = wd[t] - 1
    idx = np.arange(o - 182, o)
    dm = np.array([X[i, idx[WD_TRAIN[idx] == k + 1]].mean() for i in range(n)])
    m7 = X[:, o-7:o].mean(1); m28 = X[:, o-28:o].mean(1)
    m56 = X[:, o-56:o].mean(1); m182 = X[:, o-182:o].mean(1)
    # Price is FROZEN at the last observed week: the one historical deep promo
    # (MH_2 PICKLE wk2040) produced no lift, so we refuse to extrapolate elasticity.
    px = PR[:, o - 1] if freeze_price else PR[:, t]
    # Regime-shift features: give the model an explicit way to represent "this series
    # is in a new regime", which it otherwise lacks and which caused it to revert
    # broken series toward their pre-break level.
    a = X[:, max(0, o - 181):o - 56]; b = X[:, o - 56:o]
    tstat = np.abs(a.mean(1) - b.mean(1)) / np.sqrt(
        a.var(1) / max(a.shape[1], 1) + b.var(1) / b.shape[1] + 1e-9)
    r56 = m56 / np.maximum(m182, 1e-6)
    return np.c_[np.arange(n), itc, stc, np.full(n, h + 1), np.full(n, wd[t]),
                 m7, m28, m56, m182, X[:, o-28:o].std(1), (X[:, o-56:o] == 0).mean(1), dm,
                 dm / np.maximum(m182, 1e-6),
                 np.full(n, 1 if isinstance(ev[t], str) else 0), px,
                 px / np.maximum(PR[:, o - 1], 1e-6),
                 m28 / np.maximum(m182, 1e-6), r56, m7 / np.maximum(m56, 1e-6),
                 tstat, (tstat > 4).astype(float) * np.sign(r56 - 1)]

def fit_gbm(Xtr, ytr, seed=0):
    """Poisson objective: these are intermittent counts (14 of 60 series exceed 70%
    zero-days), so squared-error loss is misspecified. Poisson lifted the GBM from
    0.700 to 0.691 mean RMSSE in backtest -- the single largest model-side gain."""
    if HAVE_LGB:
        ds = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES,
                         categorical_feature=[FEATURE_NAMES[i] for i in CAT_IDX],
                         free_raw_data=False)
        params = dict(objective="poisson", metric="poisson", learning_rate=0.06,
                      num_leaves=63, min_data_in_leaf=40, lambda_l2=1.0,
                      feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                      seed=seed, verbosity=-1, num_threads=0)
        return lgb.train(params, ds, num_boost_round=400)
    return HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.06,
        max_depth=7, min_samples_leaf=40, l2_regularization=1.0,
        categorical_features=CAT_IDX, random_state=seed).fit(Xtr, ytr)

def gbm_predict(X, item, itc, stc, PR, wd, ev, SS, end):
    Fs, ys = [], []
    for o in range(400, end - H + 1, 14):
        for h in range(H):
            if o + h >= end: continue
            F = feats(X, itc, stc, PR, wd, ev, o, h)
            keep = np.array([o + h + 1 >= SS[i] for i in range(len(X))])
            Fs.append(F[keep]); ys.append(X[:, o + h][keep])
    g = fit_gbm(np.vstack(Fs), np.concatenate(ys))
    return np.column_stack([np.clip(g.predict(feats(X, itc, stc, PR, wd, ev, end, h)), 0, None)
                            for h in range(H)])

def break_flag(X, SS, end, tstat=4.0, lo=0.65, hi=1.55):
    """Detect a sustained recent level shift per series using ONLY data <= end.
    Compares the trailing 56 days against the preceding ~124. No hardcoded dates,
    so next month's break in a different store is caught automatically."""
    n = len(X); f = np.zeros(n, bool)
    for i in range(n):
        a = X[i, max(SS[i] - 1, end - 181):end - 56]; b = X[i, end - 56:end]
        if len(a) < 60 or a.mean() <= 0: continue
        va = a.var(ddof=1) / len(a); vb = b.var(ddof=1) / len(b)
        if va + vb <= 0: continue
        t = abs(a.mean() - b.mean()) / np.sqrt(va + vb); r = b.mean() / a.mean()
        if t > tstat and (r < lo or r > hi): f[i] = True
    return f

def main():
    # Defaults resolved relative to THIS FILE, not the caller's cwd, so that the bare
    # `python3 pipeline.py` in retailcast_config.json works from any working directory.
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, os.pardir, "data"))
    ap.add_argument("--out", default=os.path.join(here, "submission.csv"))
    ap.add_argument("--w-base", type=float, default=0.7)
    a = ap.parse_args()
    if not os.path.isdir(a.data):
        raise SystemExit(f"data dir not found: {a.data}\n"
                         f"pass --data explicitly, e.g. --data /path/to/data")
    if HAVE_LGB:
        print("estimator: lightgbm (poisson)")
    else:
        print("estimator: sklearn HistGradientBoosting (poisson) [fallback]")
        print(f"  lightgbm unavailable: {type(_LGB_ERR).__name__}: {_LGB_ERR}")
        print("  on macOS this is usually a missing OpenMP runtime -> brew install libomp")
    sales, cal, prices = load(a.data)
    global WD_TRAIN
    X, item, store, wd, ev, PR = prep(sales, cal, prices)
    WD_TRAIN = wd[:LAST]
    SS = np.array([struct_start(X, item, i) for i in range(len(X))])
    itc = pd.Categorical(item).codes; stc = pd.Categorical(store).codes
    hz_wd = wd[LAST:LAST + H]
    Pb = baseline(X, item, SS, ev, LAST, hz_wd)
    Pg = gbm_predict(X, item, itc, stc, PR, wd, ev, SS, LAST)
    bf = break_flag(X, SS, LAST)
    P = Pb.copy()
    safe = ~bf
    P[safe] = a.w_base * Pb[safe] + (1 - a.w_base) * Pg[safe]
    P = np.clip(P, 0, None)
    print(f"regime-break series held to baseline ({bf.sum()}): "
          f"{[sales['id'].iloc[i] for i in np.nonzero(bf)[0]]}")
    # Fail loudly here rather than emit a structurally invalid file.
    assert P.shape == (len(sales), H), f"bad shape {P.shape}"
    assert np.isfinite(P).all(), "non-finite forecast values"
    assert (P >= 0).all(), "negative forecast values"
    out = pd.DataFrame(P, columns=[f"F{i}" for i in range(1, H + 1)])
    out.insert(0, "id", sales["id"].values)
    out.to_csv(a.out, index=False, float_format="%.4f")
    print(f"wrote {a.out}  rows={len(out)}  total units={P.sum():.0f}")
    # component predictions kept for the sanity checks in audit/
    np.save(os.path.join(here, "_baseline.npy"), Pb)
    np.save(os.path.join(here, "_gbm.npy"), Pg)

if __name__ == "__main__":
    main()