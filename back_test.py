#!/usr/bin/env python3
"""Rolling-origin backtest — the model-selection evidence.

Uses the same estimator path as forecast.py, so running this with lightgbm installed
gives the numbers for the shipped model.

  python audit/backtest.py --data data --origins 6
"""
import argparse, sys, os
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline as fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.pardir, "data"))
    ap.add_argument("--origins", type=int, default=6)
    ap.add_argument("--w-base", type=float, default=0.7)
    a = ap.parse_args()

    sales, cal, prices = fc.load(a.data)
    X, item, store, wd, ev, PR = fc.prep(sales, cal, prices)
    fc.WD_TRAIN = wd[:fc.LAST]
    SS = np.array([fc.struct_start(X, item, i) for i in range(len(X))])
    itc = pd.Categorical(item).codes
    stc = pd.Categorical(store).codes
    ids = sales["id"].tolist()
    k3 = np.array(["KA_3" in s for s in ids])

    print("estimator: " + ("lightgbm (poisson)" if fc.HAVE_LGB
          else "sklearn HistGradientBoosting (poisson) [fallback]"))

    def denoms(end):
        out = np.empty(len(X))
        for i in range(len(X)):
            s = min(SS[i], end - 90)
            d = np.diff(X[i, s - 1:end])
            out[i] = np.mean(d ** 2) if np.mean(d ** 2) > 0 else 1.0
        return out

    origins = [fc.LAST - fc.H * k for k in range(1, a.origins + 1)]
    rows = []
    for o in origins:
        # Train only on origins whose targets end strictly before o -> no leakage.
        Fs, ys = [], []
        for oo in range(400, o - fc.H + 1, 14):
            for h in range(fc.H):
                if oo + h >= o:
                    continue
                F = fc.feats(X, itc, stc, PR, wd, ev, oo, h)
                keep = np.array([oo + h + 1 >= SS[i] for i in range(len(X))])
                Fs.append(F[keep]); ys.append(X[:, oo + h][keep])
        g = fc.fit_gbm(np.vstack(Fs), np.concatenate(ys))
        Pg = np.column_stack([
            np.clip(g.predict(fc.feats(X, itc, stc, PR, wd, ev, o, h)), 0, None)
            for h in range(fc.H)])

        Pb = fc.baseline(X, item, SS, ev, o, wd[o:o + fc.H])
        bf = fc.break_flag(X, SS, o)
        Pgate = Pb.copy()
        safe = ~bf
        Pgate[safe] = a.w_base * Pb[safe] + (1 - a.w_base) * Pg[safe]

        den = denoms(o); act = X[:, o:o + fc.H]
        r = lambda P, m=slice(None): float(
            np.sqrt(((act[m] - P[m]) ** 2).mean(1) / den[m]).mean())
        wape = lambda P: float(np.abs(act - P).sum() / act.sum())
        rows.append(dict(origin=f"d_{o+1}", base=r(Pb), gbm=r(Pg),
                         blend=r(a.w_base * Pb + (1 - a.w_base) * Pg), gated=r(Pgate),
                         base_KA3=r(Pb, k3), gbm_KA3=r(Pg, k3), gated_KA3=r(Pgate, k3),
                         wape_gated=wape(Pgate), n_break=int(bf.sum())))
        print(f"  {rows[-1]['origin']:9s} base {rows[-1]['base']:.4f}  "
              f"gbm {rows[-1]['gbm']:.4f}  gated {rows[-1]['gated']:.4f}  "
              f"(KA_3: base {rows[-1]['base_KA3']:.4f} gbm {rows[-1]['gbm_KA3']:.4f})")

    D = pd.DataFrame(rows)
    print("\n=== mean RMSSE ===")
    print(D[["base", "gbm", "blend", "gated"]].mean().round(4).to_string())
    print("\n=== mean RMSSE, KA_3 series only ===")
    print(D[["base_KA3", "gbm_KA3", "gated_KA3"]].mean().round(4).to_string())
    print(f"\nmean WAPE (gated): {D['wape_gated'].mean():.4f}")
    print(f"per-origin RMSSE sd (gated): {D['gated'].std():.4f}  "
          f"range {D['gated'].min():.4f}-{D['gated'].max():.4f}")
    print("\nNote: window lengths and blend weight were tuned on these same origins,")
    print("so treat this as a mildly optimistic estimate of horizon error.")


if __name__ == "__main__":
    main()