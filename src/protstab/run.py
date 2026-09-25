"""Train + calibrated evaluation on the meltome split.

Proper split-conformal: ridge is fit on train-minus-validation, absolute
residuals calibrated on the FLIP `validation` subset (still train-side),
and coverage measured on the held-out cluster split. The intervals are
valid only because calibration never sees the test set.

Applicability domain: Euclidean distance in embedding space to the
training centroid, bucketed into quartiles on the test set. Does
coverage degrade where the model extrapolates?
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from protstab.config import load_config
from protstab.encode import build_features, composition
from protstab.provenance import write_manifest


def fit_ridge(X, y, alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
    m.fit(X, y)
    return m


def metrics(y, yhat):
    from scipy.stats import spearmanr
    err = yhat - y
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "spearman": float(spearmanr(y, yhat)[0]),
    }


def conformal_eval(m, X_cal, y_cal, X_te, y_te, alpha):
    """Split conformal: q = (1-alpha) quantile of |y - yhat| on cal."""
    q = np.quantile(np.abs(m.predict(X_cal) - y_cal), 1 - alpha)
    yhat = m.predict(X_te)
    cov = np.abs(yhat - y_te) <= q
    return q, yhat, cov


def ad_table(centroid_dist, cov, resid_abs, bins):
    """Coverage + error by applicability-domain distance quartile."""
    edges = np.quantile(centroid_dist, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, np.inf
    rows = []
    for b in range(bins):
        sel = (centroid_dist > edges[b]) & (centroid_dist <= edges[b + 1])
        if sel.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "n": int(sel.sum()),
            "dist_range": [float(edges[b]),
                           float(min(edges[b + 1], centroid_dist.max()))],
            "coverage": float(cov[sel].mean()),
            "mae": float(resid_abs[sel].mean()),
        })
    return rows


def main(in_parquet: str, out_json: str):
    cfg = load_config()
    df = pd.read_parquet(in_parquet)
    ev = cfg["evaluation"]
    alpha = ev["conformal_alpha"]

    tr = df[(df.set == "train") & (~df.validation)].reset_index(drop=True)
    cal = df[(df.set == "train") & (df.validation)].reset_index(drop=True)
    te = df[df.set == "test"].reset_index(drop=True)
    max_len = cfg["dataset"]["max_len"]
    n_trunc = int((df.sequence.str.len() > max_len).sum())
    if n_trunc:
        print(f"truncating {n_trunc} sequences > {max_len} aa")

    enc = dict(cfg["encoder"])
    enc["max_len"] = max_len
    stem = str(Path(in_parquet).with_suffix(""))
    results = {}

    # Baseline first: composition features, same protocol
    for name, Xf in {
        "composition": composition(
            pd.concat([tr.sequence, cal.sequence, te.sequence])),
        "esm2": build_features(
            pd.concat([tr.sequence, cal.sequence, te.sequence]),
            enc, cache_stem=f"{stem}_all"),
    }.items():
        ntr, ncal = len(tr), len(cal)
        X_tr, X_cal, X_te = Xf[:ntr], Xf[ntr:ntr + ncal], Xf[ntr + ncal:]
        m = fit_ridge(X_tr, tr.target.to_numpy(), cfg["model"]["alpha"])
        q, yhat, cov = conformal_eval(
            m, X_cal, cal.target.to_numpy(), X_te, te.target.to_numpy(),
            alpha)
        centroid = X_tr.mean(axis=0)
        dist = np.linalg.norm(X_te - centroid, axis=1)
        res = {
            "conformal_q": float(q),
            "coverage": float(cov.mean()),
            "mean_interval_width": float(2 * q),
            **metrics(te.target.to_numpy(), yhat),
            "ad_table": ad_table(dist, cov, np.abs(yhat - te.target.to_numpy()),
                                 ev["ad_bins"]),
        }
        results[name] = res
        print(f"{name}: rmse {res['rmse']:.2f} mae {res['mae']:.2f} "
              f"spearman {res['spearman']:.3f} | coverage "
              f"{res['coverage']:.3f} @ target {1 - alpha:.0%} "
              f"(width {res['mean_interval_width']:.1f} C)")

    out = {
        "config": {"encoder": enc, "model": cfg["model"],
                   "conformal_alpha": alpha, "max_len": max_len,
                   "n_truncated": n_trunc},
        "n_train": len(tr), "n_cal": len(cal), "n_test": len(te),
        "target_unit": "C",
        "results": results,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    manifest = (Path(out_json).parent
                / ("provenance" + Path(out_json).stem.removeprefix("summary")
                   + ".json"))
    write_manifest(str(manifest), inputs=[in_parquet])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
