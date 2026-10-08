"""What makes a headline get clicked? Discover on the exploratory split, confirm on the confirmatory split.

Within one test, variants are randomized, so comparing headlines *inside* a test is causal for that
test's audience. Model: weighted least squares of variant log-odds CTR on headline features, with
test fixed effects (both sides demeaned within test). Only variants that share an image with another
variant in the same test are used, so the comparison isolates the headline.
Standard errors are inflated by PHI from the A/A analysis. Writes outputs/headlines.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NOT_RANDOMIZED = ("2013-06-25", "2014-01-11")   # [start, end)
PHI = json.loads((ROOT / "outputs/experiments.json").read_text())["aa"]["phi"]

FEATURES = {
    "question":      lambda h: "?" in h,
    "number":        lambda h: bool(re.search(r"\d", h)),
    "you_your":      lambda h: bool(re.search(r"\byou(r|'re|’re)?\b", h, re.I)),
    "this_these":    lambda h: bool(re.search(r"\b(this|these)\b", h, re.I)),
    "negative_word": lambda h: bool(re.search(r"\b(no|not|never|worst|stop|don't|don’t|can't|can’t|nothing)\b", h, re.I)),
    "exclamation":   lambda h: "!" in h,
    "quote":         lambda h: bool(re.search(r"[\"“”]", h)),
    "long_15plus_words": lambda h: len(h.split()) >= 15,
}
LABELS = {"question": "Asks a question", "number": "Contains a number", "you_your": "Says 'you/your'",
          "this_these": "Says 'this/these'", "negative_word": "Negative word (no, never, worst…)",
          "exclamation": "Exclamation mark", "quote": "Quotation", "long_15plus_words": "15+ words"}


def load(split):
    d = pd.read_csv(ROOT / f"data/upworthy-archive-{split}-packages-03.12.2020.csv", low_memory=False)
    d = d.rename(columns={"clickability_test_id": "test_id"})
    created = pd.to_datetime(d.created_at, format="ISO8601")
    d = d[~((created >= NOT_RANDOMIZED[0]) & (created < NOT_RANDOMIZED[1]))]           # authors: not fully randomized in this window
    d = d[(d.impressions > 0) & (d.clicks <= d.impressions)].copy()
    d["headline"] = d.headline.fillna("")
    d["eyecatcher_id"] = d.eyecatcher_id.fillna("none")
    # keep (test, image) groups with >= 2 distinct headlines: the headline is the only thing that varies
    g = ["test_id", "eyecatcher_id"]
    d["n_head"] = d.groupby(g).headline.transform("nunique")
    d = d[d.n_head >= 2].copy()
    d["grp"] = d.test_id + "|" + d.eyecatcher_id
    for k, f in FEATURES.items():
        d[k] = d.headline.map(f).astype(float)
    d["y"] = np.log((d.clicks + 0.5) / (d.impressions - d.clicks + 0.5))
    d["w"] = 1 / (PHI * (1 / (d.clicks + 0.5) + 1 / (d.impressions - d.clicks + 0.5)))
    return d


def fe_wls(d):
    cols = list(FEATURES)
    X = d[cols].to_numpy()
    y = d.y.to_numpy()
    w = d.w.to_numpy()
    # weighted within-group demeaning (fixed effect per test x image)
    gid = d.grp.astype("category").cat.codes.to_numpy()
    sw = np.bincount(gid, w)
    ybar = np.bincount(gid, w * y) / sw
    Xbar = np.column_stack([np.bincount(gid, w * X[:, j]) / sw for j in range(X.shape[1])])
    yt, Xt = y - ybar[gid], X - Xbar[gid]
    W = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xt * W[:, None], yt * W, rcond=None)
    # cluster-robust (by test) standard errors
    resid = (yt - Xt @ beta)
    XtWX_inv = np.linalg.pinv((Xt * w[:, None]).T @ Xt)
    tid = d.test_id.astype("category").cat.codes.to_numpy()
    S = np.zeros((X.shape[1], X.shape[1]))
    score = Xt * (w * resid)[:, None]
    for g in np.unique(tid):
        s = score[tid == g].sum(0)
        S += np.outer(s, s)
    V = XtWX_inv @ S @ XtWX_inv
    se = np.sqrt(np.diag(V))
    return pd.DataFrame({"beta": beta, "se": se}, index=cols)


out = {}
res = {}
for split in ("exploratory", "confirmatory"):
    d = load(split)
    r = fe_wls(d)
    r["lift"] = np.expm1(r.beta)
    r["lo"] = np.expm1(r.beta - 1.96 * r.se)
    r["hi"] = np.expm1(r.beta + 1.96 * r.se)
    r["p"] = 2 * stats.norm.sf(np.abs(r.beta / r.se))
    r["prevalence"] = d[list(FEATURES)].mean()
    res[split] = r
    out[split] = {"variants": int(len(d)), "tests": int(d.test_id.nunique()),
                  "impressions": int(d.impressions.sum())}

exp, con = res["exploratory"], res["confirmatory"]
# Discovery rule: p < 0.05 in exploratory. Confirmation: same sign and p < 0.05 in confirmatory.
rows = []
for k in FEATURES:
    disc = exp.loc[k, "p"] < 0.05
    conf = disc and np.sign(exp.loc[k, "beta"]) == np.sign(con.loc[k, "beta"]) and con.loc[k, "p"] < 0.05
    rows.append({"feature": k, "label": LABELS[k],
                 "explore_lift": exp.loc[k, "lift"], "explore_p": exp.loc[k, "p"],
                 "confirm_lift": con.loc[k, "lift"], "confirm_lo": con.loc[k, "lo"], "confirm_hi": con.loc[k, "hi"],
                 "confirm_p": con.loc[k, "p"], "prevalence": con.loc[k, "prevalence"],
                 "discovered": bool(disc), "confirmed": bool(conf)})
out["features"] = rows
(ROOT / "outputs/headlines.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps({k: v for k, v in out.items() if k != "features"}, indent=1))
print(pd.DataFrame(rows).set_index("feature").drop(columns="label").round(4).to_string())
