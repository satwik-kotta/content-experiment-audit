"""How trustworthy were 32K content A/B tests? (Upworthy Research Archive)

1. Built-in A/A tests: identical variants inside the same test should differ only by chance.
   They do not, so estimate a variance-inflation factor (PHI) and use it everywhere below.
2. Power: what lift could a typical test actually detect?
3. Winner's curse: how much of a winning headline's lift is noise? (A/A baseline + empirical-Bayes shrinkage)
4. Declared winners: how many survive a multiplicity-corrected test?
Writes outputs/experiments.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)

frames = []
for split in ("exploratory", "confirmatory", "holdout"):
    d = pd.read_csv(ROOT / f"data/upworthy-archive-{split}-packages-03.12.2020.csv", low_memory=False)
    d["split"] = split
    frames.append(d)
df = pd.concat(frames, ignore_index=True).rename(columns={"clickability_test_id": "test_id"})
n_all_tests = df.test_id.nunique()
# The archive's authors report tests created 2013-06-25 to 2014-01-10 were not fully randomized
# (a web cache served one arm at a time). Exclude them.
NOT_RANDOMIZED = ("2013-06-25", "2014-01-11")   # [start, end)
created = pd.to_datetime(df.created_at, format="ISO8601")
df = df[~((created >= NOT_RANDOMIZED[0]) & (created < NOT_RANDOMIZED[1]))]
df = df[(df.impressions > 0) & (df.clicks >= 0) & (df.clicks <= df.impressions)].copy()
df["ctr"] = df.clicks / df.impressions
for c in ("headline", "eyecatcher_id", "excerpt", "lede"):
    df[c] = df[c].fillna("")
df["arms"] = df.groupby("test_id").test_id.transform("size")
df = df[df.arms >= 2]
R = {"archive_tests": int(n_all_tests), "excluded_not_randomized": int(n_all_tests - df.test_id.nunique()),
     "tests": int(df.test_id.nunique()), "variants": int(len(df)),
     "impressions": int(df.impressions.sum()), "clicks": int(df.clicks.sum()),
     "ctr": float(df.clicks.sum() / df.impressions.sum())}


def ztest(c1, n1, c2, n2):
    p = (c1 + c2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = np.where(se > 0, (c1 / n1 - c2 / n2) / np.where(se > 0, se, 1), 0.0)
    return z, 2 * stats.norm.sf(np.abs(z))


# ---- 1. Built-in A/A tests ------------------------------------------------------------
key = ["test_id", "headline", "eyecatcher_id", "excerpt", "lede"]
df["copy_no"] = df.groupby(key).cumcount()
df["copies"] = df.groupby(key).test_id.transform("size")
dup = df[df.copies >= 2]
a = dup[dup.copy_no == 0].set_index(key)
b = dup[dup.copy_no == 1].set_index(key)
pairs = a[["clicks", "impressions"]].join(b[["clicks", "impressions"]], rsuffix="_b").reset_index()
z, p = ztest(pairs.clicks.values, pairs.impressions.values, pairs.clicks_b.values, pairs.impressions_b.values)
pairs["p"] = p
# lift of the better copy over the worse one: pure noise by construction
hi = np.maximum(pairs.clicks / pairs.impressions, pairs.clicks_b / pairs.impressions_b)
lo = np.minimum(pairs.clicks / pairs.impressions, pairs.clicks_b / pairs.impressions_b)
ok = lo > 0
aa_lift = (hi[ok] / lo[ok] - 1)
R["aa"] = {"pairs": int(len(pairs)), "tests": int(pairs.test_id.nunique()),
           "fpr_05": float((pairs.p < 0.05).mean()),
           "ks_uniform_p": float(stats.kstest(pairs.p, "uniform").pvalue),
           "median_noise_lift": float(aa_lift.median()),
           "share_noise_lift_over_10pct": float((aa_lift > 0.10).mean()),
           "p_hist": np.histogram(pairs.p, bins=20, range=(0, 1))[0].tolist()}
# Variance-inflation factor: under a calibrated test z^2 ~ chi2(1), whose median is 0.455.
# A robust (median-based) estimate of how much wider the real noise is than the binomial model says.
z2 = z ** 2
PHI = float(np.median(z2) / stats.chi2.ppf(0.5, 1))
p_cal = 2 * stats.norm.sf(np.abs(z) / np.sqrt(PHI))
R["aa"].update({"phi": PHI, "phi_mean_based": float(z2.mean()),
                "fpr_05_recalibrated": float((p_cal < 0.05).mean()),
                "p_hist_recalibrated": np.histogram(p_cal, bins=20, range=(0, 1))[0].tolist()})
q = pd.qcut(pairs.impressions, 4, labels=["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"])
R["aa"]["fpr_by_traffic"] = {str(k): float(v) for k, v in (pairs.p < 0.05).groupby(q, observed=True).mean().items()}

# ---- 2. Power of a typical test -------------------------------------------------------
t = df.groupby("test_id").agg(arms=("ctr", "size"), imp=("impressions", "sum"),
                              med_arm=("impressions", "median"), clicks=("clicks", "sum"))
t["base"] = t.clicks / t.imp
za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
# relative MDE for a two-arm comparison with med_arm impressions per arm
t["mde_rel"] = (za + zb) * np.sqrt(PHI * 2 * t.base * (1 - t.base) / t.med_arm) / t.base
t = t[np.isfinite(t.mde_rel)]
R["power"] = {"median_variant_impressions": float(t.med_arm.median()),
              "median_test_impressions": float(t.imp.median()),
              "median_mde_rel": float(t.mde_rel.median()),
              "share_can_detect_10pct": float((t.mde_rel <= 0.10).mean()),
              "share_can_detect_20pct": float((t.mde_rel <= 0.20).mean()),
              "share_can_detect_50pct": float((t.mde_rel <= 0.50).mean()),
              "mde_hist": np.histogram(np.clip(t.mde_rel, 0, 2), bins=40, range=(0, 2))[0].tolist()}
# impressions per variant needed to detect a 10% relative lift at the median CTR
pbar = float(t.base.median())
R["power"]["median_ctr"] = pbar
R["power"]["n_per_arm_for_10pct"] = float(PHI * ((za + zb) ** 2) * 2 * pbar * (1 - pbar) / (0.10 * pbar) ** 2)

# ---- 3. Winner's curse: empirical-Bayes shrinkage of variant effects --------------------
# effect = variant log-odds minus its test's pooled log-odds; method-of-moments estimate of
# true-effect variance tau^2 = var(observed effects) - mean(sampling variance)
df["lo"] = np.log((df.clicks + 0.5) / (df.impressions - df.clicks + 0.5))
df["se2"] = PHI * (1 / (df.clicks + 0.5) + 1 / (df.impressions - df.clicks + 0.5))
pool = df.groupby("test_id").agg(c=("clicks", "sum"), n=("impressions", "sum"))
pool["plo"] = np.log((pool.c + 0.5) / (pool.n - pool.c + 0.5))
df = df.join(pool.plo, on="test_id")
df["eff"] = df.lo - df.plo
nondup = df[df.copies == 1]
tau2 = max(float(nondup.eff.var() - nondup.se2.mean()), 1e-6)
df["shrink"] = tau2 / (tau2 + df.se2)
df["eff_eb"] = df.eff * df.shrink
best = df.loc[df.groupby("test_id").ctr.idxmax()]
best = best[best.ctr > 0]
R["winners_curse"] = {"tau": float(np.sqrt(tau2)),
                      "median_observed_lift": float(np.expm1(best.eff).median()),
                      "median_eb_lift": float(np.expm1(best.eff_eb).median()),
                      "median_shrink": float(best.shrink.median())}
R["winners_curse"]["obs_hist"] = np.histogram(np.clip(np.expm1(best.eff), 0, 1.5), bins=30, range=(0, 1.5))[0].tolist()
R["winners_curse"]["eb_hist"] = np.histogram(np.clip(np.expm1(best.eff_eb), 0, 1.5), bins=30, range=(0, 1.5))[0].tolist()
R["winners_curse"]["share_of_lift_that_is_noise"] = 1 - (
    R["winners_curse"]["median_eb_lift"] / R["winners_curse"]["median_observed_lift"])

# ---- 4. Declared winners vs a corrected test -----------------------------------------
w = df[df.winner == True]
w = w[~w.test_id.duplicated(keep=False)]           # tests with exactly one declared winner
rest = df.groupby("test_id").agg(c=("clicks", "sum"), n=("impressions", "sum"))
w = w.join(rest, on="test_id")
c_ot, n_ot = w.c - w.clicks, w.n - w.impressions
zw, pw = ztest(w.clicks.values, w.impressions.values, c_ot.values, n_ot.values)
R_naive = float((np.minimum(np.where(zw > 0, pw / 2, 1 - pw / 2) * w.arms.values, 1) < 0.05).mean())
zw = zw / np.sqrt(PHI)
pw = 2 * stats.norm.sf(np.abs(zw))
# one-sided (winner better than the rest), Bonferroni across the k arms that could have won
p_one = np.where(zw > 0, pw / 2, 1 - pw / 2)
p_adj = np.minimum(p_one * w.arms.values, 1)
R["declared_winners"] = {"tests": int(len(w)),
                         "share_tests_with_winner": float(len(w) / df.test_id.nunique()),
                         "survive_bonferroni_only": R_naive,
                         "survive_corrected": float((p_adj < 0.05).mean())}

(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs/experiments.json").write_text(json.dumps(R, indent=2))
print(json.dumps(R, indent=2))
