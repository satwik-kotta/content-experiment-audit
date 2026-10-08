# Content Experiment Audit: How Trustworthy Were 25,000 Headline A/B Tests?

An audit of the **Upworthy Research Archive**: 32,487 real headline A/B tests run on a large content site from 2013 to 2015. The archive's authors report that tests created between 25 June 2013 and 10 January 2014 were not fully randomized, so those 7,017 tests are excluded. That leaves **25,470 randomized tests**, 122,381 variants, **426M impressions** and 5.6M clicks.

**Live dashboard:** https://satwik-kotta.github.io/content-experiment-audit/dashboard/

## Why this matters
Content platforms run thousands of small experiments to decide which headline, image or layout people see. This project asks two questions an experimentation team cares about:
1. **Can we trust the platform's test results?** It looks at false positives, statistical power and inflated winners.
2. **What actually makes content get clicked?** It finds headline effects in one half of the data and then checks them on a separate half.

## Key findings
| | Result |
|---|---|
| Built-in A/A tests | 2,815 pairs of **identical** variants inside the same test. A standard z-test flags **13.2%** of them as different, against 5% expected. |
| Why | The noise is 1.56× wider than the binomial model assumes, and it gets worse with traffic (false positives rise from 8.0% to 19.6% across traffic quartiles). Inflating the variance by 1.56 brings false positives down to 6.6%. |
| Noise alone | Two identical variants differ in CTR by a median of **20%**. |
| Power | A typical variant got 3,083 impressions. The median test could only detect an **80% relative lift**, and 8% of tests could detect 50%. Detecting a 10% lift at a 1.2% CTR needs about 210K impressions per variant. |
| Winner's curse | The best variant's median observed lift is +31%. After empirical-Bayes shrinkage it is +13%, so **60% of the apparent win is noise**. |
| Declared winners | 5,949 tests (23%) had a declared winner. Only **18%** of those survive a test that corrects for multiple variants and the extra noise (30% with the multiple-variant correction alone). |

### What makes a headline get clicked
Variants inside a test are randomized, so comparing headlines that share the same image inside the same test gives a causal estimate. The model is weighted least squares on log-odds CTR with test × image fixed effects and test-clustered, variance-inflated standard errors.

Features were **discovered on the exploratory split** (2,000 tests) and **confirmed on the separate confirmatory split** (9,628 tests, 157M impressions). Every feature that was significant in exploration replicated, and every null stayed null.

| Headline feature | Share of headlines | Exploratory | Confirmatory (95% CI) | Result |
|---|---|---|---|---|
| Contains a number | 18% | +5.3% | +6.1% [+4.6, +7.6] | Confirmed |
| Says 'this/these' | 25% | +4.1% | +5.1% [+4.0, +6.3] | Confirmed |
| 15+ words | 60% | +5.2% | +4.3% [+3.4, +5.3] | Confirmed |
| Says 'you/your' | 25% | +0.1% | +0.8% [-0.3, +1.9] | No effect |
| Negative word (no, never, worst…) | 18% | +0.1% | +0.6% [-0.6, +1.7] | No effect |
| Quotation | 1% | +1.0% | -3.2% [-7.5, +1.4] | No effect |
| Exclamation mark | 2% | -11.7% | -6.0% [-8.8, -3.1] | Confirmed |
| Asks a question | 17% | -8.7% | -8.3% [-9.4, -7.2] | Confirmed |

## Recommendations for an experimentation platform
1. **Calibrate with A/A tests.** Run identical variants continuously and inflate the variance until false positives sit at 5%. A plain binomial z-test is overconfident here.
2. **Size tests before launch.** Detecting realistic effects of 5–10% needs over 10× the traffic a typical test received. Pool traffic or test fewer, bolder variants.
3. **Shrink winners.** Report empirical-Bayes estimates, not raw winning lifts, so roadmaps aren't built on inflated numbers.
4. **Correct for multiple variants.** Picking the best of about 4 variants without correction declares too many winners.
5. **Content guidance:** headlines with numbers and concrete "this/these" framing do better. Questions and exclamation marks do worse.

## Method
```
CSV → PostgreSQL (load, quality checks, scale profile, within-test window functions)
    → Python: A/A calibration → variance inflation → power/MDE → empirical-Bayes shrinkage
    → corrected re-test of declared winners → discover/confirm headline model → ECharts dashboard
```
- **A/A pairs:** variants with identical headline, image, excerpt and lede inside the same test, comparing the first two copies.
- **Variance inflation (φ = 1.56):** the median of z² across A/A pairs divided by the χ²(1) median of 0.455. This is robust to a few extreme pairs. The mean-based estimate is 1.88.
- **Empirical Bayes:** each variant's effect is its log-odds minus the test's pooled log-odds. The true-effect variance is τ² = Var(observed) − mean(sampling variance), giving τ = 0.168. The shrinkage factor is τ² / (τ² + se²).
- **Declared winners:** each winner is tested one-sided against the pooled other variants, with a Bonferroni correction for the number of variants and the variance inflation applied.

## Limitations
- A single φ under-corrects the largest tests, where noise grows fastest; recalibrated false positives are 6.6%, not exactly 5%. A beta-binomial or time-blocked variance model would fit better.
- Following the archive's authors, all results are intent-to-treat: some variants appeared below the fold.
- Headline effects are averages across Upworthy's 2013–2015 audience and may not transfer to other sites or eras.
- Impressions are not independent page views over time, so day-of-week and traffic-source shifts within a test are a likely cause of the extra noise.

## Run it
```bash
pip install -r requirements.txt
createdb content && ./run_sql.sh content     # PostgreSQL profile (from repo root)
python analysis/experiments.py               # A/A, power, winner's curse, declared winners
python analysis/headlines.py                 # discover/confirm headline model
python analysis/build_dashboard.py           # -> dashboard/index.html
```

## Data and citation
Matias, J.N., Munger, K., Aubin Le Quere, M. & Ebersole, C. *The Upworthy Research Archive, a time series of 32,487 experiments in U.S. media.* **Scientific Data** 8, 195 (2021). The data is the public archive (exploratory, confirmatory and holdout package files).
