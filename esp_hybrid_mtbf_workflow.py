"""
Hybrid censoring-aware Weibull / Kaplan-Meier MTBF workflow for ESP run-life data.

Corrected version of the original per-well script. Fixes:
  1. MTTF used stats.gamma(1 + 1/shape).mean()  -> that is the MEAN OF A GAMMA
     DISTRIBUTION (= 1 + 1/shape), not the Gamma FUNCTION. Correct: scipy.special.gamma.
  2. AD test used stats.anderson_ksamp([failure_times, cdf_values]) -> a TWO-SAMPLE test
     comparing days (hundreds) with probabilities (0-1); its p-value is also clipped to
     [0.001, 0.25]. Replaced with a one-sample AD statistic and a parametric-bootstrap
     p-value (scipy.stats.goodness_of_fit), valid when parameters are estimated.
  3. Every run was treated as an ESP failure. Non-ESP pulls (P.W.O.R, S.F.W.O., tubing
     leak, ...) are now right-censored and fitted with the full censored likelihood.
  4. Adds the Kaplan-Meier cross-check (D_cross vs D_crit) and Chi-square bin merging
     (E_i >= 5). Fits are pooled at fleet level; per-well fits with few runs are flagged.

Usage:
  python esp_hybrid_mtbf_workflow.py "path/to/Wells2/Leftover 1"
"""
import sys, glob, os, re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn

RUN_LIFE_COL = 9          # 'RUN LIFE.1' (same column the original script used)
REASON_COL = "SHUT DOWN REASON"
SHUTDOWN_DATE_COL = "SHUT DOWN DATE"

# Shutdown reasons treated as ESP-related failures. Everything else (e.g. P.W.O.R,
# S.F.W.O., TBG LEAK, high water cut) is treated as a censored (non-ESP) pull.
# >>> Review this list against field definitions before final use. <<<
ESP_FAILURE_PATTERN = (r"SHORT|OVER ?LOAD|UNDER ?CURRENT|UNDER ?LOAD|SHAFT|BURN|GROUND|PHASE|"
                       r"SEAL|MOTOR|CABLE|PENETRATOR|POT ?HEAD|STUCK|LOCKED|NO FLOW|INSULATION|EARTH")
MIN_RUNS_PER_WELL = 10    # per-well Weibull fits below this are unreliable


# ----------------------------------------------------------------- data loading
def load_runs(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.xlsx")) +
                   glob.glob(os.path.join(folder, "Leftover 2", "*.xlsx")))
    frames = []
    for f in files:
        df = pd.read_excel(f, sheet_name=0)
        out = pd.DataFrame({
            "well": os.path.basename(f)[:-5],
            "run_life": pd.to_numeric(df.iloc[:, RUN_LIFE_COL], errors="coerce"),
            "reason": df.get(REASON_COL, pd.Series(dtype=str)).fillna("").astype(str).str.upper(),
            "shutdown": df.get(SHUTDOWN_DATE_COL),
        })
        frames.append(out)
    runs = pd.concat(frames, ignore_index=True)
    runs = runs[runs.run_life > 0].copy()
    esp_fail = runs.reason.str.contains(ESP_FAILURE_PATTERN, regex=True)
    still_running = runs.shutdown.isna()
    runs["event"] = (esp_fail & ~still_running).astype(int)   # 1 = ESP failure, 0 = censored
    return runs


# ----------------------------------------------------------------- estimators
def _neg_loglik(params, t, e):
    beta, eta = params
    if beta <= 0 or eta <= 0:
        return 1e12
    return -(stats.weibull_min.logpdf(t[e == 1], beta, 0, eta).sum() +
             stats.weibull_min.logsf(t[e == 0], beta, 0, eta).sum())

def fit_censored_weibull(t, e):
    """Full censored-likelihood MLE (Eq. 2a)."""
    b0, _, h0 = stats.weibull_min.fit(t[e == 1], floc=0)
    res = minimize(_neg_loglik, [b0, h0], args=(t, e), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
    return res.x

def mttf(beta, eta):
    return eta * gamma_fn(1 + 1 / beta)          # Gamma FUNCTION (bug fix #1)

def kaplan_meier(t, e):
    order = np.argsort(t)
    ts, es = t[order], e[order]
    n, s, kt, ks = len(t), 1.0, [0.0], [1.0]
    for i, (ti, ei) in enumerate(zip(ts, es)):
        if ei:
            s *= 1 - 1 / (n - i)
        kt.append(ti); ks.append(s)
    return np.array(kt), np.array(ks)

def cross_check(beta, eta, t, e):
    kt, ks = kaplan_meier(t, e)
    d_cross = np.max(np.abs(np.exp(-(kt / eta) ** beta) - ks))
    d_crit = 1.36 / np.sqrt(e.sum())
    return d_cross, d_crit, d_cross <= d_crit


# ----------------------------------------------------------------- goodness of fit
def gof_failed_units(failed, n_boot=999, seed=0):
    """One-sample AD and KS on failed units against THEIR OWN Weibull fit.
    Under heavy censoring, the KM cross-check is the primary adequacy test; failed units
    are a biased subsample and must not be tested against the censored-MLE CDF."""
    rng = np.random.default_rng(seed)
    ad = stats.goodness_of_fit(stats.weibull_min, failed, known_params={"loc": 0},
                               statistic="ad", n_mc_samples=n_boot, random_state=rng)
    ks = stats.goodness_of_fit(stats.weibull_min, failed, known_params={"loc": 0},
                               statistic="ks", n_mc_samples=n_boot, random_state=rng)
    return ad.statistic, ad.pvalue, ks.statistic, ks.pvalue

def chi_square_merged(failed, beta, eta, min_expected=5):
    n = len(failed)
    q75, q25 = np.percentile(failed, [75, 25])
    nbins = max(3, int(np.ceil((failed.max() - failed.min()) / (2 * (q75 - q25) / n ** (1 / 3)))))
    obs, edges = np.histogram(failed, bins=nbins)
    exp = n * np.diff(stats.weibull_min.cdf(edges, beta, 0, eta))
    obs, exp = list(obs), list(exp)
    while min(exp) < min_expected and len(exp) > 2:
        j = int(np.argmin(exp))
        k = j + 1 if j == 0 else (j - 1 if j == len(exp) - 1 else (j - 1 if exp[j - 1] <= exp[j + 1] else j + 1))
        exp[k] += exp[j]; obs[k] += obs[j]; del exp[j]; del obs[j]
    obs, exp = np.array(obs), np.array(exp)
    chi2 = ((obs - exp) ** 2 / exp).sum()
    dof = len(exp) - 3
    return chi2, dof, (stats.chi2.sf(chi2, dof) if dof > 0 else np.nan), len(exp)


# ----------------------------------------------------------------- workflow
def analyse(t, e, label):
    t, e = np.asarray(t, float), np.asarray(e, int)
    failed = t[e == 1]
    b_fo, _, h_fo = stats.weibull_min.fit(failed, floc=0)
    b_nv, _, h_nv = stats.weibull_min.fit(t, floc=0)
    beta, eta = fit_censored_weibull(t, e)
    d_cross, d_crit, passed = cross_check(beta, eta, t, e)
    ad, ad_p, ks, ks_p = gof_failed_units(failed)
    chi2, dof, chi_p, nb = chi_square_merged(failed, b_fo, h_fo)
    out = {
        "label": label, "n_runs": len(t), "n_failed": int(e.sum()), "n_censored": int((e == 0).sum()),
        "beta": beta, "eta": eta, "MTBF": mttf(beta, eta),
        "MTBF_failed_only": mttf(b_fo, h_fo), "MTBF_censored_as_failed": mttf(b_nv, h_nv),
        "D_cross": d_cross, "D_crit": d_crit, "cross_check_passed": passed,
        "AD_failed": ad, "AD_p": ad_p, "KS_failed": ks, "KS_p": ks_p,
        "Chi2": chi2, "Chi2_dof": dof, "Chi2_p": chi_p, "Chi2_bins": nb,
    }
    for x in (180, 365, 700, 1000):
        out[f"F({x})"] = stats.weibull_min.cdf(x, beta, 0, eta)
    return out


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "Wells2/Leftover 1"
    runs = load_runs(folder)

    fleet = analyse(runs.run_life, runs.event, "FLEET")
    print(pd.Series(fleet).to_string())

    per_well = []
    for well, g in runs.groupby("well"):
        if g.event.sum() >= MIN_RUNS_PER_WELL:
            per_well.append(analyse(g.run_life, g.event, well))
    print(f"\nWells with >= {MIN_RUNS_PER_WELL} ESP failures (reliable per-well fit): {len(per_well)}")

    with pd.ExcelWriter("hybrid_mtbf_results.xlsx") as w:
        pd.DataFrame([fleet]).to_excel(w, sheet_name="Fleet", index=False)
        pd.DataFrame(per_well).to_excel(w, sheet_name="Per-well (n>=10)", index=False)
        runs.to_excel(w, sheet_name="Runs (classified)", index=False)
