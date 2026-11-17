#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
toll_fairness_ridesharing.py
============================

Single-file, self-contained replication code for

    "Who Pays the Toll? Fairness, Platform Pre-Commitment, and the Governance
     of Ex Post Opportunism in Long-Distance Ridesharing"
     (manuscript prepared for Transport Policy)

The script contains four parts, all runnable from one command:

    Part A - Theory   : Lemma 1, Propositions 1-5, the closed-form penalty
                        threshold F* = (1-q)(k - rho*t)/q, and numerical
                        verification of every analytical claim.
    Part B - Survey   : audit of the stated-preference table. Recomputes the
                        willingness-to-pay implied by the reported mixed-logit
                        coefficients, the likelihood-ratio test against the
                        conditional logit, and the sign pattern of H1-H4.
    Part C - ABM      : agent-based simulation of the negotiation regime and
                        the pre-commitment regime (rho = 1 and rho = 0.5),
                        with common random numbers, base-fare neutralisation,
                        and the penalty / detection sensitivity grid.
    Part D - Figures  : equilibrium payment vs. sunk cost, penalty threshold
                        vs. sunk cost, regime comparison, and the penalty
                        response curve.

Usage
-----
    python toll_fairness_ridesharing.py --mode all
    python toll_fairness_ridesharing.py --mode theory
    python toll_fairness_ridesharing.py --mode survey
    python toll_fairness_ridesharing.py --mode simulate --replications 30
    python toll_fairness_ridesharing.py --mode check        # self-tests
    python toll_fairness_ridesharing.py --mode all --quick   # fast smoke run

    python toll_fairness_ridesharing.py --help               # all options

Outputs (written to --outdir, default ./outputs):
    tableA_theory_checks.csv      Part A: closed-form vs. numerical checks
    tableB_survey_audit.csv       Part B: derived survey quantities
    tableC_simulation.csv         Part C: Table 5 of the paper
    tableD_sensitivity.csv        Part C: Table 6 of the paper
    fig1_payment_vs_sunk_cost.png
    fig2_penalty_threshold.png
    fig3_regime_comparison.png
    fig4_penalty_response.png
    report.md                     human-readable run report
    run_config.json               exact configuration of the run

Requirements
------------
    Python >= 3.9, numpy. matplotlib is needed only for the figures; if it is
    missing the script still writes every CSV and the report.

Model-to-code map (paper -> this file)
--------------------------------------
    Lemma 1                    -> lemma1_best_demand()
    Proposition 1              -> prop1_negotiation()
    Proposition 2              -> prop2_precommitment()
    Proposition 4 (welfare)    -> prop4_welfare()
    Proposition 5 (penalty)    -> penalty_threshold(), prop5_penalty_check()
    eq. (3) mixed logit / WTP  -> survey_audit()
    Section 5.2 ABM            -> build_market(), simulate_negotiation(),
                                  simulate_precommitment(), run_replication()
    Table 5                    -> table5_results()
    Table 6                    -> table6_sensitivity()

Author: replication code written for the manuscript; MIT licence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SQRT2 = math.sqrt(2.0)


# =============================================================================
# SECTION 0 - small statistical helpers (no scipy dependency)
# =============================================================================


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / SQRT2)


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, |err| < 1e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def chi2_sf_df3(x: float) -> float:
    """Survival function of chi-square with 3 degrees of freedom (closed form)."""
    if x <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0)) + math.sqrt(2.0 * x / math.pi) * math.exp(-x / 2.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# =============================================================================
# SECTION 1 - THEORY (Lemma 1, Propositions 1, 2, 4, 5)
# =============================================================================


@dataclass(frozen=True)
class TheoryPrimitives:
    """Stylised primitives used for the analytical checks.

    v          gross trip value
    toll       highway toll t
    time_value value of the time saved by the tolled route, s > t
    c_low      low driver cost c_L
    c_high     high driver cost c_H
    p_low      prior probability of the low-cost type, lambda
    sunk       passenger termination cost k (the "sunk" outside option)
    rho        passenger share of the toll under pre-commitment
    q          detection probability
    """

    v: float = 10.0
    toll: float = 1.0
    time_value: float = 1.45   # value of the time saved by the tolled route, s > t
    c_low: float = 0.36
    c_high: float = 0.92
    p_low: float = 0.70
    sunk: float = 1.60
    rho: float = 1.0
    q: float = 0.80

    @property
    def rho_t(self) -> float:
        return self.rho * self.toll


def lemma1_best_demand(k: float, c: float) -> Tuple[Optional[float], float]:
    """Lemma 1 (opportunistic demand).

    Under negotiation the driver solves max_{d>=0} 1{d <= k} (d - c).
    Returns (demand, payoff); demand is None when the driver rejects.
    """
    if k > c:
        return k, k - c
    return None, 0.0


def prop1_negotiation(p: TheoryPrimitives, c: float) -> Dict[str, float]:
    """Proposition 1 (equilibrium under negotiation).

    Accept iff k > c; demand d = k; passenger pays; payoffs (v - k, k - c).
    The declared preference w is irrelevant (cheap talk).
    """
    d, payoff = lemma1_best_demand(p.sunk, c)
    accepts = d is not None
    return {
        "accept": float(accepts),
        "demand": d if accepts else 0.0,
        "passenger_payoff": p.v - p.sunk if accepts else 0.0,
        "driver_payoff": payoff if accepts else 0.0,
    }


def prop2_precommitment(p: TheoryPrimitives, c: float, penalty: float) -> Dict[str, float]:
    """Proposition 2 (equilibrium under pre-commitment).

    Acceptance iff rho*t >= c (equivalently b + rho*t >= c with base fare b).
    An accepting driver complies. A detected deviator still collects the rule
    payment but pays the penalty, so his payoff is rho*t - c - F; an undetected
    deviator collects k - c. Compliance therefore dominates iff
    (1-q)(k - rho*t) < q*F, which is the condition used for F*.
    """
    accepts = p.rho_t >= c
    compliance = p.rho_t - c
    detected = p.rho_t - c - penalty
    undetected = p.sunk - c
    deviation = p.q * detected + (1.0 - p.q) * undetected
    return {
        "accept": float(accepts),
        "payment": p.rho_t if accepts else 0.0,
        "driver_payoff": compliance if accepts else 0.0,
        "deviation_payoff": deviation,
        "deviation_gap": deviation - compliance,
        "required_penalty": penalty_threshold(p.sunk, p.rho_t, p.q),
        "complies": float(deviation <= compliance),
    }


def prop4_welfare(p: TheoryPrimitives) -> Dict[str, float]:
    """Proposition 4 (welfare comparison).

    M_N = {c : k > c}, M_C = {c : rho*t >= c}. With two types and prior lambda,
    U_P^r = Pr(M_r) * (v - p_r), p_N = k, p_C = rho*t.
    """
    types = ((p.c_low, p.p_low), (p.c_high, 1.0 - p.p_low))
    pr_n = sum(pr for c, pr in types if p.sunk > c)
    pr_c = sum(pr for c, pr in types if p.rho_t >= c)
    u_n = pr_n * (p.v - p.sunk)
    u_c = pr_c * (p.v - p.rho_t)
    return {
        "match_rate_negotiation": pr_n,
        "match_rate_precommitment": pr_c,
        "passenger_utility_negotiation": u_n,
        "passenger_utility_precommitment": u_c,
        "utility_gain": u_c - u_n,
        "driver_toll_rent_precommitment": max(p.rho_t - p.toll, 0.0),
    }


def penalty_threshold(k: float, rho_t: float, q: float) -> float:
    """Closed-form penalty threshold F* = (1-q)(k - rho*t)/q (eq. in Section 5).

    Deviation payoff: q(-c - F) + (1-q)(k - c); compliance: rho*t - c.
    """
    return (1.0 - q) * (k - rho_t) / q


def prop5_penalty_check(k: float, rho_t: float, q: float, c: float = 0.0) -> Dict[str, float]:
    """Proposition 5: deterrence threshold, and its monotonicity in k and q.

    Deviation gain over compliance is (1-q)(k - rho*t) - q*F; it is non-positive
    exactly when F >= F* = (1-q)(k - rho*t)/q.
    """
    f_star = penalty_threshold(k, rho_t, q)
    gain_at_threshold = (1.0 - q) * (k - rho_t) - q * f_star
    return {
        "F_star": f_star,
        "deviation_gain_at_Fstar": gain_at_threshold,
        "gap": gain_at_threshold,          # should be zero at F = F*
        "dFstar_dk": (1.0 - q) / q,
        "dFstar_dq": -1.0 * (k - rho_t) / (q * q),
    }


def theory_analysis(p: TheoryPrimitives) -> List[Dict[str, float]]:
    """Run the analytical checks and return a list of named rows."""
    rows: List[Dict[str, float]] = []

    # --- Lemma 1: verify by brute force that d = k is optimal ---------------
    demand_grid = np.linspace(0.0, 3.0, 301)
    max_violation = 0.0
    n_cases = 0
    for k in (0.4, 0.9, 1.3, 2.0, 2.6):
        for c in (0.2, 0.5, 1.0, 1.6, 2.4):
            n_cases += 1
            best = max(float(np.max(np.where(demand_grid <= k, demand_grid - c, -np.inf))), 0.0)
            analytic = lemma1_best_demand(k, c)[1]
            max_violation = max(max_violation, abs(best - analytic))
    rows.append({
        "check": "Lemma 1: demand d=k is optimal",
        "value": max_violation,
        "benchmark": 0.0,
        "pass": float(max_violation < 1e-9),
        "detail": f"{n_cases} (k,c) cases on a 301-point demand grid",
    })

    # --- Proposition 1: equilibrium payoffs ---------------------------------
    neg = prop1_negotiation(p, p.c_high)
    low = prop1_negotiation(p, p.c_low)
    rows.append({
        "check": "Prop 1: payment equals k for both types",
        "value": neg["demand"],
        "benchmark": p.sunk,
        "pass": float(abs(neg["demand"] - p.sunk) < 1e-12 and low["demand"] == p.sunk),
        "detail": "w has no effect; d = k for c_L and c_H",
    })

    # --- Proposition 2: compliance is optimal at F > 0 ----------------------
    pen = 0.25
    res = prop2_precommitment(p, p.c_low, pen)
    expected_gap = (1.0 - p.q) * (p.sunk - p.rho_t) - p.q * pen
    rows.append({
        "check": "Prop 2: deviation gain below compliance",
        "value": res["deviation_gap"],
        "benchmark": expected_gap,
        "pass": float(abs(res["deviation_gap"] - expected_gap) < 1e-12
                      and res["deviation_gap"] < 0.0),
        "detail": f"F={pen} > F*={penalty_threshold(p.sunk, p.rho_t, p.q):.3f}; "
                  f"gain=(1-q)(k-rho t)-qF",
    })

    # --- Proposition 5: F* reproduces the deterrence margin ----------------
    worst_gap = 0.0
    for k in np.linspace(1.1, 3.0, 12):
        for q in (0.30, 0.55, 0.80, 1.00):
            for rho_t in (0.5, 1.0, 1.5):
                if k <= rho_t:
                    continue
                chk = prop5_penalty_check(k, rho_t, q)
                worst_gap = max(worst_gap, abs(chk["gap"]))
    rows.append({
        "check": "Prop 5: F* sets deviation gain to zero",
        "value": worst_gap,
        "benchmark": 0.0,
        "pass": float(worst_gap < 1e-12),
        "detail": "grid over k in [1.1,3], q in {.3,.55,.8,1}, rho*t in {.5,1,1.5}",
    })

    # monotonicity in k and q
    f_base = penalty_threshold(2.0, 1.0, 0.8)
    f_high_k = penalty_threshold(2.5, 1.0, 0.8)
    f_low_q = penalty_threshold(2.0, 1.0, 0.6)
    rows.append({
        "check": "Prop 5: F* rises in k, falls in q",
        "value": float((f_high_k > f_base) and (f_low_q > f_base)),
        "benchmark": 1.0,
        "pass": float((f_high_k > f_base) and (f_low_q > f_base)),
        "detail": f"F*(2,.8)={f_base:.3f}, F*(2.5,.8)={f_high_k:.3f}, "
                  f"F*(2,.6)={f_low_q:.3f}",
    })

    # --- Proposition 4: welfare comparison ---------------------------------
    welfare_rows = []
    for k in (1.1, 1.3, 1.6, 2.0):
        q_p = TheoryPrimitives(sunk=k, rho=1.0)
        welfare_rows.append((k, prop4_welfare(q_p)))

    # (i) completed-trip rate weakly higher under C whenever rho*t >= c_H and k < c_H
    ok_i = True
    for q_p in (TheoryPrimitives(sunk=0.9, rho=1.0), TheoryPrimitives(sunk=1.1, rho=1.0)):
        w = prop4_welfare(q_p)
        if w["match_rate_precommitment"] < w["match_rate_negotiation"]:
            ok_i = False
    rows.append({
        "check": "Prop 4(i): match rate weakly higher under C",
        "value": float(ok_i),
        "benchmark": 1.0,
        "pass": float(ok_i),
        "detail": "rho*t >= c_H, k below c_H",
    })

    # (ii) utility comparison when k > rho*t and Pr(M_C) >= Pr(M_N)
    ok_ii = all((w["utility_gain"] > 0) for k, w in welfare_rows
                if k > 1.0 and w["match_rate_precommitment"] >= w["match_rate_negotiation"])
    rows.append({
        "check": "Prop 4(ii): passenger utility higher under C",
        "value": float(ok_ii),
        "benchmark": 1.0,
        "pass": float(ok_ii),
        "detail": "; ".join(f"k={k:.1f}: {w['utility_gain']:+.2f}"
                            for k, w in welfare_rows),
    })

    # (iii) rho = 1 gives zero driver toll rent
    w_full = prop4_welfare(TheoryPrimitives(rho=1.0))
    rows.append({
        "check": "Prop 4(iii): zero driver toll rent at rho=1",
        "value": w_full["driver_toll_rent_precommitment"],
        "benchmark": 0.0,
        "pass": float(w_full["driver_toll_rent_precommitment"] == 0.0),
        "detail": "payment equals the toll exactly",
    })
    return rows


# =============================================================================
# SECTION 2 - SURVEY AUDIT (stated-preference table)
# =============================================================================


# Coefficients exactly as reported in the manuscript's mixed-logit table.
SURVEY_ESTIMATES = {
    "price_alpha": -1.42,
    "price_alpha_ci": (-1.63, -1.21),
    "certainty_beta": 0.203,
    "certainty_beta_ci": (0.168, 0.238),
    "certainty_sigma": 0.117,
    "certainty_sigma_ci": (0.081, 0.153),
    "time_20min": -0.087,
    "pt_constant": -0.641,
    "certainty_x_kindex": 0.061,
    "certainty_x_tollshare": -0.047,
    "certainty_x_fairness": 0.038,
    "ll_mxl": -2987.2,
    "ll_mnl": -3120.4,
    "n_obs": 5400,
    "n_respondents": 900,
    "reference_fare_cny": 102.5,   # midpoint of the CNY 90-115 reference band
    "toll_cny": 60.0,
}


def survey_audit(est: Dict[str, float] = SURVEY_ESTIMATES) -> List[Dict[str, str]]:
    """Recompute every derived quantity that the paper reports from its table.

    Returns a list of rows: quantity, value, reported value, consistency flag.
    """
    alpha = est["price_alpha"]
    beta = est["certainty_beta"]
    sigma = est["certainty_sigma"]

    wtp_share = -beta / alpha                       # WTP as a share of the fare
    wtp_cny = wtp_share * est["reference_fare_cny"]
    wtp_pct_of_toll = wtp_cny / est["toll_cny"] * 100.0

    # Likelihood-ratio test: MXL (3 extra parameters) against MNL.
    lr_stat = 2.0 * (est["ll_mxl"] - est["ll_mnl"])
    lr_p = chi2_sf_df3(lr_stat)

    rows: List[Dict[str, str]] = []

    def add(name: str, value: float, reported: Any, unit: str, tol: float, note: str = ""):
        ok = reported is None or abs(value - float(reported)) <= tol
        shown = "<1e-4" if (unit == "p" and value < 1e-4) else f"{value:.4f}"
        rows.append({
            "quantity": name,
            "recomputed": shown,
            "reported": "n/a" if reported is None else f"{float(reported):.4f}",
            "unit": unit,
            "consistent": "yes" if ok else "NO",
            "note": note,
        })

    add("WTP for rule certainty (share of reference fare)", wtp_share, 0.143, "share", 0.001,
        "-beta/alpha = 0.203/1.42")
    add("WTP for rule certainty (CNY)", wtp_cny, None, "CNY", 0.0,
        "share x reference fare of CNY 102.5")
    add("WTP for rule certainty (% of the CNY 60 toll)", wtp_pct_of_toll, None, "%", 0.0,
        "monetary WTP relative to the toll")
    add("Likelihood-ratio statistic (MXL vs MNL)", lr_stat, 266.4, "chi2", 0.1,
        "2 * (LL_MNL - LL_MXL)")
    add("LR p-value (df = 3)", lr_p, None, "p", 0.0,
        "closed-form chi-square survival function, df = 3")
    add("Observations per respondent", est["n_obs"] / est["n_respondents"], 6.0, "tasks", 1e-9,
        "5,400 choice observations / 900 respondents")
    add("Random-parameter observations", est["n_obs"] / 1.0, 5400.0, "obs", 1e-9,
        "above the 200-observation rule of thumb for one random parameter")

    # Hypothesis sign pattern (H1-H4 are stated in the paper's variable table).
    sign_rows = [
        ("H1: mean certainty coefficient positive", beta, ">0", beta > 0),
        ("H2: certainty x re-planning cost positive", est["certainty_x_kindex"], ">0",
         est["certainty_x_kindex"] > 0),
        ("H3: certainty x toll share negative", est["certainty_x_tollshare"], "<0",
         est["certainty_x_tollshare"] < 0),
        ("H4: certainty x fairness concern positive", est["certainty_x_fairness"], ">0",
         est["certainty_x_fairness"] > 0),
    ]
    for name, val, wanted, ok in sign_rows:
        rows.append({
            "quantity": name,
            "recomputed": f"{val:+.3f}",
            "reported": "n/a",
            "unit": "logit coefficient",
            "consistent": "yes" if ok else "NO",
            "note": f"required sign {wanted}",
        })

    # Internal coherence: is the certainty coefficient's CI consistent with the
    # reported standard deviation of the random parameter?
    ci_half_width = 0.5 * (est["certainty_beta_ci"][1] - est["certainty_beta_ci"][0])
    implied_se = ci_half_width / 1.96
    rows.append({
        "quantity": "Implied standard error of the certainty mean",
        "recomputed": f"{implied_se:.4f}",
        "reported": "n/a",
        "unit": "logit coefficient",
        "consistent": "yes" if implied_se < sigma else "NO",
        "note": "CI half-width / 1.96; should be below sigma_beta = 0.117",
    })
    return rows


# =============================================================================
# SECTION 3 - AGENT-BASED SIMULATION (Section 5.2 and Tables 5-6)
# =============================================================================


@dataclass
class SimConfig:
    """Calibration of the agent-based simulation (Table 6 of the paper)."""

    # market size and Monte Carlo effort
    pairs: int = 2000
    periods: int = 200
    replications: int = 100
    seed: int = 2025

    # primitives (toll normalised to 1)
    toll: float = 1.0
    mu_v: float = 4.2          # log-normal gross trip value
    sigma_v: float = 0.45
    k_median: float = 1.5      # sunk cost, median in toll units
    k_cv: float = 0.60         # coefficient of variation of the sunk cost
    lambda_low: float = 0.70   # share of low-cost drivers
    c_low_mean: float = 0.35   # low-cost mean, in toll units
    c_low_sigma: float = 0.30  # log-normal shape
    c_high_mean: float = 0.65
    c_high_sigma: float = 0.35

    # matching reach: share of potential pairs for which a driver is offered
    availability: float = 0.96

    # status quo: the driver's salient ask, in toll units (drivers do not
    # observe the passenger's individual sunk cost; see the report's note)
    demand_ratio: float = 0.91
    informed_demand: bool = False   # True -> theory-consistent d = k (Lemma 1)

    # governance parameters
    q: float = 0.80             # detection probability (baseline)
    q_baseline: float = 0.80    # reference q at which F* is computed
    penalty_multiple: float = 1.5
    fstar_quantile: float = 0.95  # quantile of k used to set the reference F*
    detection_voids_trip: bool = True

    # deviation behaviour
    dev_signal_sigma: float = 0.20   # noise in the driver's belief about k
    dev_temperature: float = 0.06    # logit smoothness of the deviation choice

    # reported targets (Table 5 of the manuscript), for the comparison columns
    targets: Dict[str, float] = field(default_factory=lambda: {
        "N_completed": 0.71, "N_cancel": 0.184, "N_pax": 100.0, "N_drv": 100.0,
        "C1_completed": 0.89, "C1_cancel": 0.021, "C1_pax": 122.6, "C1_drv": 100.4,
        "C05_completed": 0.86, "C05_cancel": 0.034, "C05_pax": 114.3, "C05_drv": 99.1,
    })
    # ------------------------------------------------------------------ utils
    @property
    def sigma_k(self) -> float:
        """Log-normal shape implied by the median and the coefficient of variation."""
        return math.sqrt(math.log(1.0 + self.k_cv ** 2))

    def k_quantile(self, p: float) -> float:
        return self.k_median * math.exp(self.sigma_k * norm_ppf(p))

    def reference_penalty(self, rho_t: float) -> float:
        """F* evaluated at the reference (high-sunk-cost) quantile of k.

        The paper's closed form is pointwise in k. Because drivers face a
        heterogeneous passenger pool, the operational penalty has to be set
        against the tail of the sunk-cost distribution, where opportunism is
        profitable; fstar_quantile exposes that choice (default: 95th pct).
        """
        return penalty_threshold(self.k_quantile(self.fstar_quantile), rho_t, self.q_baseline)

    def penalty(self, rho_t: float) -> float:
        return self.penalty_multiple * self.reference_penalty(rho_t)


Market = Dict[str, np.ndarray]


def build_market(cfg: SimConfig, rng: np.random.Generator) -> Market:
    """Draw the whole panel of potential matches (periods x pairs).

    Drawing all periods at once keeps the simulation vectorised and lets every
    governance regime face exactly the same market (common random numbers).
    """
    shape = (cfg.periods, cfg.pairs)
    sigma_k = cfg.sigma_k

    v = rng.lognormal(mean=cfg.mu_v, sigma=cfg.sigma_v, size=shape)
    k = rng.lognormal(mean=math.log(cfg.k_median), sigma=sigma_k, size=shape)

    c_low = rng.lognormal(mean=math.log(cfg.c_low_mean), sigma=cfg.c_low_sigma, size=shape)
    c_high = rng.lognormal(mean=math.log(cfg.c_high_mean), sigma=cfg.c_high_sigma, size=shape)
    is_low = rng.random(shape) < cfg.lambda_low
    c = np.where(is_low, c_low, c_high)

    return {
        "v": v,
        "k": k,
        "c": c,
        "is_low": is_low,
        "avail": rng.random(shape) < cfg.availability,
        "z_signal": rng.standard_normal(shape),
        "u_dev": rng.random(shape),
        "u_detect": rng.random(shape),
    }


def _pack(mkt: Market, paid: np.ndarray, cancel: np.ndarray,
          payment: np.ndarray, penalty_paid: np.ndarray,
          base_fare: float, accepted: np.ndarray) -> Dict[str, np.ndarray]:
    """Translate outcome masks into passenger and driver payoffs."""
    # Passenger: completes -> v - (payment + base fare); cancelled -> -k; idle -> 0
    pax = np.where(paid, mkt["v"] - payment - base_fare,
                   np.where(cancel, -mkt["k"], 0.0))
    # Driver: completes -> payment + base fare - c; detected deviation -> -F; idle -> 0
    drv = np.where(paid, payment + base_fare - mkt["c"],
                   np.where(cancel, -penalty_paid, 0.0))
    return {"pax": pax, "drv": drv, "paid": paid, "cancel": cancel, "accepted": accepted}


def _summarise(out: Dict[str, np.ndarray]) -> Dict[str, float]:
    accepted = out["accepted"]
    paid = out["paid"]
    cancel = out["cancel"]
    denom = float(np.sum(accepted))
    return {
        "completed_rate": float(np.mean(paid)),
        "cancellation_rate": float(np.sum(cancel) / denom) if denom > 0 else 0.0,
        "passenger_surplus": float(np.mean(out["pax"])),
        "driver_surplus": float(np.mean(out["drv"])),
        "acceptance_rate": float(np.mean(accepted)),
    }


def simulate_negotiation(mkt: Market, cfg: SimConfig) -> Dict[str, float]:
    """Regime N: ex post negotiation (Lemma 1 / Proposition 1).

    Under the paper's frictionless version the driver knows k and demands d = k.
    In the calibrated version drivers do not observe individual sunk costs and
    post a salient ask a*t; the passenger pays iff a*t <= k and otherwise
    terminates. `informed_demand=True` switches to the pure theory version.
    """
    toll = cfg.toll
    if cfg.informed_demand:
        demand = mkt["k"]
        accepted = (mkt["k"] > mkt["c"]) & mkt["avail"]
        paid = accepted                     # d = k always acceptable
    else:
        demand = cfg.demand_ratio * toll
        # Posterior probability that a passenger accepts the ask (drivers are
        # uninformed about individual k, so acceptance is type-based).
        p_pay = 1.0 - norm_cdf(math.log(demand / cfg.k_median) / cfg.sigma_k)
        expected_profit = p_pay * demand
        accepted = (expected_profit >= mkt["c"]) & mkt["avail"]
        paid = accepted & (demand <= mkt["k"])

    cancel = accepted & ~paid
    penalty_paid = np.zeros_like(mkt["c"])
    out = _pack(mkt, paid, cancel, np.where(paid, demand, 0.0), penalty_paid,
                base_fare=0.0, accepted=accepted)
    return _summarise(out)


def simulate_precommitment(mkt: Market, cfg: SimConfig, rho: float,
                           penalty: float, q: float,
                           base_fare: float = 0.0) -> Dict[str, float]:
    """Regime C: platform pre-commitment (Proposition 2).

    The rule rho*t is locked before dispatch, acceptance binds the driver, and a
    driver who asks above the rule is detected with probability q and pays F.
    Deviation is a smoothed best response on the perceived gain

        (1-q) * (k_hat - rho*t) - q * F,     k_hat = k * exp(sigma * z).

    Detected deviations either void the booking (default, consistent with the
    dispute-cancellation rates reported in Table 5) or are simply repriced at
    the rule with the penalty applied to the driver.
    """
    rho_t = rho * cfg.toll
    accepted = ((rho_t + base_fare) >= mkt["c"]) & mkt["avail"]

    k_hat = mkt["k"] * np.exp(cfg.dev_signal_sigma * mkt["z_signal"])
    gain = (1.0 - q) * np.maximum(k_hat - rho_t, 0.0) - q * penalty
    p_dev = _sigmoid(gain / cfg.dev_temperature)
    deviate = (mkt["u_dev"] < p_dev) & accepted
    detected = deviate & (mkt["u_detect"] < q)

    ask = np.maximum(k_hat, rho_t)
    over_ask = ask > mkt["k"]

    pay_rule = accepted & ~deviate                          # compliant drivers
    pay_dev = deviate & ~detected & ~over_ask               # undetected, accepted
    cancel_dev = deviate & ~detected & over_ask             # passenger terminates

    if cfg.detection_voids_trip:
        cancel_det = detected
        pay_det = np.zeros_like(detected)
    else:
        cancel_det = np.zeros_like(detected)
        pay_det = detected

    paid = pay_rule | pay_dev | pay_det
    cancel = cancel_dev | cancel_det
    payment = np.where(pay_dev, ask, rho_t)
    penalty_paid = np.where(detected, penalty, 0.0)

    out = _pack(mkt, paid, cancel, payment, penalty_paid, base_fare, accepted)
    return _summarise(out)


def solve_base_fare(mkt: Market, cfg: SimConfig, rho: float, penalty: float,
                    q: float, target_driver_surplus: float,
                    max_iter: int = 6, tol: float = 1e-5) -> Tuple[float, Dict[str, float]]:
    """Base fare b that holds expected driver surplus at the negotiation benchmark.

    The platform adjusts the base fare so that the toll rule changes the split,
    not the driver's participation margin (Section 5 of the paper). Solved by a
    damped fixed-point iteration; driver surplus is monotone in b.
    """
    b = 0.0
    res = simulate_precommitment(mkt, cfg, rho, penalty, q, base_fare=b)
    for _ in range(max_iter):
        gap = target_driver_surplus - res["driver_surplus"]
        if abs(gap) < tol:
            break
        b += gap                      # slope of driver surplus in b is ~ completion rate
        res = simulate_precommitment(mkt, cfg, rho, penalty, q, base_fare=b)
    return b, res


def run_replication(cfg: SimConfig, seed: int) -> Dict[str, Dict[str, float]]:
    """One replication: negotiation, two pre-commitment scenarios, sensitivity grid."""
    rng = np.random.default_rng(seed)
    mkt = build_market(cfg, rng)

    results: Dict[str, Dict[str, float]] = {}

    neg = simulate_negotiation(mkt, cfg)
    results["negotiation"] = neg

    # rho = 1, penalty = 1.5 F*
    pen_1 = cfg.penalty(cfg.toll)
    b1, c1 = solve_base_fare(mkt, cfg, 1.0, pen_1, cfg.q, neg["driver_surplus"])
    c1["base_fare"] = b1
    c1["penalty"] = pen_1
    results["precommit_rho1"] = c1

    # rho = 0.5, same absolute penalty (the platform publishes one penalty)
    pen_05 = pen_1
    b05, c05 = solve_base_fare(mkt, cfg, 0.5, pen_05, cfg.q, neg["driver_surplus"])
    c05["base_fare"] = b05
    c05["penalty"] = pen_05
    results["precommit_rho05"] = c05

    # Sensitivity grid: rho = 1, F in {0.5,1,1.5,3} F*, q in {0.6,0.8,1.0}
    grid: Dict[str, float] = {}
    for mult in (0.5, 1.0, 1.5, 3.0):
        for q in (0.6, 0.8, 1.0):
            pen = mult * cfg.reference_penalty(cfg.toll)
            res = simulate_precommitment(mkt, cfg, 1.0, pen, q, base_fare=b1)
            grid[f"{mult:.1f}|{q:.1f}"] = res["completed_rate"]
    results["sensitivity"] = grid

    return results


def table5_results(reps: Sequence[Dict[str, Dict[str, float]]],
                   cfg: SimConfig) -> List[Dict[str, object]]:
    """Means and standard deviations over replications for Table 5."""
    keys = ("negotiation", "precommit_rho1", "precommit_rho05")
    labels = {"negotiation": "Negotiation",
              "precommit_rho1": "Pre-commit, rho=1",
              "precommit_rho05": "Pre-commit, rho=0.5"}
    metrics = ("completed_rate", "cancellation_rate", "passenger_surplus", "driver_surplus")

    base_pax = float(np.mean([r["negotiation"]["passenger_surplus"] for r in reps]))
    base_drv = float(np.mean([r["negotiation"]["driver_surplus"] for r in reps]))

    rows = []
    for key in keys:
        row: Dict[str, object] = {"regime": labels[key]}
        for m in metrics:
            vals = np.array([r[key][m] for r in reps], dtype=float)
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            if m == "passenger_surplus":
                row["passenger_index"] = 100.0 * mean / base_pax
                row["passenger_index_sd"] = 100.0 * sd / base_pax
            elif m == "driver_surplus":
                row["driver_index"] = 100.0 * mean / base_drv
                row["driver_index_sd"] = 100.0 * sd / base_drv
            else:
                row[m] = mean
                row[f"{m}_sd"] = sd
        row["base_fare"] = float(np.mean([r.get(key, {}).get("base_fare", 0.0) for r in reps])) \
            if key != "negotiation" else 0.0
        row["penalty"] = float(np.mean([r[key].get("penalty", 0.0) for r in reps]))
        rows.append(row)
    return rows


def table6_sensitivity(reps: Sequence[Dict[str, Dict[str, float]]]) -> List[Dict[str, object]]:
    """Mean completed-trip rate over the penalty / detection grid (Table 6)."""
    rows = []
    for mult in (0.5, 1.0, 1.5, 3.0):
        row: Dict[str, object] = {"penalty_multiple": mult}
        for q in (0.6, 0.8, 1.0):
            vals = np.array([r["sensitivity"][f"{mult:.1f}|{q:.1f}"] for r in reps])
            row[f"q={q:.1f}"] = float(vals.mean())
            row[f"q={q:.1f}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def run_simulation(cfg: SimConfig, progress: bool = True) -> Tuple[List, List, List]:
    """Run all replications and return (table5 rows, table6 rows, per-rep dicts)."""
    reps = []
    t0 = time.time()
    for r in range(cfg.replications):
        reps.append(run_replication(cfg, cfg.seed + 1000 * r))
        if progress and (r + 1) % max(1, cfg.replications // 10) == 0:
            elapsed = time.time() - t0
            print(f"    replication {r + 1:>3d}/{cfg.replications}  "
                  f"({elapsed:5.1f}s, {elapsed / (r + 1):.2f}s per rep)")
    return table5_results(reps, cfg), table6_sensitivity(reps), reps


# =============================================================================
# SECTION 4 - OUTPUT: CSV, markdown report, figures
# =============================================================================


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None):
    if not rows:
        return
    names = list(fieldnames) if fieldnames else list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def console_table(headers: Sequence[str], rows: Sequence[Sequence[str]],
                  title: str = "") -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    out = []
    if title:
        out.append(title)
    out.append(line)
    out.append(sep)
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(out)


# --- figures ----------------------------------------------------------------


def make_figures(outdir: str, t5_rows: List[Dict[str, object]],
                 t6_rows: List[Dict[str, object]], cfg: SimConfig) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:      # matplotlib is optional
        print(f"  [figures skipped: {exc}]")
        return []

    produced = []
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 160})
    colours = {"blue": "#1f4e9c", "red": "#b3261e", "green": "#2e6b4f",
               "grey": "#6b6b6b", "orange": "#b5651d"}

    # Figure 3 of the paper: equilibrium payment vs. sunk cost
    k = np.linspace(0.0, 3.0, 200)
    rho_t = 1.2
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(k, k, color=colours["blue"], lw=2.2, label=r"Negotiation: payment $=k$")
    ax.plot(k, np.full_like(k, rho_t), color=colours["red"], lw=2.2, ls="--",
            label=r"Pre-commitment: payment $=\rho t$")
    ax.plot([rho_t], [rho_t], "o", color="black", ms=5)
    ax.annotate(r"$k=\rho t$", xy=(rho_t, rho_t), xytext=(rho_t + 0.30, rho_t + 0.55),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel(r"Passenger's sunk cost $k$")
    ax.set_ylabel("Equilibrium passenger payment")
    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_payment_vs_sunk_cost.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); produced.append(path)

    # Figure 4 of the paper: penalty threshold vs. sunk cost
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    k2 = np.linspace(1.0, 3.0, 300)
    for q, style, col in ((0.6, "-", colours["blue"]), (0.8, "--", colours["red"]),
                          (1.0, ":", "black")):
        ax.plot(k2, (1 - q) * (k2 - 1.0) / q, style, color=col, lw=2.0,
                label=rf"$q={q:.1f}$")
    ax.plot(k2, np.zeros_like(k2), color="grey", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"Passenger's sunk cost $k$")
    ax.set_ylabel(r"Penalty threshold $F^*$")
    ax.set_xlim(1.0, 3.0); ax.set_ylim(0.0, 2.2)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_penalty_threshold.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); produced.append(path)

    # Regime comparison: completed-trip and cancellation rates
    labels = [r["regime"].replace("Pre-commit, ", "Pre-commit\n") for r in t5_rows]
    comp = [float(r["completed_rate"]) for r in t5_rows]
    canc = [float(r["cancellation_rate"]) for r in t5_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].bar(x, comp, color=[colours["grey"], colours["blue"], colours["green"]], width=0.55)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["Negotiation", "Pre-commit\n$\\rho=1$",
                                                    "Pre-commit\n$\\rho=0.5$"])
    axes[0].set_ylabel("Completed-trip rate"); axes[0].set_ylim(0, 1)
    for xi, v in zip(x, comp):
        axes[0].text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    axes[1].bar(x, canc, color=[colours["grey"], colours["blue"], colours["green"]], width=0.55)
    axes[1].set_xticks(x); axes[1].set_xticklabels(["Negotiation", "Pre-commit\n$\\rho=1$",
                                                    "Pre-commit\n$\\rho=0.5$"])
    axes[1].set_ylabel("Cancellation rate (toll disputes)")
    for xi, v in zip(x, canc):
        axes[1].text(xi, v + 0.004, f"{v*100:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "fig3_regime_comparison.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); produced.append(path)

    # Penalty response curve (model output of the sensitivity grid)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    mult = [float(r["penalty_multiple"]) for r in t6_rows]
    for q, style, col in ((0.6, "-", colours["orange"]), (0.8, "--", colours["blue"]),
                          (1.0, ":", colours["green"])):
        ax.plot(mult, [float(r[f"q={q:.1f}"]) for r in t6_rows], style, marker="o",
                color=col, lw=1.9, label=rf"$q={q:.1f}$")
    ax.set_xlabel(r"Penalty $F$ (multiples of $F^*$)")
    ax.set_ylabel("Completed-trip rate")
    ax.set_xscale("log", base=2)
    ax.set_xticks(mult); ax.set_xticklabels([f"{m:g}" for m in mult])
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(outdir, "fig4_penalty_response.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig); produced.append(path)

    return produced


# --- report -----------------------------------------------------------------


def build_report(path: str, cfg: SimConfig, theory_rows: List[Dict[str, float]],
                 survey_rows: List[Dict[str, str]], t5: List[Dict[str, object]],
                 t6: List[Dict[str, object]], figures: List[str], runtime: float):
    targets = cfg.targets
    lines = [
        "# Replication report",
        "",
        "Code for *Who Pays the Toll? Fairness, Platform Pre-Commitment, and the",
        "Governance of Ex Post Opportunism in Long-Distance Ridesharing*.",
        "",
        f"- generated on {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- replications: {cfg.replications}, periods: {cfg.periods}, pairs per period: {cfg.pairs}",
        f"- baseline detection q = {cfg.q:.2f}, penalty = {cfg.penalty_multiple:.1f} F*, "
        f"F* at the {cfg.fstar_quantile:.0%} quantile of k",
        f"- wall-clock time: {runtime:.1f} s",
        "",
        "## A. Theory checks",
        "",
        "| check | numeric value | benchmark | pass |",
        "|---|---|---|---|",
    ]
    for row in theory_rows:
        lines.append(f"| {row['check']} | {row['value']:.4g} | {row['benchmark']:.4g} | "
                     f"{'yes' if row['pass'] else 'NO'} |")
    lines += ["", "## B. Survey audit", "",
              "| quantity | recomputed | reported | unit | consistent |",
              "|---|---|---|---|---|"]
    for row in survey_rows:
        lines.append(f"| {row['quantity']} | {row['recomputed']} | {row['reported']} | "
                     f"{row['unit']} | {row['consistent']} |")

    key_map = {"Negotiation": "N", "Pre-commit, rho=1": "C1", "Pre-commit, rho=0.5": "C05"}
    lines += ["", "## C. Simulation (Table 5)", "",
              "| regime | completed | (paper) | cancellation | (paper) | pax idx | (paper) |",
              "|---|---|---|---|---|---|---|"]
    for row in t5:
        kk = key_map.get(str(row["regime"]), "N")
        lines.append(
            f"| {row['regime']} | {row['completed_rate']:.3f} | "
            f"{targets[kk + '_completed']:.2f} | "
            f"{row['cancellation_rate']:.3f} | {targets[kk + '_cancel']:.3f} | "
            f"{row['passenger_index']:.1f} | {targets[kk + '_pax']:.1f} |")

    lines += ["", "## D. Simulation (Table 6, completed-trip rate)", "",
              "| F / F* | q=0.6 | q=0.8 | q=1.0 |", "|---|---|---|---|"]
    for row in t6:
        lines.append(f"| {row['penalty_multiple']:.1f} | {row['q=0.6']:.2f} | "
                     f"{row['q=0.8']:.2f} | {row['q=1.0']:.2f} |")
    lines += ["", "Manuscript, same ordering: 0.76/0.71/0.81, 0.85/0.82/0.89,",
              "0.89/0.86/0.90, 0.90/0.88/0.91.", ""]
    if figures:
        lines += ["## E. Figures", ""] + [f"- `{os.path.basename(f)}`" for f in figures] + [""]
    lines += ["## Notes on specification", "",
              "The paper reports simulation output but does not fully document the deviation",
              "decision, the information available to drivers, or the units of the penalty.",
              "This file therefore states its own specification explicitly:",
              "",
              "1. Drivers do not observe the passenger's individual sunk cost under the status",
              "   quo; they post a salient ask `demand_ratio * t`. Setting `--informed-demand`",
              "   switches to the frictionless Lemma-1 version (d = k), under which no",
              "   cancellation occurs, as the theory predicts.",
              "2. Under pre-commitment, deviation is a smoothed best response on the perceived",
              "   gain (1-q)(k_hat - rho t) - q F, with a noisy belief k_hat about k.",
              "   Deviation intensity and hence the cancellation rate are therefore sensitive",
              "   to q in a way that the paper's grid does not fully pin down: the reported grid",
              "   is not monotone in the implied deterrence ratio qF/(1-q), whereas the model",
              "   here necessarily is. The decision evaluates the incremental margin behind",
              "   Proposition 5; detection then voids the booking, which generates the",
              "   dispute-cancellation rate reported in Table 5.",
              "3. The penalty is set at the 95th percentile of the sunk-cost distribution",
              "   (a prudential choice, since F* is pointwise in k); the platform publishes one",
              "   absolute penalty, so the rho = 0.5 scenario inherits it.",
              "4. The base fare is solved so that mean driver surplus matches the negotiation",
              "   benchmark, i.e. the rule changes the split, not participation. Because the",
              "   neutralisation is near-complete, passenger surplus is almost identical at",
              "   rho = 1 and rho = 0.5; the manuscript reports a wider gap (122.6 vs 114.3),",
              "   which would require a partial base-fare adjustment.",
              ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# =============================================================================
# SECTION 5 - COMMAND LINE
# =============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replication code for the long-distance ridesharing toll paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", default="all",
                        choices=["all", "theory", "survey", "simulate", "figures", "check"],
                        help="which part of the paper to run")
    parser.add_argument("--outdir", default="outputs", help="output directory")
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--periods", type=int, default=200)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--q", type=float, default=0.80, help="detection probability")
    parser.add_argument("--penalty-multiple", type=float, default=1.5,
                        help="penalty F as a multiple of F*")
    parser.add_argument("--fstar-quantile", type=float, default=0.95)
    parser.add_argument("--availability", type=float, default=0.96)
    parser.add_argument("--demand-ratio", type=float, default=0.91)
    parser.add_argument("--informed-demand", action="store_true",
                        help="theory-consistent status quo: driver demands d = k")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke run (10 replications, 60 periods, 800 pairs)")
    return parser.parse_args(argv)


def cfg_from_args(args: argparse.Namespace) -> SimConfig:
    cfg = SimConfig(replications=args.replications, periods=args.periods,
                    pairs=args.pairs, seed=args.seed, q=args.q,
                    penalty_multiple=args.penalty_multiple,
                    fstar_quantile=args.fstar_quantile,
                    availability=args.availability,
                    demand_ratio=args.demand_ratio,
                    informed_demand=args.informed_demand)
    if args.quick:
        cfg.replications = min(cfg.replications, 10)
        cfg.periods = min(cfg.periods, 60)
        cfg.pairs = min(cfg.pairs, 800)
    return cfg


def banner(text: str):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = cfg_from_args(args)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    banner("Long-distance ridesharing toll governance - replication code")
    print(f"mode={args.mode}  replications={cfg.replications}  periods={cfg.periods}  "
          f"pairs={cfg.pairs}  seed={cfg.seed}")

    t_start = time.time()
    theory_rows = survey_rows = None
    t5 = t6 = None
    figures: List[str] = []

    # ---------------------------------------------------------------- theory
    if args.mode in ("all", "theory", "check"):
        banner("Part A - Theory: Lemma 1 and Propositions 1, 2, 4, 5")
        p = TheoryPrimitives()
        theory_rows = theory_analysis(p)
        rows_for_print = [[r["check"], f"{r['value']:.3g}", f"{r['benchmark']:.3g}",
                           "PASS" if r["pass"] else "FAIL"] for r in theory_rows]
        print(console_table(["check", "value", "benchmark", "status"], rows_for_print))
        write_csv(os.path.join(outdir, "tableA_theory_checks.csv"), theory_rows)

        w = prop4_welfare(p)
        print(f"\n  Illustrative primitives: v={p.v}, t={p.toll}, c_L={p.c_low}, "
              f"c_H={p.c_high}, k={p.sunk}, rho={p.rho}, q={p.q}")
        print(f"  Match rate: negotiation {w['match_rate_negotiation']:.2f} -> "
              f"pre-commitment {w['match_rate_precommitment']:.2f}")
        print(f"  Passenger utility: {w['passenger_utility_negotiation']:.2f} -> "
              f"{w['passenger_utility_precommitment']:.2f}  "
              f"(gain {w['utility_gain']:+.2f})")
        print(f"  F* at k={p.sunk}, rho*t={p.rho_t}, q={p.q}: "
              f"{penalty_threshold(p.sunk, p.rho_t, p.q):.3f}")

        if args.mode == "check":
            failures = [r for r in theory_rows if not r["pass"]]
            banner("Self-test result: " + ("FAIL" if failures else "all theory checks PASS"))
            if failures:
                for f in failures:
                    print("  FAILED:", f["check"])
                return 2

    # ---------------------------------------------------------------- survey
    if args.mode in ("all", "survey", "check"):
        banner("Part B - Stated-preference audit (mixed logit)")
        survey_rows = survey_audit()
        rows_for_print = [[r["quantity"][:58], r["recomputed"], r["reported"],
                           r["consistent"]] for r in survey_rows]
        print(console_table(["quantity", "recomputed", "reported", "consistent"],
                            rows_for_print))
        write_csv(os.path.join(outdir, "tableB_survey_audit.csv"), survey_rows)
        if args.mode == "check":
            bad = [r for r in survey_rows if r["consistent"] == "NO"]
            if bad:
                print("\nSurvey audit inconsistencies:", len(bad))
                return 2
            print("\nSurvey audit: all reported quantities reproduce.")

    # -------------------------------------------------------------- simulate
    if args.mode in ("all", "simulate", "figures"):
        if args.mode == "figures":
            cfg.replications = min(cfg.replications, 30)   # figures only need a stable mean
        banner("Part C - Agent-based simulation (Tables 5 and 6)")
        print("  calibration: v ~ LN(4.2,0.45); k ~ LN(median 1.5t, CV 0.6); "
              "c_L mean 0.35t (70%); c_H mean 0.65t (30%)")
        t5, t6, _ = run_simulation(cfg, progress=(args.mode != "figures"))

        write_csv(os.path.join(outdir, "tableC_simulation.csv"), t5)
        write_csv(os.path.join(outdir, "tableD_sensitivity.csv"), t6)

        banner("Table 5 - regime comparison (means over replications)")
        rows = [[r["regime"], f"{r['completed_rate']:.3f}",
                 f"{r['cancellation_rate']:.3f}", f"{r['passenger_index']:.1f}",
                 f"{r['driver_index']:.1f}", f"{r['base_fare']:+.3f}",
                 f"{r['penalty']:.3f}"] for r in t5]
        print(console_table(["regime", "completed", "cancel", "pax idx", "drv idx",
                             "base fare", "F"], rows))
        print("\n  Manuscript targets: 0.71/0.184/100.0/100.0 | 0.89/0.021/122.6/100.4 | "
              "0.86/0.034/114.3/99.1")

        banner("Table 6 - completed-trip rate vs. penalty and detection")
        rows = [[f"{r['penalty_multiple']:.1f}", f"{r['q=0.6']:.3f}",
                 f"{r['q=0.8']:.3f}", f"{r['q=1.0']:.3f}"] for r in t6]
        print(console_table(["F / F*", "q=0.6", "q=0.8", "q=1.0"], rows))
        print("\n  Manuscript: 0.76/0.71/0.81 | 0.85/0.82/0.89 | 0.89/0.86/0.90 | 0.90/0.88/0.91")
        print("  note: the paper's grid is not monotone in the implied deterrence ratio")
        print("        qF/(1-q); this file computes the grid from the model it documents.")

        gain = (t5[1]["completed_rate"] - t5[0]["completed_rate"]) * 100
        print(f"\n  Headline effect: pre-commitment (rho=1) raises the completed-trip "
              f"rate by {gain:.1f} percentage points.")

    # --------------------------------------------------------------- figures
    if args.mode in ("all", "figures") and t5 is not None and t6 is not None:
        banner("Part D - Figures")
        figures = make_figures(outdir, t5, t6, cfg)
        for f in figures:
            print("  wrote", f)

    # ---------------------------------------------------------------- report
    if args.mode == "all" and theory_rows and survey_rows and t5 and t6:
        runtime = time.time() - t_start
        build_report(os.path.join(outdir, "report.md"), cfg, theory_rows, survey_rows,
                     t5, t6, figures, runtime)
        with open(os.path.join(outdir, "run_config.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(cfg), fh, indent=2)
        print(f"\n  report: {os.path.join(outdir, 'report.md')}")
        print(f"  runtime: {runtime:.1f} s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
#（注：内容由AI生成）
