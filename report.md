# Replication report

Code for *Who Pays the Toll? Fairness, Platform Pre-Commitment, and the
Governance of Ex Post Opportunism in Long-Distance Ridesharing*.

- generated on 2026-11-17 11:16:30
- replications: 100, periods: 200, pairs per period: 2000
- baseline detection q = 0.80, penalty = 1.5 F*, F* at the 95% quantile of k
- wall-clock time: 119.7 s

## A. Theory checks

| check | numeric value | benchmark | pass |
|---|---|---|---|
| Lemma 1: demand d=k is optimal | 0 | 0 | yes |
| Prop 1: payment equals k for both types | 1.6 | 1.6 | yes |
| Prop 2: deviation gain below compliance | -0.08 | -0.08 | yes |
| Prop 5: F* sets deviation gain to zero | 2.22e-16 | 0 | yes |
| Prop 5: F* rises in k, falls in q | 1 | 1 | yes |
| Prop 4(i): match rate weakly higher under C | 1 | 1 | yes |
| Prop 4(ii): passenger utility higher under C | 1 | 1 | yes |
| Prop 4(iii): zero driver toll rent at rho=1 | 0 | 0 | yes |

## B. Survey audit

| quantity | recomputed | reported | unit | consistent |
|---|---|---|---|---|
| WTP for rule certainty (share of reference fare) | 0.1430 | 0.1430 | share | yes |
| WTP for rule certainty (CNY) | 14.6532 | n/a | CNY | yes |
| WTP for rule certainty (% of the CNY 60 toll) | 24.4219 | n/a | % | yes |
| Likelihood-ratio statistic (MXL vs MNL) | 266.4000 | 266.4000 | chi2 | yes |
| LR p-value (df = 3) | <1e-4 | n/a | p | yes |
| Observations per respondent | 6.0000 | 6.0000 | tasks | yes |
| Random-parameter observations | 5400.0000 | 5400.0000 | obs | yes |
| H1: mean certainty coefficient positive | +0.203 | n/a | logit coefficient | yes |
| H2: certainty x re-planning cost positive | +0.061 | n/a | logit coefficient | yes |
| H3: certainty x toll share negative | -0.047 | n/a | logit coefficient | yes |
| H4: certainty x fairness concern positive | +0.038 | n/a | logit coefficient | yes |
| Implied standard error of the certainty mean | 0.0179 | n/a | logit coefficient | yes |

## C. Simulation (Table 5)

| regime | completed | (paper) | cancellation | (paper) | pax idx | (paper) |
|---|---|---|---|---|---|---|
| Negotiation | 0.698 | 0.71 | 0.184 | 0.184 | 100.0 | 100.0 |
| Pre-commit, rho=1 | 0.873 | 0.89 | 0.020 | 0.021 | 125.3 | 122.6 |
| Pre-commit, rho=0.5 | 0.867 | 0.86 | 0.031 | 0.034 | 124.4 | 114.3 |

## D. Simulation (Table 6, completed-trip rate)

| F / F* | q=0.6 | q=0.8 | q=1.0 |
|---|---|---|---|
| 0.5 | 0.51 | 0.68 | 0.89 |
| 1.0 | 0.65 | 0.83 | 0.89 |
| 1.5 | 0.74 | 0.87 | 0.89 |
| 3.0 | 0.85 | 0.89 | 0.89 |

Manuscript, same ordering: 0.76/0.71/0.81, 0.85/0.82/0.89,
0.89/0.86/0.90, 0.90/0.88/0.91.

## E. Figures

- `fig1_payment_vs_sunk_cost.png`
- `fig2_penalty_threshold.png`
- `fig3_regime_comparison.png`
- `fig4_penalty_response.png`

## Notes on specification

The paper reports simulation output but does not fully document the deviation
decision, the information available to drivers, or the units of the penalty.
This file therefore states its own specification explicitly:

1. Drivers do not observe the passenger's individual sunk cost under the status
   quo; they post a salient ask `demand_ratio * t`. Setting `--informed-demand`
   switches to the frictionless Lemma-1 version (d = k), under which no
   cancellation occurs, as the theory predicts.
2. Under pre-commitment, deviation is a smoothed best response on the perceived
   gain (1-q)(k_hat - rho t) - q F, with a noisy belief k_hat about k.
   Deviation intensity and hence the cancellation rate are therefore sensitive
   to q in a way that the paper's grid does not fully pin down: the reported grid
   is not monotone in the implied deterrence ratio qF/(1-q), whereas the model
   here necessarily is. The decision evaluates the incremental margin behind
   Proposition 5; detection then voids the booking, which generates the
   dispute-cancellation rate reported in Table 5.
3. The penalty is set at the 95th percentile of the sunk-cost distribution
   (a prudential choice, since F* is pointwise in k); the platform publishes one
   absolute penalty, so the rho = 0.5 scenario inherits it.
4. The base fare is solved so that mean driver surplus matches the negotiation
   benchmark, i.e. the rule changes the split, not participation. Because the
   neutralisation is near-complete, passenger surplus is almost identical at
   rho = 1 and rho = 0.5; the manuscript reports a wider gap (122.6 vs 114.3),
   which would require a partial base-fare adjustment.

> （注：部分内容由豆包 AI 生成）