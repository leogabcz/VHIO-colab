"""Simulation validation helpers for Notebook_2_NB_fit.ipynb.

Run this after the notebook has defined:
    d_qc, DONORS, cal, nb, simulate, short

The helpers use the API visible in Notebook_2_NB_fit(1).ipynb.
They are deliberately separate from tcr_pipeline so that the notebook's existing
fit/test implementation remains the procedure under validation.

Stages:
  1. Repeated independent train/test neutral simulations using the notebook's
     two-step generator; refit phi(f) for every replicate; report FPR by misses.
  2. Sensitivity grid for the low-frequency power-law exponent (optionally N and
     phi_tech), scored against observed detection-pattern / count summaries.
  3. Model-aligned NB power simulations with known log-linear expansions and
     contractions, fit on an independent neutral training simulation.

Important: Stage 3 is a transparent NB2 power benchmark, not a power analysis under
exactly the same two-step data-generating mechanism as Stage 1. Use it as a first
power benchmark; a full two-step power analysis requires extending the project-local
simulate_two_step generator to accept time-varying latent frequencies.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence
import itertools
import math
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _seed(master_seed: int, *parts: int) -> int:
    """Deterministic independent uint32 seed for a particular task/replicate."""
    ss = np.random.SeedSequence([int(master_seed), *[int(x) for x in parts]])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _one_sample_per_day(g: pd.DataFrame) -> pd.DataFrame:
    required = {"patient", "day", "D_t", "lib", "c_t"}
    missing = required.difference(g.columns)
    if missing:
        raise KeyError(f"Sample design missing columns: {sorted(missing)}")
    return (g.sort_values("day")
             .groupby(["patient", "day"], as_index=False, sort=True)
             .first()
             .sort_values("day")
             .reset_index(drop=True))


def _simulate_two_step(
    person: str,
    design_df: pd.DataFrame,
    calibration: Mapping[str, Any],
    simulate_module: Any,
    *,
    seed: int,
    alpha_low: Optional[float] = None,
    N_scale: float = 1.0,
    phi_tech_scale: float = 1.0,
) -> pd.DataFrame:
    """Generate one two-step dataset with independent frequency/count RNG seeds."""
    cal = calibration[person]
    freq_seed = _seed(seed, 11)
    count_seed = _seed(seed, 29)
    exponent = float(cal["alpha_low"] if alpha_low is None else alpha_low)
    lam = simulate_module.draw_powerlaw(
        cal["pl"], f_lo=cal["f_lo"], alpha_low=exponent, seed=freq_seed
    )
    simulated, _meta = simulate_module.simulate_two_step(
        design_df,
        lam,
        float(cal["N"]) * float(N_scale),
        float(cal["phi_tech"]) * float(phi_tech_scale),
        seed=count_seed,
    )
    return simulated


def _n_detected_by_clone(df: pd.DataFrame) -> pd.Series:
    """Number of distinct observed timepoints per clonotype in a sparse table."""
    if df.empty:
        return pd.Series(dtype=int, name="n_detected")
    return df.groupby("clono")["day"].nunique().rename("n_detected")


def _fit_and_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    design_df: pd.DataFrame,
    *,
    nb_module: Any,
    min_freq: float = 1e-5,
    beta_bound: float = math.log(100.0),
) -> tuple[pd.DataFrame, Any]:
    """Refit phi(f) on train, then run the notebook's test on independent test data.

    As in the current notebook, dropout_N is estimated from the test data because
    that is how the proposed real-data pipeline currently operates. The trend itself
    is learned only from the independent training simulation.
    """
    fit = nb_module.fit_phi_trend(train)
    trend = fit["trend"]
    dropout_N = nb_module.estimate_dropout_N(test, phi=trend)
    result = nb_module.nb_trajectory_test(
        test,
        trend,
        likelihood="dropout",
        dropout_N=dropout_N,
        min_freq=min_freq,
        beta_bound=beta_bound,
    )
    result = nb_module.add_qvalues(result, "p_value", by="patient")

    if "clono" not in result.columns:
        raise KeyError("nb_trajectory_test output needs a 'clono' column to validate by missingness")
    n_tp = int(design_df["day"].nunique())
    n_det = _n_detected_by_clone(test)
    result["n_detected"] = result["clono"].map(n_det)
    result["n_miss"] = n_tp - result["n_detected"]
    result["n_miss"] = result["n_miss"].fillna(n_tp).astype(int)
    return result, fit


# ---------------------------------------------------------------------------
# STEP 1 — repeated independent train/test null validation
# ---------------------------------------------------------------------------

def run_repeated_null_validation(
    *,
    d_qc: pd.DataFrame,
    donors: Sequence[str],
    calibration: Mapping[str, Any],
    nb_module: Any,
    simulate_module: Any,
    n_reps: int = 20,
    master_seed: int = 20261015,
    min_freq: float = 1e-5,
    beta_bound: float = math.log(100.0),
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated all-null test using independent two-step train/test simulations.

    Returns
    -------
    run_metrics : one row per donor x replicate.
    missingness_metrics : one row per donor x replicate x number of misses.

    Start with n_reps=5-10 to benchmark runtime, then use >=100 for final summaries.
    """
    run_rows: list[dict[str, Any]] = []
    miss_rows: list[dict[str, Any]] = []

    for donor_i, person in enumerate(donors):
        g = d_qc[d_qc["patient"] == person].copy()
        design = _one_sample_per_day(g)
        for rep in range(n_reps):
            try:
                # Independent underlying frequency draws AND independent count RNGs.
                train = _simulate_two_step(
                    person, g, calibration, simulate_module,
                    seed=_seed(master_seed, donor_i, rep, 1001),
                )
                test = _simulate_two_step(
                    person, g, calibration, simulate_module,
                    seed=_seed(master_seed, donor_i, rep, 2003),
                )
                result, fit = _fit_and_test(
                    train, test, design, nb_module=nb_module,
                    min_freq=min_freq, beta_bound=beta_bound,
                )
                pvals = pd.to_numeric(result["p_value"], errors="coerce")
                qvals = pd.to_numeric(result["q_value"], errors="coerce")
                valid = pvals.notna()
                run_rows.append({
                    "donor": person,
                    "replicate": rep,
                    "status": "ok",
                    "n_tested": int(valid.sum()),
                    "fpr_p05": float((pvals[valid] < 0.05).mean()) if valid.any() else np.nan,
                    "fpr_p01": float((pvals[valid] < 0.01).mean()) if valid.any() else np.nan,
                    "bh_calls_q05": int((qvals < 0.05).sum()),
                    "any_bh_call_q05": bool((qvals < 0.05).any()),
                    "fit_phi_bins": len(fit.get("bins", [])) if isinstance(fit, dict) else np.nan,
                })

                for n_miss, sub in result.loc[valid].groupby("n_miss", dropna=False):
                    pv = pd.to_numeric(sub["p_value"], errors="coerce").dropna()
                    miss_rows.append({
                        "donor": person,
                        "replicate": rep,
                        "n_miss": int(n_miss),
                        "n_tested": int(len(pv)),
                        "fpr_p05": float((pv < 0.05).mean()) if len(pv) else np.nan,
                        "fpr_p01": float((pv < 0.01).mean()) if len(pv) else np.nan,
                    })
            except Exception as exc:
                run_rows.append({
                    "donor": person, "replicate": rep, "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_tested": 0, "fpr_p05": np.nan, "fpr_p01": np.nan,
                    "bh_calls_q05": np.nan, "any_bh_call_q05": np.nan,
                })
            if progress and (rep == 0 or (rep + 1) % 10 == 0 or rep + 1 == n_reps):
                print(f"[{person}] replicate {rep + 1}/{n_reps}")

    return pd.DataFrame(run_rows), pd.DataFrame(miss_rows)


def summarise_null_validation(
    run_metrics: pd.DataFrame,
    missingness_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise replicate-to-replicate uncertainty, rather than pooling all p-values."""
    ok = run_metrics[run_metrics["status"] == "ok"].copy()
    if ok.empty:
        overall = pd.DataFrame()
    else:
        overall = (ok.groupby("donor")
                   .agg(
                       n_reps=("replicate", "nunique"),
                       mean_fpr_p05=("fpr_p05", "mean"),
                       median_fpr_p05=("fpr_p05", "median"),
                       fpr_p05_q025=("fpr_p05", lambda x: x.quantile(0.025)),
                       fpr_p05_q975=("fpr_p05", lambda x: x.quantile(0.975)),
                       mean_fpr_p01=("fpr_p01", "mean"),
                       mean_bh_calls=("bh_calls_q05", "mean"),
                       fraction_reps_with_bh_call=("any_bh_call_q05", "mean"),
                   ).reset_index())
    if missingness_metrics.empty:
        by_miss = pd.DataFrame()
    else:
        by_miss = (missingness_metrics.groupby(["donor", "n_miss"])
                   .agg(
                       reps_with_bin=("replicate", "nunique"),
                       mean_n_tested=("n_tested", "mean"),
                       mean_fpr_p05=("fpr_p05", "mean"),
                       fpr_p05_q025=("fpr_p05", lambda x: x.quantile(0.025)),
                       fpr_p05_q975=("fpr_p05", lambda x: x.quantile(0.975)),
                       mean_fpr_p01=("fpr_p01", "mean"),
                   ).reset_index())
    return overall, by_miss


# ---------------------------------------------------------------------------
# STEP 2 — fit/check detection-pattern matching for the two-step generator
# ---------------------------------------------------------------------------

def _detection_pattern_summary(df: pd.DataFrame, simulate_module: Any) -> dict[str, float]:
    """Summaries of detection sparsity and count scale for observed/simulated tables."""
    base = simulate_module.repertoire_summary(df)
    base = base.to_dict() if hasattr(base, "to_dict") else dict(base)
    if df.empty:
        n1 = n2 = n3 = np.nan
        n_unique = 0
    else:
        detections = df.groupby("clono")["day"].nunique()
        denom = max(len(detections), 1)
        n1 = float((detections == 1).sum() / denom)
        n2 = float((detections == 2).sum() / denom)
        n3 = float((detections >= 3).sum() / denom)
        n_unique = int(detections.size)
    return {
        "fraction_seen_once": n1,
        "fraction_seen_twice": n2,
        "fraction_seen_3plus": n3,
        "distinct_clones": float(base.get("distinct clones", n_unique)),
        "clones_used_for_phi": float(base.get("clones used for phi", np.nan)),
        "median_count": float(base.get("median count", pd.to_numeric(df.get("count", pd.Series(dtype=float)), errors="coerce").median())),
        "clones_per_sample_median": float(base.get("clones per sample (median)", np.nan)),
    }


def _pattern_loss(target: Mapping[str, float], candidate: Mapping[str, float]) -> float:
    """Transparent, tunable discrepancy score. Lower is a better descriptive match.

    Fraction tolerances of 0.03 and log-count tolerances of log(1.25) are practical
    defaults, not formal statistical confidence limits. Inspect component losses too.
    """
    terms = []
    for key in ("fraction_seen_once", "fraction_seen_twice"):
        a, b = float(target.get(key, np.nan)), float(candidate.get(key, np.nan))
        if np.isfinite(a) and np.isfinite(b):
            terms.append(((b - a) / 0.03) ** 2)
    for key in ("distinct_clones", "clones_used_for_phi", "median_count"):
        a, b = float(target.get(key, np.nan)), float(candidate.get(key, np.nan))
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            terms.append((np.log(b / a) / np.log(1.25)) ** 2)
    return float(np.mean(terms)) if terms else np.nan


def tune_two_step_detection_pattern(
    person: str,
    *,
    d_qc: pd.DataFrame,
    calibration: Mapping[str, Any],
    simulate_module: Any,
    alpha_offsets: Sequence[float] = (-0.4, -0.2, 0.0, 0.2, 0.4),
    N_scales: Sequence[float] = (1.0,),
    phi_tech_scales: Sequence[float] = (1.0,),
    n_reps: int = 3,
    master_seed: int = 77031,
    progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Explore whether the low-frequency exponent can improve detection-pattern fit.

    Keep N/phi_tech fixed initially. Expand N_scales / phi_tech_scales only if a
    low-tail-exponent sweep cannot bring the detection pattern close to the real data.
    This is simulator calibration, so use an independent validation set/donor for
    final claims; do not present the best-in-sample grid cell as independent evidence.
    """
    g = d_qc[d_qc["patient"] == person].copy()
    c = calibration[person]
    target = _detection_pattern_summary(g, simulate_module)
    baseline_alpha = float(c["alpha_low"])
    candidate_alphas = sorted({max(1.05, baseline_alpha + float(v)) for v in alpha_offsets})
    rows: list[dict[str, Any]] = []
    ncomb = len(candidate_alphas) * len(N_scales) * len(phi_tech_scales)
    comb_i = 0
    for alpha_low, n_scale, phi_scale in itertools.product(candidate_alphas, N_scales, phi_tech_scales):
        summaries = []
        for rep in range(n_reps):
            sim_df = _simulate_two_step(
                person, g, calibration, simulate_module,
                seed=_seed(master_seed, 301, comb_i, rep),
                alpha_low=alpha_low,
                N_scale=float(n_scale),
                phi_tech_scale=float(phi_scale),
            )
            summaries.append(_detection_pattern_summary(sim_df, simulate_module))
        m = pd.DataFrame(summaries).mean(numeric_only=True).to_dict()
        rows.append({
            "person": person,
            "alpha_low": alpha_low,
            "N_scale": float(n_scale),
            "phi_tech_scale": float(phi_scale),
            "n_reps": n_reps,
            "loss": _pattern_loss(target, m),
            **{f"target_{k}": v for k, v in target.items()},
            **{f"sim_{k}": v for k, v in m.items()},
        })
        comb_i += 1
        if progress:
            print(f"[{person}] pattern candidate {comb_i}/{ncomb}")
    result = pd.DataFrame(rows).sort_values("loss").reset_index(drop=True)
    return result, target


# ---------------------------------------------------------------------------
# STEP 3 — power/FDR under a known NB2 model with planted log-linear effects
# ---------------------------------------------------------------------------

def _empirical_frequency_pool(g: pd.DataFrame, simulate_module: Any) -> np.ndarray:
    f = np.asarray(simulate_module.pooled_frequency(g), dtype=float)
    f = f[np.isfinite(f) & (f > 0)]
    if not len(f):
        raise ValueError("No positive finite frequencies returned by simulate.pooled_frequency")
    return f


def _draw_freqs(pool: np.ndarray, n_clones: int, rng: np.random.Generator) -> np.ndarray:
    # Resample with replacement to permit n_clones larger than the observed pool.
    return rng.choice(pool, size=int(n_clones), replace=True).astype(float)


def _simulate_nb_sparse(
    design: pd.DataFrame,
    freqs: np.ndarray,
    *,
    phi: float,
    signal_indices: np.ndarray,
    fold_change: float,
    direction: str,
    seed: int,
) -> tuple[pd.DataFrame, set[str]]:
    """NB2 trajectory generator with exact known signals and real sample thresholds.

    Frequencies are treated as focal clonotype frequencies against an unmodelled
    background. They are not renormalized across the analysed subset; this avoids
    forcing non-signal clonotypes to change merely because only a subset of the
    complete repertoire is represented. NB sampling uses the notebook's lib offset.
    """
    if phi <= 0 or fold_change < 1:
        raise ValueError("phi must be positive and fold_change must be >= 1")
    if direction not in {"null", "expansion", "contraction"}:
        raise ValueError("direction must be null/expansion/contraction")
    rng = np.random.default_rng(seed)
    smeta = _one_sample_per_day(design)
    days = smeta["day"].to_numpy(float)
    if len(days) > 1 and days.max() > days.min():
        s = (days - days.min()) / (days.max() - days.min())
    else:
        s = np.zeros_like(days)
    n_clones = len(freqs)
    beta = np.zeros(n_clones, dtype=float)
    if direction != "null" and len(signal_indices):
        sign = 1.0 if direction == "expansion" else -1.0
        beta[np.asarray(signal_indices, dtype=int)] = sign * np.log(fold_change)
    f_ct = np.asarray(freqs, dtype=float)[:, None] * np.exp(beta[:, None] * s[None, :])

    offsets = smeta["lib"].to_numpy(float)
    cutoffs = smeta["c_t"].to_numpy(int)
    mu = f_ct * offsets[None, :]
    p = phi / (phi + mu)
    counts = rng.negative_binomial(phi, p)

    rows: list[dict[str, Any]] = []
    clone_ids = np.array([f"SIM_{i:08d}" for i in range(n_clones)], dtype=object)
    for j, sample in smeta.iterrows():
        ix = np.flatnonzero(counts[:, j] >= int(cutoffs[j]))
        if not len(ix):
            continue
        base = sample.to_dict()
        for i in ix:
            row = dict(base)
            row.update({"clono": clone_ids[i], "count": int(counts[i, j])})
            rows.append(row)
    sim_df = pd.DataFrame(rows)
    signal_ids = set(clone_ids[np.asarray(signal_indices, dtype=int)].tolist())
    return sim_df, signal_ids


def run_power_experiment_nb(
    person: str,
    *,
    d_qc: pd.DataFrame,
    nb_module: Any,
    simulate_module: Any,
    n_reps: int = 50,
    n_clones: int = 30000,
    phi_true: float = 2.1,
    signal_fraction: float = 0.05,
    signal_freq_min: float = 1e-5,
    signal_freq_max: float = np.inf,
    folds: Sequence[float] = (1.0, 2.0, 4.0, 10.0),
    directions: Sequence[str] = ("expansion", "contraction"),
    master_seed: int = 119003,
    min_freq: float = 1e-5,
    beta_bound: float = math.log(100.0),
    likelihood: str = "censored",
    progress: bool = True,
) -> pd.DataFrame:
    """Estimate FPR/power for known alternatives under a simple NB2 generator.

    An independent neutral NB2 simulation trains phi(f) in each replicate. For each
    test scenario, the true signal labels are known exactly. Power is reported both
    unconditionally (all planted clones) and conditional on a planted clone entering
    the test results. By default likelihood='censored', which matches the NB2 count
    generator plus the sample-specific threshold. Choosing likelihood='dropout' is a
    deliberate model-mismatch sensitivity analysis: this generator does not include
    the extra cell-sampling dropout layer used by that likelihood.
    """
    if likelihood not in {"censored", "truncated", "dropout"}:
        raise ValueError("likelihood must be 'censored', 'truncated', or 'dropout'")
    g = d_qc[d_qc["patient"] == person].copy()
    design = _one_sample_per_day(g)
    pool = _empirical_frequency_pool(g, simulate_module)
    rows: list[dict[str, Any]] = []
    scenarios = [("null", 1.0, 0.0)]
    scenarios += [(direction, float(fold), float(signal_fraction))
                  for direction in directions for fold in folds if float(fold) > 1]

    for rep in range(n_reps):
        # Independent neutral training dataset; fitted trend re-estimated each replicate.
        rng_train = np.random.default_rng(_seed(master_seed, rep, 71))
        freqs_train = _draw_freqs(pool, n_clones, rng_train)
        train_df, _ = _simulate_nb_sparse(
            design, freqs_train, phi=phi_true,
            signal_indices=np.array([], dtype=int), fold_change=1.0,
            direction="null", seed=_seed(master_seed, rep, 73),
        )
        fit = nb_module.fit_phi_trend(train_df)
        trend = fit["trend"]

        # Reuse one independent test baseline per replicate to make effect-size
        # comparisons paired. Counts are re-drawn independently for every scenario.
        rng_test = np.random.default_rng(_seed(master_seed, rep, 89))
        freqs_test = _draw_freqs(pool, n_clones, rng_test)
        eligible = np.flatnonzero(
            (freqs_test >= signal_freq_min) & (freqs_test < signal_freq_max)
        )
        if len(eligible) == 0 and any(s[0] != "null" for s in scenarios):
            raise ValueError(
                f"No simulated clones in signal frequency band [{signal_freq_min}, {signal_freq_max}). "
                "Increase n_clones or widen the band."
            )
        n_signal = max(1, int(round(signal_fraction * len(eligible)))) if len(eligible) else 0
        rng_sig = np.random.default_rng(_seed(master_seed, rep, 97))
        signal_indices = (rng_sig.choice(eligible, size=min(n_signal, len(eligible)), replace=False)
                          if n_signal else np.array([], dtype=int))

        for scen_i, (direction, fold, frac) in enumerate(scenarios):
            idx = signal_indices if direction != "null" else np.array([], dtype=int)
            test_df, signal_ids = _simulate_nb_sparse(
                design, freqs_test, phi=phi_true,
                signal_indices=idx,
                fold_change=fold,
                direction=direction,
                seed=_seed(master_seed, rep, 101 + scen_i),
            )
            test_kwargs = dict(
                likelihood=likelihood,
                min_freq=min_freq,
                beta_bound=beta_bound,
            )
            if likelihood == "dropout":
                # The data generator is pure NB2; this option intentionally tests
                # the dropout likelihood under model mismatch, not exact calibration.
                test_kwargs["dropout_N"] = nb_module.estimate_dropout_N(test_df, phi=trend)
            result = nb_module.nb_trajectory_test(test_df, trend, **test_kwargs)
            result = nb_module.add_qvalues(result, "p_value", by="patient")
            valid = result["p_value"].notna()
            tested_ids = set(result.loc[valid, "clono"].astype(str)) if "clono" in result.columns else set()
            signal_ids_str = {str(x) for x in signal_ids}
            tested_signal = signal_ids_str.intersection(tested_ids)
            discoveries = result.loc[result["q_value"] < 0.05].copy()
            discovered_ids = set(discoveries["clono"].astype(str)) if "clono" in discoveries.columns else set()
            true_positives = len(discovered_ids.intersection(signal_ids_str))
            false_positives = len(discovered_ids.difference(signal_ids_str))
            n_tested_signal = len(tested_signal)
            n_planted = len(signal_ids_str)
            n_disc = len(discovered_ids)
            if "clono" in result.columns:
                tested_is_signal = result["clono"].astype(str).isin(signal_ids_str)
                null_valid = valid & (~tested_is_signal)
            else:
                null_valid = valid
            beta_col = next((col for col in (
                "beta_hat", "beta", "beta_mle", "beta_est", "beta_estimate",
                "slope", "logFC", "log_fc"
            ) if col in result.columns), None)
            directional_tp = np.nan
            if beta_col is not None and direction in {"expansion", "contraction"} and not discoveries.empty:
                sig_beta = pd.to_numeric(discoveries[beta_col], errors="coerce")
                sig_ids = discoveries["clono"].astype(str)
                correct_sign = (sig_beta > 0) if direction == "expansion" else (sig_beta < 0)
                directional_tp = int((sig_ids.isin(signal_ids_str) & correct_sign).sum())
            row = {
                "person": person,
                "replicate": rep,
                "likelihood": likelihood,
                "direction": direction,
                "fold_change": fold,
                "signal_fraction_of_eligible": frac,
                "n_clones_generated": int(n_clones),
                "n_tested": int(valid.sum()),
                "n_planted": n_planted,
                "n_planted_tested": n_tested_signal,
                "signal_detection_rate": n_tested_signal / n_planted if n_planted else np.nan,
                "power_unconditional_q05": true_positives / n_planted if n_planted else np.nan,
                "power_conditional_q05": true_positives / n_tested_signal if n_tested_signal else np.nan,
                "direction_correct_power_unconditional_q05": directional_tp / n_planted if n_planted and np.isfinite(directional_tp) else np.nan,
                "discoveries_q05": n_disc,
                "false_discoveries_q05": false_positives,
                "fdp_q05": false_positives / n_disc if n_disc else 0.0,
                "fpr_p05": float((result.loc[null_valid, "p_value"] < 0.05).mean()) if null_valid.any() else np.nan,
                "fpr_p01": float((result.loc[null_valid, "p_value"] < 0.01).mean()) if null_valid.any() else np.nan,
            }
            rows.append(row)
        if progress and (rep == 0 or (rep + 1) % 10 == 0 or rep + 1 == n_reps):
            print(f"[{person}] power replicate {rep + 1}/{n_reps}")
    return pd.DataFrame(rows)


def summarise_power_results(power_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate power/FDP over replicates with replicate-level uncertainty."""
    keys = ["person", "direction", "fold_change"]
    return (power_runs.groupby(keys, dropna=False)
            .agg(
                n_reps=("replicate", "nunique"),
                mean_power_unconditional=("power_unconditional_q05", "mean"),
                power_unconditional_q025=("power_unconditional_q05", lambda x: x.quantile(0.025)),
                power_unconditional_q975=("power_unconditional_q05", lambda x: x.quantile(0.975)),
                mean_power_conditional=("power_conditional_q05", "mean"),
                mean_direction_correct_power=("direction_correct_power_unconditional_q05", "mean"),
                mean_signal_detection_rate=("signal_detection_rate", "mean"),
                mean_fdp=("fdp_q05", "mean"),
                fraction_runs_with_discovery=("discoveries_q05", lambda x: (x > 0).mean()),
                mean_fpr_p05=("fpr_p05", "mean"),
                mean_fpr_p01=("fpr_p01", "mean"),
                mean_discoveries=("discoveries_q05", "mean"),
            ).reset_index())
