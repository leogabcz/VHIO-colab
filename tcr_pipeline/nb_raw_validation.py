"""Validation helpers for the raw-count NB + Cox--Reid TCR pipeline.

Requires a project-local module exposing the API in tcr_pipeline.nb_glm, plus a
long-format detected-only dataframe with columns patient, day, clono, count, D_t,
lib. The detection threshold is supplied as a scalar ``c_t`` argument (normally 1);
legacy NB module versions that expect a cutoff column are supported through an internal
temporary column.

The procedures are deliberately separated:
  1) estimator recovery under a known NB2 + independent cell-dropout generator;
  2) empirical leave-one-healthy-donor-out calibration on real data;
  3) parametric null calibration conditional on fitted donor trends;
  4) power under planted slope alternatives using the same NB2 + dropout generator.

Simulation results are conditional on the chosen empirical frequency pool and plug-in
estimates. They are validation diagnostics, not proof of model adequacy.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence
import warnings
import inspect

import numpy as np
import pandas as pd


def _validate_cutoff(c_t: int) -> int:
    """Validate the scalar count threshold used by the observation model."""
    if isinstance(c_t, bool) or not isinstance(c_t, (int, np.integer)) or int(c_t) < 1:
        raise ValueError("c_t must be a positive integer (normally 1 for raw counts)")
    return int(c_t)


def _sample_design(g: pd.DataFrame, c_t: int = 1) -> pd.DataFrame:
    """One row per sample; derive the detection threshold from scalar ``c_t``.

    A c_t column is not required. If an old input happens to contain one, it must be
    constant within each sample and agree with the explicitly supplied scalar.
    """
    c_t = _validate_cutoff(c_t)
    required = {"patient", "day", "D_t", "lib"}
    missing = required - set(g.columns)
    if missing:
        raise KeyError(f"Data are missing sample-design columns: {sorted(missing)}")
    cols = ["patient", "day", "D_t", "lib"]
    if "tmm_factor" in g.columns:
        cols.append("tmm_factor")
    if "c_t" in g.columns:
        cols.append("c_t")
    d = g[cols].groupby(["patient", "day"], as_index=False, sort=True).first()
    for col in ["D_t", "lib"] + (["tmm_factor"] if "tmm_factor" in d else []):
        nuniq = g.groupby(["patient", "day"])[col].nunique(dropna=False)
        if (nuniq > 1).any():
            raise ValueError(f"{col} is not constant within patient/day")
    if "c_t" in d.columns:
        nuniq = g.groupby(["patient", "day"])["c_t"].nunique(dropna=False)
        if (nuniq > 1).any():
            raise ValueError("c_t column is not constant within patient/day")
        existing = pd.to_numeric(d["c_t"], errors="raise").astype(int)
        if not (existing == c_t).all():
            raise ValueError(
                f"Existing c_t column disagrees with scalar c_t={c_t}; "
                "remove the old column or pass the intended scalar cutoff."
            )
    d["c_t"] = c_t
    d = d.sort_values(["patient", "day"]).reset_index(drop=True)
    return d


def _call_with_cutoff(func: Callable, df: pd.DataFrame, c_t: int, kwargs: Optional[dict] = None):
    """Call NB API whether it takes scalar ``c_t`` or a legacy cutoff-column name.

    Supports current signatures with c_t=<int> and older signatures with c_col or
    cutoff_col plus a dataframe column. The caller's raw dataframe is not mutated.
    """
    c_t = _validate_cutoff(c_t)
    kw = dict(kwargs or {})
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        params = {}

    # Prefer an explicit scalar cutoff parameter when the module offers one.
    scalar_name = next((name for name in ("c_val", "c_t", "cutoff_count", "count_cutoff", "cutoff")
                        if name in params), None)
    if scalar_name is not None:
        for legacy in ("c_col", "cutoff_col"):
            kw.pop(legacy, None)
        kw[scalar_name] = c_t
        call_df = df
    elif "c_col" in params or "cutoff_col" in params:
        # Compatibility path for the uploaded legacy nb_glm.py API.
        call_df = df.copy()
        temp_col = "__validation_c_t"
        call_df[temp_col] = c_t
        if "c_col" in params:
            kw.pop("cutoff_col", None)
            kw["c_col"] = temp_col
        else:
            kw.pop("c_col", None)
            kw["cutoff_col"] = temp_col
    else:
        # If the function has no threshold argument, its implementation must be
        # threshold-independent; don't send unsupported legacy column-name kwargs.
        call_df = df
        kw.pop("c_col", None)
        kw.pop("cutoff_col", None)
        for name in ("c_t", "cutoff_count", "count_cutoff", "cutoff"):
            kw.pop(name, None)

    # Guard against optional kwargs differing between module versions.
    accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if params and not accepts_varkw:
        kw = {k: v for k, v in kw.items() if k in params}
    return func(call_df, **kw)


def check_raw_count_data(df: pd.DataFrame, *, c_t: int = 1,
                         require_c1: Optional[bool] = None) -> pd.DataFrame:
    """Check raw detected-only rows and return a sample table with derived ``c_t``.

    ``c_t`` is a scalar threshold, not a required column. ``require_c1`` is accepted
    for backward compatibility; when true it requires the scalar cutoff to be 1.
    """
    c_t = _validate_cutoff(c_t)
    required = {"patient", "day", "clono", "count", "D_t", "lib"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Raw input is missing columns: {sorted(missing)}")
    counts = pd.to_numeric(df["count"], errors="coerce")
    if counts.isna().any() or (counts < c_t).any():
        raise ValueError(f"Detected-only input contains counts below c_t={c_t}")
    if require_c1 and c_t != 1:
        raise ValueError("require_c1=True requires scalar c_t=1")
    d = _sample_design(df, c_t=c_t)
    minima = (df.groupby(["patient", "day"])["count"].min()
                .rename("min_observed_count").reset_index())
    return d.merge(minima, on=["patient", "day"], how="left")


def _frequency_pool(g: pd.DataFrame, c_t: int = 1) -> np.ndarray:
    """Per-clone observed mean frequency, with all sample depths in denominator.

    Absent clone-sample rows contribute zero to the numerator and still contribute
    their sample depth to the denominator. This is a plug-in pool conditional on clones
    observed at least once; unseen clonotypes cannot be recovered from sparse data.
    """
    design = _sample_design(g, c_t=c_t)
    total_depth = float(design["D_t"].sum())
    if total_depth <= 0:
        raise ValueError("Total depth must be positive")
    freq = g.groupby("clono")["count"].sum().astype(float) / total_depth
    values = freq.to_numpy(float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        raise ValueError("No positive frequencies available")
    return values


def _resolve_N(N: Any, days: np.ndarray) -> np.ndarray:
    if isinstance(N, pd.DataFrame):
        if not {"day", "N_t"}.issubset(N.columns):
            raise KeyError("N dataframe must contain 'day' and 'N_t'")
        vals = N.set_index("day")["N_t"].reindex(days).to_numpy(float)
    elif isinstance(N, Mapping):
        vals = np.asarray([N[d] for d in days], dtype=float)
    else:
        vals = np.asarray(N, dtype=float)
        if vals.ndim == 0:
            vals = np.full(len(days), float(vals))
    if len(vals) != len(days) or np.any(~np.isfinite(vals)) or np.any(vals <= 0):
        raise ValueError("N must give one finite positive value per sample/day")
    return vals


def simulate_nb_dropout(
    g_design_source: pd.DataFrame,
    frequencies: Sequence[float],
    *,
    phi: Any,
    N: Any,
    seed: int,
    signal_indices: Optional[Sequence[int]] = None,
    direction: str = "null",
    fold_change: float = 1.0,
    c_t: int = 1,
) -> tuple[pd.DataFrame, set[str], pd.DataFrame]:
    """Simulate the pipeline's NB2 count model plus structural cell-sampling dropout.

    The generator matches the likelihood in nb_glm._traj_dropout_ll:
      pi_ct = exp(-N_t * lambda_ct)
      observed detection probability = (1-pi_ct) * P_NB(X >= c_t).
    A count is retained iff structural dropout did not occur and X >= c_t.

    phi may be a scalar or a callable such as a fitted nb.PhiTrend. Frequencies are
    fixed baseline frequencies; a planted fold change acts from first to last sample.
    """
    design = _sample_design(g_design_source, c_t=c_t)
    if design["patient"].nunique() != 1:
        raise ValueError("Simulate one person/design at a time")
    freqs = np.asarray(frequencies, dtype=float)
    if freqs.ndim != 1 or len(freqs) == 0 or np.any(~np.isfinite(freqs)) or np.any(freqs <= 0):
        raise ValueError("frequencies must be a non-empty positive finite 1D array")
    if direction not in {"null", "expansion", "contraction"}:
        raise ValueError("direction must be null/expansion/contraction")
    if fold_change < 1:
        raise ValueError("fold_change is the magnitude and must be >= 1")

    rng = np.random.default_rng(seed)
    days = design["day"].to_numpy(float)
    span = float(np.ptp(days))
    if span <= 0:
        raise ValueError("Need at least two distinct sampling days")
    time_s = (days - days.mean()) / span

    beta = np.zeros(len(freqs), dtype=float)
    indices = np.asarray([] if signal_indices is None else signal_indices, dtype=int)
    if direction != "null" and len(indices):
        sign = 1.0 if direction == "expansion" else -1.0
        beta[indices] = sign * np.log(float(fold_change))
    lambda_ct = freqs[:, None] * np.exp(beta[:, None] * time_s[None, :])
    offsets = design["lib"].to_numpy(float)
    cutoffs = design["c_t"].to_numpy(int)
    mu = lambda_ct * offsets[None, :]

    if callable(phi):
        phi_clone = np.asarray(phi(freqs), dtype=float)
    else:
        ph = np.asarray(phi, dtype=float)
        phi_clone = np.full(len(freqs), float(ph)) if ph.ndim == 0 else ph
    if len(phi_clone) != len(freqs) or np.any(~np.isfinite(phi_clone)) or np.any(phi_clone <= 0):
        raise ValueError("phi must yield one finite positive value per clone")

    prob = phi_clone[:, None] / (phi_clone[:, None] + mu)
    counts = rng.negative_binomial(phi_clone[:, None], prob)
    N_vec = _resolve_N(N, days)
    pi = np.exp(-N_vec[None, :] * lambda_ct)
    sampled = rng.random(counts.shape) >= pi
    observed = np.where(sampled, counts, 0)
    detected = observed >= cutoffs[None, :]

    clone_ids = np.asarray([f"SIM_{i:08d}" for i in range(len(freqs))], dtype=object)
    rows: list[dict[str, Any]] = []
    for j, sample in design.iterrows():
        ix = np.flatnonzero(detected[:, j])
        if len(ix) == 0:
            continue
        base = sample.drop(labels=["c_t"], errors="ignore").to_dict()
        for i in ix:
            row = dict(base)
            row.update({"clono": clone_ids[i], "count": int(observed[i, j])})
            rows.append(row)
    sim_df = pd.DataFrame(rows)
    sig_ids = set(clone_ids[indices].astype(str)) if direction != "null" else set()
    truth = pd.DataFrame({
        "clono": clone_ids,
        "baseline_frequency": freqs,
        "phi_true": phi_clone,
        "beta_logfold_series": beta,
        "is_signal": np.isin(np.arange(len(freqs)), indices) if direction != "null" else False,
    })
    return sim_df, sig_ids, truth


def _fit_global_phi(nb: Any, sim_df: pd.DataFrame, *, truncate: bool, cox_reid: bool,
                    min_freq_mult: float = 10.0, c_t: int = 1) -> dict:
    kwargs = dict(
        min_detect=2, min_freq_mult=min_freq_mult, cutoff_frac=1e-6,
        n_bins=1, min_bin_clones=10**9, truncate=truncate, cox_reid=cox_reid,
    )
    fit = _call_with_cutoff(nb.fit_nb_dispersion_cr, sim_df, c_t, kwargs)
    if "phi" not in fit and "phi_global" in fit:
        fit = dict(fit)
        fit["phi"] = fit["phi_global"]
    return fit


def run_phi_recovery(
    d_qc: pd.DataFrame,
    nb: Any,
    *,
    donors: Optional[Sequence[str]] = None,
    true_phis: Sequence[float] = (0.2, 0.5, 1.0),
    n_reps: int = 5,
    n_clones: int = 5_000,
    min_freq_for_pool: float = 1e-5,
    seed: int = 20261022,
    compare_estimators: bool = True,
    progress: bool = True,
    c_t: int = 1,
) -> pd.DataFrame:
    """Known-phi recovery under NB2 + independent structural dropout.

    This conditions the frequency pool on the range used for dispersion estimation.
    It therefore tests estimator recovery in the identifiable range, not unseen-clone
    frequency-distribution recovery. Three estimators are compared on identical counts.
    """
    if donors is None:
        donors = list(d_qc["patient"].drop_duplicates())
    methods = [("truncated_CR", True, True)]
    if compare_estimators:
        methods += [("untruncated_CR", False, True), ("truncated_noCR", True, False)]
    out: list[dict[str, Any]] = []

    for di, person in enumerate(donors):
        g = d_qc[d_qc["patient"] == person].copy()
        design = _sample_design(g, c_t=c_t)
        pool_all = _frequency_pool(g, c_t=c_t)
        pool = pool_all[(pool_all >= min_freq_for_pool) & (pool_all <= 0.05)]
        if len(pool) < 200:
            raise ValueError(
                f"{person}: only {len(pool)} frequencies in the recovery range "
                f">={min_freq_for_pool:g}; lower min_freq_for_pool or use fewer bins."
            )
        n_draw = min(int(n_clones), max(200, len(pool)))
        for pi, phi_true in enumerate(true_phis):
            # Fix N by sample from the raw data for this assumed phi; re-use across replicates.
            N_true = _call_with_cutoff(nb.estimate_dropout_N, g, c_t, {"phi": float(phi_true)})
            N_true = N_true[["day", "N_t"]].copy()
            for rep in range(n_reps):
                rng = np.random.default_rng(np.random.SeedSequence([seed, di, pi, rep]))
                freqs = rng.choice(pool, size=n_draw, replace=(n_draw > len(pool)))
                sim_df, _, _ = simulate_nb_dropout(
                    g, freqs, phi=float(phi_true), N=N_true,
                    seed=int(rng.integers(0, 2**32 - 1)), c_t=c_t,
                )
                for method, trunc, cr in methods:
                    row = {"donor": person, "replicate": rep, "phi_true": float(phi_true),
                           "estimator": method, "n_clones_generated": n_draw,
                           "n_rows_detected": len(sim_df)}
                    try:
                        fit = _fit_global_phi(nb, sim_df, truncate=trunc, cox_reid=cr, c_t=c_t)
                        hat = float(fit["phi"])
                        row.update({
                            "phi_hat": hat,
                            "ratio_hat_true": hat / float(phi_true),
                            "relative_error": (hat - float(phi_true)) / float(phi_true),
                            "at_bound": bool(fit.get("phi_at_bound", False)),
                            "n_clones_used": int(fit.get("n_clones", 0)),
                            "fit_status": "ok",
                        })
                    except Exception as exc:
                        row.update({"phi_hat": np.nan, "ratio_hat_true": np.nan,
                                    "relative_error": np.nan, "at_bound": np.nan,
                                    "n_clones_used": np.nan,
                                    "fit_status": f"{type(exc).__name__}: {exc}"})
                    out.append(row)
                if progress and (rep == 0 or rep + 1 == n_reps):
                    print(f"[phi recovery: {person}, true phi={phi_true:g}] {rep + 1}/{n_reps}")
    return pd.DataFrame(out)


def run_phi_trend_recovery(
    d_qc: pd.DataFrame,
    nb: Any,
    person: str,
    *,
    n_reps: int = 5,
    n_clones: int = 12_000,
    min_freq: float = 1e-5,
    max_freq: float = 1e-3,
    phi_at_reference: float = 0.6,
    true_slope: float = 0.35,
    seed: int = 20261027,
    progress: bool = True,
    c_t: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test recovery of a flat and known log-linear phi(f) trend.

    The trend is defined by log(phi)=a+b*log10(f); the rising case has phi=
    phi_at_reference at f=1e-4 and slope b=true_slope. The same dropout likelihood
    and raw thresholds are used to generate counts, then `fit_phi_trend` is refit.
    """
    if not (0 < min_freq < max_freq):
        raise ValueError("Require 0 < min_freq < max_freq")
    g = d_qc[d_qc["patient"] == person].copy()
    pool_all = _frequency_pool(g, c_t=c_t)
    pool = pool_all[(pool_all >= min_freq) & (pool_all <= max_freq)]
    if len(pool) < 200:
        raise ValueError(f"{person}: only {len(pool)} baseline frequencies in trend recovery range")

    loglo, loghi = np.log10(min_freq), np.log10(max_freq)
    logf_ref = np.log10(1e-4)
    a_rising = np.log(phi_at_reference) - true_slope * logf_ref
    truth_specs = {
        "flat": (nb.PhiTrend.constant(phi_at_reference), 0.0),
        "rising_loglinear": (nb.PhiTrend.loglinear(a_rising, true_slope, loglo, loghi), true_slope),
    }
    eval_freqs = np.geomspace(min_freq, max_freq, 7)
    rows: list[dict[str, Any]] = []

    for ti, (truth_name, (phi_true_fn, slope_true)) in enumerate(truth_specs.items()):
        N_true = _call_with_cutoff(nb.estimate_dropout_N, g, c_t, {"phi": phi_true_fn})[["day", "N_t"]].copy()
        for rep in range(n_reps):
            rng = np.random.default_rng(np.random.SeedSequence([seed, ti, rep]))
            freqs = rng.choice(pool, size=n_clones, replace=(n_clones > len(pool)))
            sim_df, _, _ = simulate_nb_dropout(
                g, freqs, phi=phi_true_fn, N=N_true,
                seed=int(rng.integers(0, 2**32 - 1)), c_t=c_t,
            )
            try:
                fit = _fit_trend_safe(
                    sim_df, nb, c_t=c_t, min_detect=2, min_freq_mult=10.0, cutoff_frac=1e-6,
                    n_bins=6, min_bin_clones=200, truncate=True, cox_reid=True,
                )
                est_fn = fit["trend"]
                est_vals = np.asarray(est_fn(eval_freqs), float)
                true_vals = np.asarray(phi_true_fn(eval_freqs), float)
                for f, tv, ev in zip(eval_freqs, true_vals, est_vals):
                    rows.append({
                        "person": person, "truth": truth_name, "replicate": rep,
                        "frequency": float(f), "phi_true": float(tv),
                        "phi_hat": float(ev), "ratio_hat_true": float(ev / tv),
                        "slope_true": float(slope_true), "slope_hat": float(fit.get("b", np.nan)),
                        "n_clones_used": int(fit.get("n_clones", 0)),
                        "n_bins_fitted": int(len(fit.get("bins", []))),
                        "fit_status": "ok",
                    })
            except Exception as exc:
                rows.append({
                    "person": person, "truth": truth_name, "replicate": rep,
                    "frequency": np.nan, "phi_true": np.nan, "phi_hat": np.nan,
                    "ratio_hat_true": np.nan, "slope_true": float(slope_true),
                    "slope_hat": np.nan, "n_clones_used": np.nan,
                    "n_bins_fitted": np.nan, "fit_status": f"{type(exc).__name__}: {exc}",
                })
            if progress and (rep == 0 or rep + 1 == n_reps):
                print(f"[phi trend recovery: {person}, {truth_name}] {rep + 1}/{n_reps}")

    runs = pd.DataFrame(rows)
    ok = runs[runs["fit_status"] == "ok"].copy()
    summary = (ok.groupby(["truth", "frequency"], as_index=False)
               .agg(n_reps=("replicate", "nunique"),
                    mean_ratio=("ratio_hat_true", "mean"),
                    median_ratio=("ratio_hat_true", "median"),
                    q025_ratio=("ratio_hat_true", lambda x: x.quantile(.025)),
                    q975_ratio=("ratio_hat_true", lambda x: x.quantile(.975)),
                    mean_slope_hat=("slope_hat", "mean"),
                    slope_true=("slope_true", "first"))) if len(ok) else pd.DataFrame()
    return runs, summary


def summarise_phi_recovery(runs: pd.DataFrame) -> pd.DataFrame:
    ok = runs[runs["fit_status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    return (ok.groupby(["donor", "phi_true", "estimator"], as_index=False)
              .agg(n_reps=("replicate", "nunique"),
                   mean_phi_hat=("phi_hat", "mean"),
                   median_phi_hat=("phi_hat", "median"),
                   mean_ratio=("ratio_hat_true", "mean"),
                   median_ratio=("ratio_hat_true", "median"),
                   rel_error_q025=("relative_error", lambda x: x.quantile(.025)),
                   rel_error_q975=("relative_error", lambda x: x.quantile(.975)),
                   rmse=("relative_error", lambda x: float(np.sqrt(np.mean(x*x)))),
                   bound_rate=("at_bound", "mean"),
                   median_clones_used=("n_clones_used", "median")))


def _fit_trend_safe(g: pd.DataFrame, nb: Any, *, c_t: int = 1, **kwargs) -> dict:
    """Fit a dispersion trend, with a constant-phi fallback for empty-bin versions."""
    fit_kwargs = dict(min_detect=2, min_freq_mult=10.0, cutoff_frac=1e-6,
                      n_bins=6, min_bin_clones=200, truncate=True, cox_reid=True)
    fit_kwargs.update(kwargs)
    try:
        result = _call_with_cutoff(nb.fit_phi_trend, g, c_t, fit_kwargs)
        if len(result.get("bins", [])) == 0:
            raise KeyError("at_bound")
        return result
    except KeyError as exc:
        if "at_bound" not in str(exc) and "KeyError" not in str(exc):
            raise
        glob = _fit_global_phi(nb, g, truncate=True, cox_reid=True,
                               min_freq_mult=fit_kwargs["min_freq_mult"], c_t=c_t)
        tr = nb.PhiTrend.constant(float(glob.get("phi", glob.get("phi_global"))))
        return {"phi_global": float(glob.get("phi", glob.get("phi_global"))),
                "bins": pd.DataFrame(columns=["log10_freq", "n_clones", "phi", "at_bound"]),
                "a": float(np.log(glob.get("phi", glob.get("phi_global")))), "b": 0.0,
                "trend": tr, "n_clones": int(glob.get("n_clones", 0)),
                "fallback_constant": True}


def fit_real_donor_trends(
    d_qc: pd.DataFrame,
    nb: Any,
    *,
    donors: Optional[Sequence[str]] = None,
    c_t: int = 1,
    **fit_kwargs,
) -> dict:
    if donors is None:
        donors = list(d_qc["patient"].drop_duplicates())
    return {
        person: _fit_trend_safe(d_qc[d_qc["patient"] == person].copy(), nb,
                                c_t=c_t, **fit_kwargs)
        for person in donors
    }


def _attach_n_miss(result: pd.DataFrame) -> pd.DataFrame:
    r = result.copy()
    if "n_detected" in r and "n_timepoints" in r:
        r["n_miss"] = (r["n_timepoints"] - r["n_detected"]).astype(int)
    return r


def _fit_test_one(g: pd.DataFrame, phi_test: Any, nb: Any, *,
                  likelihood: str = "dropout", min_freq: float = 1e-5,
                  beta_bound: float = np.log(100), n_jobs: int = 1,
                  c_t: int = 1) -> pd.DataFrame:
    kwargs = dict(likelihood=likelihood, min_freq=min_freq,
                  beta_bound=beta_bound, n_jobs=n_jobs)
    if likelihood == "dropout":
        N_hat = _call_with_cutoff(nb.estimate_dropout_N, g, c_t, {"phi": phi_test})
        kwargs["dropout_N"] = N_hat
    res = _call_with_cutoff(nb.nb_trajectory_test, g, c_t,
                            {**kwargs, "phi": phi_test})
    res = nb.add_qvalues(res, p_col="p_value", by="patient")
    return _attach_n_miss(res)


def summarise_test_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()
    r = results.copy()
    r["p_value"] = pd.to_numeric(r["p_value"], errors="coerce")
    r["q_value"] = pd.to_numeric(r["q_value"], errors="coerce")
    if "converged" not in r:
        r["converged"] = True
    r["converged"] = r["converged"].fillna(False).astype(bool)

    def stats_for(g):
        p_all = g["p_value"].dropna()
        p_ok = g.loc[g["converged"], "p_value"].dropna()
        return pd.Series({
            "n_tested": len(g),
            "convergence_failure_rate": float((~g["converged"]).mean()),
            "FPR_p05_all_finite": float((p_all < .05).mean()) if len(p_all) else np.nan,
            "FPR_p05_converged": float((p_ok < .05).mean()) if len(p_ok) else np.nan,
            "FPR_p01_converged": float((p_ok < .01).mean()) if len(p_ok) else np.nan,
            "BH_calls_q05": int((g["q_value"] < .05).sum()),
            "any_BH_call_q05": bool((g["q_value"] < .05).any()),
        })

    by_person = r.groupby("held_out", dropna=False).apply(stats_for).reset_index()
    by_miss = (r.groupby(["held_out", "n_miss"], dropna=False)
                .apply(stats_for).reset_index()) if "n_miss" in r else pd.DataFrame()
    return by_person, by_miss


def run_empirical_loo(
    d_qc: pd.DataFrame,
    nb: Any,
    *,
    donors: Optional[Sequence[str]] = None,
    likelihood: str = "dropout",
    min_freq: float = 1e-5,
    beta_bound: float = np.log(100),
    n_jobs: int = 1,
    trends: Optional[dict] = None,
    c_t: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Empirical leave-one-donor-out check on real data, using the current reference rule.

    The scalar count threshold ``c_t`` is passed to the NB functions; the input dataframe
    does not need a c_t column.
    """
    if donors is None:
        donors = list(d_qc["patient"].drop_duplicates())
    check_raw_count_data(d_qc[d_qc["patient"].isin(donors)], c_t=c_t, require_c1=(c_t == 1))
    if trends is None:
        trends = fit_real_donor_trends(d_qc, nb, donors=donors, c_t=c_t)
    outs = []
    for person in donors:
        g = d_qc[d_qc["patient"] == person].copy()
        ref = nb.reference_phi_trend(trends, exclude=person)
        res = _fit_test_one(g, ref, nb, likelihood=likelihood, min_freq=min_freq,
                            beta_bound=beta_bound, n_jobs=n_jobs, c_t=c_t)
        res["held_out"] = person
        res["null"] = "pointwise_min_other_donors"
        outs.append(res)
        print(f"[empirical LOO] {person}: {len(res):,} tested clones")
    all_res = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
    overall, by_miss = summarise_test_results(all_res)
    return all_res, overall, by_miss


def _draw_freqs(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    replace = int(n) > len(pool)
    return rng.choice(pool, size=int(n), replace=replace).astype(float)


def run_parametric_null_loo(
    d_qc: pd.DataFrame,
    nb: Any,
    trends: dict,
    *,
    donors: Optional[Sequence[str]] = None,
    n_reps: int = 3,
    n_clones: int = 20_000,
    min_freq: float = 1e-5,
    beta_bound: float = np.log(100),
    master_seed: int = 20261022,
    truth_mode: str = "donor",
    progress: bool = True,
    c_t: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parametric-bootstrap LOO test with the NB2 + structural-dropout generator.

    truth_mode='donor': generate held-out counts using that donor's fitted trend, then
        test using the minimum trend from the other donors. This stress-tests the
        heterogeneity assumption behind the conservative reference.
    truth_mode='reference': generate and test using the same leave-one-out reference;
        this checks calibration under the fitted reference model itself (more optimistic).

    The trends and effective N_t values are plug-in estimates from real data and are
    held fixed across replicates. Frequency pools are conditional on observed clones.
    """
    if truth_mode not in {"donor", "reference"}:
        raise ValueError("truth_mode must be 'donor' or 'reference'")
    if donors is None:
        donors = list(d_qc["patient"].drop_duplicates())
    pools, n_truth = {}, {}
    for di, person in enumerate(donors):
        g = d_qc[d_qc["patient"] == person].copy()
        pool = _frequency_pool(g, c_t=c_t)
        pools[person] = pool
        ref_person = nb.reference_phi_trend(trends, exclude=person)
        phi_gen = trends[person]["trend"] if truth_mode == "donor" else ref_person
        n_truth[person] = _call_with_cutoff(nb.estimate_dropout_N, g, c_t, {"phi": phi_gen})[["day", "N_t"]].copy()

    run_rows, miss_rows = [], []
    for rep in range(n_reps):
        for hi, person in enumerate(donors):
            g = d_qc[d_qc["patient"] == person].copy()
            ref = nb.reference_phi_trend(trends, exclude=person)
            phi_gen = trends[person]["trend"] if truth_mode == "donor" else ref
            rng = np.random.default_rng(np.random.SeedSequence([master_seed, rep, hi, 13]))
            freqs = _draw_freqs(pools[person], min(n_clones, len(pools[person])), rng)
            sim_df, _, _ = simulate_nb_dropout(
                g, freqs, phi=phi_gen, N=n_truth[person], seed=int(rng.integers(0, 2**32 - 1)), c_t=c_t
            )
            try:
                res = _fit_test_one(sim_df, ref, nb, likelihood="dropout", min_freq=min_freq,
                                    beta_bound=beta_bound, n_jobs=1, c_t=c_t)
                p = res["p_value"].dropna()
                q = res["q_value"]
                run_rows.append({
                    "replicate": rep, "held_out": person, "truth_mode": truth_mode,
                    "n_tested": int(p.size),
                    "FPR_p05": float((p < .05).mean()) if len(p) else np.nan,
                    "FPR_p01": float((p < .01).mean()) if len(p) else np.nan,
                    "BH_calls_q05": int((q < .05).sum()),
                    "any_BH_call_q05": bool((q < .05).any()),
                    "convergence_failure_rate": float((~res["converged"].fillna(False)).mean()) if len(res) else np.nan,
                })
                if not res.empty:
                    for n_miss, s in res.groupby("n_miss"):
                        pv = s.loc[s["converged"].fillna(False), "p_value"].dropna()
                        miss_rows.append({
                            "replicate": rep, "held_out": person, "truth_mode": truth_mode,
                            "n_miss": int(n_miss), "n_tested": int(len(pv)),
                            "FPR_p05": float((pv < .05).mean()) if len(pv) else np.nan,
                            "FPR_p01": float((pv < .01).mean()) if len(pv) else np.nan,
                        })
            except Exception as exc:
                run_rows.append({"replicate": rep, "held_out": person, "truth_mode": truth_mode,
                                 "n_tested": 0, "FPR_p05": np.nan, "FPR_p01": np.nan,
                                 "BH_calls_q05": np.nan, "any_BH_call_q05": np.nan,
                                 "convergence_failure_rate": np.nan,
                                 "error": f"{type(exc).__name__}: {exc}"})
            if progress and (hi == len(donors) - 1):
                print(f"[parametric LOO] replicate {rep + 1}/{n_reps} complete ({truth_mode})")
    runs = pd.DataFrame(run_rows)
    misses = pd.DataFrame(miss_rows)
    if len(runs):
        summary = (runs.groupby("held_out").agg(
            n_reps=("replicate", "nunique"), mean_FPR_p05=("FPR_p05", "mean"),
            q025_FPR_p05=("FPR_p05", lambda x: x.quantile(.025)),
            q975_FPR_p05=("FPR_p05", lambda x: x.quantile(.975)),
            mean_FPR_p01=("FPR_p01", "mean"), mean_BH_calls=("BH_calls_q05", "mean"),
            any_BH_call_fraction=("any_BH_call_q05", "mean"),
            mean_convergence_failure_rate=("convergence_failure_rate", "mean"),
        ).reset_index())
    else:
        summary = pd.DataFrame()
    return runs, misses, summary


def run_power_simulation(
    d_qc: pd.DataFrame,
    nb: Any,
    trends: dict,
    person: str,
    *,
    n_reps: int = 10,
    n_clones: int = 30_000,
    signal_fraction: float = 0.05,
    signal_freq_min: float = 1e-5,
    signal_freq_max: float = np.inf,
    fold_changes: Sequence[float] = (2.0, 4.0, 10.0),
    master_seed: int = 20261023,
    min_freq: float = 1e-5,
    beta_bound: float = np.log(100),
    progress: bool = True,
    c_t: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Power/FDR under the current pointwise-min phi(f) + dropout likelihood.

    Uses the full healthy reference curve as the generating and fitted NB dispersion,
    and the target person's fitted effective N_t as the plug-in input-sampling process.
    Thus this is a model-aligned, conditional power assessment, not a test of uncertainty
    in the reference curve or model misspecification.
    """
    g = d_qc[d_qc["patient"] == person].copy()
    ref = nb.reference_phi_trend(trends)
    N_true = _call_with_cutoff(nb.estimate_dropout_N, g, c_t, {"phi": ref})[["day", "N_t"]].copy()
    pool = _frequency_pool(g, c_t=c_t)
    rows: list[dict[str, Any]] = []
    scenarios = [("null", 1.0, 0.0)]
    for direction in ("expansion", "contraction"):
        scenarios.extend((direction, float(fold), float(signal_fraction))
                         for fold in fold_changes if float(fold) > 1)

    for rep in range(n_reps):
        rng = np.random.default_rng(np.random.SeedSequence([master_seed, rep, 1]))
        freqs = _draw_freqs(pool, min(n_clones, len(pool)), rng)
        eligible = np.flatnonzero((freqs >= signal_freq_min) & (freqs < signal_freq_max))
        n_signal = min(len(eligible), max(1, int(round(signal_fraction * len(eligible))))) if len(eligible) else 0
        sig_rng = np.random.default_rng(np.random.SeedSequence([master_seed, rep, 2]))
        signal_idx = sig_rng.choice(eligible, size=n_signal, replace=False) if n_signal else np.array([], dtype=int)

        for si, (direction, fold, frac) in enumerate(scenarios):
            idx = signal_idx if direction != "null" else np.array([], dtype=int)
            sim_df, signal_ids, _ = simulate_nb_dropout(
                g, freqs, phi=ref, N=N_true, seed=int(np.random.SeedSequence(
                    [master_seed, rep, si, 3]).generate_state(1, dtype=np.uint32)[0]),
                signal_indices=idx, direction=direction, fold_change=fold, c_t=c_t,
            )
            try:
                res = _fit_test_one(sim_df, ref, nb, likelihood="dropout", min_freq=min_freq,
                                    beta_bound=beta_bound, n_jobs=1, c_t=c_t)
                tested = set(res.loc[res["p_value"].notna(), "clono"].astype(str))
                signals = set(map(str, signal_ids))
                discovered = set(res.loc[res["q_value"] < .05, "clono"].astype(str))
                tested_signals = tested & signals
                true_pos = discovered & signals
                false_pos = discovered - signals
                beta_col = pd.to_numeric(res.set_index("clono")["beta"], errors="coerce") if not res.empty else pd.Series(dtype=float)
                correct = sum((beta_col.get(cl, np.nan) > 0 if direction == "expansion"
                               else beta_col.get(cl, np.nan) < 0) for cl in true_pos)
                n_disc = len(discovered)
                null_rows = res[~res["clono"].astype(str).isin(signals)]
                p_null = null_rows.loc[null_rows["converged"].fillna(False), "p_value"].dropna()
                rows.append({
                    "replicate": rep, "person": person, "direction": direction,
                    "fold_change": fold, "n_clones_generated": len(freqs),
                    "n_eligible_signal_pool": len(eligible), "n_planted": len(signals),
                    "n_planted_tested": len(tested_signals),
                    "signal_detection_rate": len(tested_signals) / len(signals) if signals else np.nan,
                    "power_unconditional_q05": len(true_pos) / len(signals) if signals else np.nan,
                    "power_conditional_q05": len(true_pos) / len(tested_signals) if tested_signals else np.nan,
                    "direction_correct_power_q05": correct / len(signals) if signals else np.nan,
                    "discoveries_q05": n_disc,
                    "false_discoveries_q05": len(false_pos),
                    "FDP_q05": len(false_pos) / n_disc if n_disc else 0.0,
                    "FPR_p05_null": float((p_null < .05).mean()) if len(p_null) else np.nan,
                    "FPR_p01_null": float((p_null < .01).mean()) if len(p_null) else np.nan,
                    "convergence_failure_rate": float((~res["converged"].fillna(False)).mean()) if len(res) else np.nan,
                })
            except Exception as exc:
                rows.append({"replicate": rep, "person": person, "direction": direction,
                             "fold_change": fold, "error": f"{type(exc).__name__}: {exc}"})
        if progress and (rep == 0 or rep + 1 == n_reps or (rep + 1) % 5 == 0):
            print(f"[power: {person}] replicate {rep + 1}/{n_reps}")

    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs, pd.DataFrame()
    summary = (runs.groupby(["person", "direction", "fold_change"], dropna=False)
               .agg(n_reps=("replicate", "nunique"),
                    mean_power_unconditional=("power_unconditional_q05", "mean"),
                    power_q025=("power_unconditional_q05", lambda x: x.quantile(.025)),
                    power_q975=("power_unconditional_q05", lambda x: x.quantile(.975)),
                    mean_power_conditional=("power_conditional_q05", "mean"),
                    mean_signal_detection_rate=("signal_detection_rate", "mean"),
                    mean_direction_correct_power=("direction_correct_power_q05", "mean"),
                    mean_FDP=("FDP_q05", "mean"),
                    fraction_runs_with_discovery=("discoveries_q05", lambda x: float((x > 0).mean())),
                    mean_FPR_p05_null=("FPR_p05_null", "mean"),
                    mean_FPR_p01_null=("FPR_p01_null", "mean"),
                    mean_discoveries=("discoveries_q05", "mean"),
                    mean_convergence_failure_rate=("convergence_failure_rate", "mean"))
               .reset_index())
    return runs, summary
