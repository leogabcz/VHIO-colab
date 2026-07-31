"""
noise_model.py

Functions for characterizing the natural variability of TCR clonotype
trajectories from a reference set of donors (typically healthy donors).

The slope noise model computes, for each clonotype between each pair of
consecutive timepoints, a rate of change per unit day. The formula depends
on whether the input column is already on a log scale:

    linear-scale columns (e.g. 'norm_count', 'cpm', 'tmm_norm'):
        rate = log2(norm_count_t2 / norm_count_t1) / (day_t2 - day_t1)

    log-scale columns (e.g. 'clr', which can be zero or negative):
        rate = (clr_t2 - clr_t1) / (day_t2 - day_t1)

Applying the log-ratio formula to an already-log-scale column such as CLR
silently produces NaNs/garbage (log of a negative number) and drops most
of the data via the min_count filter (which assumes non-negative values).
By default this module auto-detects log-scale columns by name (anything
containing 'clr', case-insensitive) and switches formulas accordingly; this
can be overridden explicitly via the `log_scale` parameter.

This "slope" is computed for all clonotypes across all consecutive
timepoint pairs within each donor. A normal (or Student-t) distribution is
then fit to the resulting rate distribution.

The fitted parameters define what "neutral" looks like: trajectory jumps
that fall far in the tails of this distribution are candidates for
non-neutral (expanding or contracting) clonotypes.

Available functions
-------------------
compute_slope_noise_model : fit per-donor and pooled Normal/Student-t
                            distributions to the rate-of-change distribution
fit_stratified_slope_model : fit a frequency-stratified (per mean-frequency
                            bin) slope null across all patients in `df`,
                            pooled rather than per-donor
plot_noise_model          : (not yet implemented in this module)

Note: compute_fisher_non_neutral_trajectories and
compute_slope_non_neutral_trajectories are not implemented here; if you
need them, check statistics.py or noise_models_new.py.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple



def compute_slope_noise_model(
    df: pd.DataFrame,
    norm_col: str,
    min_count: float = 0.0,
    pseudotime: bool = False,
    fit_distribution: str = "normal",
    log_scale: Optional[bool] = None,
) -> dict:
    """
    Fit a slope noise model to a set of donors from a long-format DataFrame.

    For each donor, computes the rate of change (per unit day) for every
    clonotype between every pair of consecutive timepoints, then fits a
    Normal or Student-t distribution to the resulting rate distribution.
    A pooled distribution across all donors is also computed.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical long-format DataFrame with columns:
        patient, day, clono, count, and at least one normalized column.
        All donors present in df are included in the model — subset df
        before calling if you want to restrict to healthy donors only.
    norm_col : str
        Name of the normalized count column to use for rate computation
        (e.g. 'clr', 'norm_count', 'tmm_norm'). Must be present in df.
    min_count : float
        Only applied when log_scale is False. Minimum normalized count
        required at both timepoints for a rate to be included. Pairs where
        either timepoint is at or below this value are excluded. Default
        0.0 (only exclude exact zeros). Increase this to filter out very
        sparse clonotypes.
    pseudotime : bool
        If true, use pseudotime for rate computation. Else, use absolute time. Default False.
    fit_distribution : {'normal', 't'}
        Distribution family to fit to the rate values. Both are fit via
        maximum likelihood estimation (scipy.stats.norm.fit / t.fit).

        'normal' — standard Normal distribution. Simple, matches the R
                   pipeline, but underfits real TCR rate distributions,
                   which are typically sharply peaked with heavy tails
                   (high excess kurtosis).
        't'      — Student-t distribution. Has an extra degrees-of-freedom
                   parameter (df) controlling tail weight; low df gives a
                   sharp peak and heavy tails. Recommended if your rate
                   histogram visibly has a taller peak and heavier tails
                   than the fitted Normal curve (check excess kurtosis
                   with scipy.stats.kurtosis(rates) — values above ~1
                   indicate Normal is a poor fit).
    log_scale : bool, optional
        Whether norm_col is already on a log scale (e.g. CLR), in which
        case the rate is a plain difference (c2 - c1) / interval rather
        than a log2 fold-change. If None (default), this is auto-detected
        from norm_col: True if 'clr' appears in the name (case-insensitive),
        False otherwise. Get this wrong and rates will silently be NaN or
        nonsensical for log-scale columns (log2 of a negative CLR value),
        or wrong in scale for linear columns treated as log-scale.

    Returns
    -------
    dict with the following keys:

        'per_donor' : dict
            { donor_id: { 'mu', 'sigma', 'rates', 'clonotype_rates', ['df'] } }
            Per-donor fitted parameters and raw rate values.
            'rates' is a flat array across all clonotypes for that donor.
            'clonotype_rates' stores the same rates keyed by clonotype.
            'mu'/'sigma' are always present (location/scale).
            'df' (degrees of freedom) is present only when
            fit_distribution='t'.

        'pooled' : dict
            { 'mu', 'sigma', 'rates', ['df'] }
            Same structure as per_donor, pooled across all donors.

        'norm_col' : str
            The normalization column used (stored for traceability).

        'fit_distribution' : str
            The distribution family used (stored for traceability).

        'log_scale' : bool
            The resolved log_scale flag actually used (stored for
            traceability, especially when auto-detected).

    Notes
    -----
    Rate computation:
        For each clonotype in each donor, rates are computed between every
        pair of *consecutive* timepoints (sorted by day):
            log_scale=False: rate = log2(norm_t2 / norm_t1) / (day_t2 - day_t1)
            log_scale=True:  rate = (norm_t2 - norm_t1) / (day_t2 - day_t1)

        When log_scale=False, pairs are excluded if:
        - Either normalized count is <= min_count (avoids log(0) and
          division by very small/negative values)
        - The interval (day_t2 - day_t1) is zero

        When log_scale=True, min_count is not applied (log-scale values are
        legitimately negative/zero); pairs are excluded only if the interval
        is zero or either value is NaN.

    Normal fit:
        Parameters are estimated via maximum likelihood (scipy.stats.norm.fit),
        which for a normal distribution is equivalent to the sample mean and
        standard deviation.

    Examples
    --------
    # Healthy donors only, CLR (log-scale auto-detected)
    healthy_df = long_df[long_df["patient"].isin(healthy_ids)]
    model = compute_slope_noise_model(healthy_df, norm_col="clr")

    # Per-patient (use the patient themselves as reference)
    patient_df = long_df[long_df["patient"] == "PATIENT_01"]
    model = compute_slope_noise_model(patient_df, norm_col="clr")

    # Pooled across everyone, linear-scale normalized counts
    model = compute_slope_noise_model(long_df, norm_col="tmm_norm")

    # Access results
    model["pooled"]["mu"]
    model["per_donor"]["HD106"]["sigma"]
    model["per_donor"]["HD106"]["rates"]
    model["per_donor"]["HD106"]["clonotype_rates"]["clonotype_001"]
    """
    if norm_col not in df.columns:
        raise ValueError(
            f"Column '{norm_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )
    if fit_distribution not in ("normal", "t"):
        raise ValueError(f"fit_distribution must be 'normal' or 't', got '{fit_distribution}'.")
    if log_scale is None:
        log_scale = "clr" in norm_col.lower()

    per_donor = {}
    all_rates = []

    for donor, donor_df in df.groupby("patient"):
        print(f"Computing rates of  donor '{donor}'")
        donor_df = donor_df.sort_values(["clono", "day"])
        rates, clonotype_rates = _compute_rates(donor_df, norm_col, min_count, pseudotime, log_scale)

        if len(rates) < 3:
            print(f"Warning: donor '{donor}' has fewer than 3 valid rates — skipping.")
            continue

        per_donor[donor] = {
            **_fit_distribution(rates, fit_distribution),
            "rates": rates,
            "clonotype_rates": clonotype_rates,
        }
        all_rates.append(rates)

    if not all_rates:
        raise ValueError("No valid rates could be computed from the input data.")
    print(f"Computing pooled rates")
    pooled_rates = np.concatenate(all_rates)
    pooled_params = _fit_distribution(pooled_rates, fit_distribution)
    
    return {
        "per_donor": per_donor,
        "pooled": {**pooled_params, "rates": pooled_rates},
        "norm_col": norm_col,
        "fit_distribution": fit_distribution,
        "log_scale": log_scale,
    }

def _fit_distribution(rates: np.ndarray, fit_distribution: str) -> dict:
    """
    Fit either a Normal or Student-t distribution to an array of rates
    via maximum likelihood estimation (scipy.stats).

    Returns
    -------
    dict
        {'mu': float, 'sigma': float} for fit_distribution='normal'
        {'mu': float, 'sigma': float, 'df': float} for fit_distribution='t'
        ('mu'/'sigma' for 't' are the fitted loc/scale of the Student-t
        curve, not the same as the Normal's mean/std.)
    """
    if fit_distribution == "normal":
        mu, sigma = stats.norm.fit(rates)
        return {"mu": float(mu), "sigma": float(sigma)}
    else:  # Student-t
        df, loc, scale = stats.t.fit(rates)
        return {"mu": float(loc), "sigma": float(scale), "df": float(df)}


def _pdf(x: np.ndarray, params: dict):
    """
    Evaluate the fitted pdf at x for either Normal or Student-t parameters,
    and return a short label string for plotting.
    """
    if "df" in params:
        y = stats.t.pdf(x, params["df"], loc=params["mu"], scale=params["sigma"])
        label = f"μ={params['mu']:.3f}, σ={params['sigma']:.3f}, df={params['df']:.1f}"
    else:
        y = stats.norm.pdf(x, params["mu"], params["sigma"])
        label = f"μ={params['mu']:.3f}, σ={params['sigma']:.3f}"
    return y, label

def _compute_rates(
    donor_df: pd.DataFrame,
    norm_col: str,
    min_count: float,
    pseudotime: bool,
    log_scale: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute per-clonotype rate of change per day between consecutive
    timepoints for a single donor.

    If log_scale is False, rate = log2(c2 / c1) / interval (for linear-scale
    columns such as raw/normalized counts; pairs with either value <=
    min_count are skipped to avoid log(0) or division issues).

    If log_scale is True, rate = (c2 - c1) / interval (for columns already
    on a log scale, such as CLR, which can be zero or negative; min_count
    is not applied, only NaN values are skipped).
    """
    rates = []
    clonotype_rates = {}

    for clono, clono_df in donor_df.groupby("clono"):
        clono_df = clono_df.sort_values("day")
        days = clono_df["day"].values
        counts = clono_df[norm_col].values.astype(float)
        rates_for_clonotype = []

        for i in range(len(days) - 1):
            c1, c2 = counts[i], counts[i + 1]
            d1, d2 = days[i], days[i + 1]
            if pseudotime: interval = (d2 - d1)/np.max(days)
            else: interval = (d2 - d1)

            if interval == 0:
                continue

            if log_scale:
                if np.isnan(c1) or np.isnan(c2):
                    continue
                rate = (c2 - c1) / interval
            else:
                if c1 <= min_count or c2 <= min_count:
                    continue
                rate = np.log2(c2 / c1) / interval

            rates.append(rate)
            rates_for_clonotype.append(rate)

        if rates_for_clonotype:
            clonotype_rates[clono] = np.array(rates_for_clonotype)

    return np.array(rates), clonotype_rates


def _compute_rates_long(
    df: pd.DataFrame,
    values: str,
    min_count: float = 0.0,
    pseudotime: bool = False,
    log_scale: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Multi-patient counterpart to _compute_rates: computes per-transition
    slopes across every (patient, clono) group in df and returns them as a
    long DataFrame, for use by fit_stratified_slope_model.

    Returns
    -------
    pd.DataFrame with columns: patient, clono, slope
    """
    if log_scale is None:
        log_scale = "clr" in values.lower()

    rows = []
    for (patient, clono), sub in df.groupby(["patient", "clono"]):
        sub = sub.sort_values("day")
        days = sub["day"].values
        vals = sub[values].values.astype(float)

        for i in range(len(days) - 1):
            c1, c2 = vals[i], vals[i + 1]
            d1, d2 = days[i], days[i + 1]
            interval = (d2 - d1) / np.max(days) if pseudotime else (d2 - d1)
            if interval == 0:
                continue

            if log_scale:
                if np.isnan(c1) or np.isnan(c2):
                    continue
                slope = (c2 - c1) / interval
            else:
                if c1 <= min_count or c2 <= min_count:
                    continue
                slope = np.log2(c2 / c1) / interval

            rows.append({"patient": patient, "clono": clono, "slope": slope})

    return pd.DataFrame(rows, columns=["patient", "clono", "slope"])

def compute_stratified_slope_model(
    df: pd.DataFrame,
    values: str = 'clr',
    n_bins: int = 10,
    dist: str = 't',
    freq_col: Optional[str] = None,
    min_count: float = 0.0,
    pseudotime: bool = False,
    log_scale: Optional[bool] = None,
) -> dict:
    """
    Fit a frequency-stratified slope null model from healthy-donor data.
 
    Slopes (rate of change between adjacent timepoints, per unit day) are
    binned by the clonotype's mean frequency, and a separate null distribution
    is fit per bin. This captures the mean-variance relationship: noise in the
    slope is larger for rare clonotypes than abundant ones, so a single global
    null is simultaneously too permissive for abundant clones and too strict
    for rare ones.
 
    Parameters
    ----------
    df : pd.DataFrame
        Long-format healthy-donor data with columns: patient, day, clono, and
        the `values` column. Should be imputed/complete (no gaps in slopes).
    values : str
        Column used to compute slopes. Default 'clr'.
    n_bins : int
        Number of mean-frequency bins (quantile-based). Default 10.
    dist : str
        Null distribution per bin: 'normal' or 't' (Student-t via MLE).
        't' is preferred (slope data has excess kurtosis). Default 't'.
    freq_col : str, optional
        Column to stratify on (the clonotype mean frequency).
        If None, uses the mean of `values` per (patient, clono).
    min_count : float
        Only applied when log_scale is False; see _compute_rates_long.
    pseudotime : bool
        If true, use pseudotime for slope computation. Default False.
    log_scale : bool, optional
        Whether `values` is already on a log scale (e.g. CLR), in which case
        slope = (c2 - c1) / interval rather than log2(c2 / c1) / interval.
        If None (default), auto-detected from `values`: True if 'clr'
        appears in the name (case-insensitive).
 
    Returns
    -------
    dict
        {
          'bin_edges'  : np.ndarray of quantile edges defining the frequency bins,
          'params'     : list of per-bin fitted parameter dicts
                          ({'mu', 'sigma'} or {'mu', 'sigma', 'df'}), or None
                          for bins with too few observations,
          'dist'       : the distribution name,
          'values'     : the values column used,
          'freq_basis' : how the stratification frequency was computed,
          'log_scale'  : the resolved log_scale flag actually used,
        }
    """
    if log_scale is None:
        log_scale = "clr" in values.lower()

    # --- compute per-transition slopes across all patients ---
    slopes = _compute_rates_long(df, values, min_count=min_count, pseudotime=pseudotime, log_scale=log_scale)
 
    # --- stratification frequency: mean of `values` per clonotype ---
    if freq_col is None:
        mean_freq = (
            df.groupby(['patient', 'clono'])[values]
            .mean()
            .rename('mean_freq')
            .reset_index()
        )
        freq_basis = f'mean of {values} per (patient, clono)'
    else:
        mean_freq = (
            df.groupby(['patient', 'clono'])[freq_col]
            .mean()
            .rename('mean_freq')
            .reset_index()
        )
        freq_basis = f'mean of {freq_col} per (patient, clono)'
 
    slopes = slopes.merge(mean_freq, on=['patient', 'clono'], how='left')
 
    # --- quantile bin edges from the mean-frequency distribution ---
    bin_edges = np.quantile(
        mean_freq['mean_freq'].dropna().values,
        np.linspace(0, 1, n_bins + 1),
    )
    # ensure strictly increasing edges (guard against duplicate quantiles)
    bin_edges = np.unique(bin_edges)
    actual_bins = len(bin_edges) - 1
 
    slopes['bin'] = np.clip(
        np.digitize(slopes['mean_freq'].values, bin_edges[1:-1]),
        0, actual_bins - 1,
    )
 
    # --- fit a null distribution per bin ---
    if dist not in ("normal", "t"):
        raise ValueError(f"dist must be 'normal' or 't', got '{dist}'")
    params = []
    for b in range(actual_bins):
        s = slopes.loc[slopes['bin'] == b, 'slope'].dropna().values
        if len(s) < 10:
            # too few points to fit reliably — fall back to neighbouring data
            params.append(None)
            continue
        params.append(_fit_distribution(s, dist))
 
    return {
        'bin_edges': bin_edges,
        'params': params,
        'dist': dist,
        'values': values,
        'freq_basis': freq_basis,
        'log_scale': log_scale,
    }
 
