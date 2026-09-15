"""
normalization.py

Normalization functions for TCR count data.

Available functions
-------------------
library_size_normalize 
clr_normalize          : centered log-ratio normalization 
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Library-size normalization
# ---------------------------------------------------------------------------

def library_size_normalize(
    df: pd.DataFrame,
    output_col: str = "norm_count",
) -> pd.DataFrame:
    """
    Normalize counts by library size (total counts per donor x timepoint).

    Parameters
    ----------
    df : pd.DataFrame
        Canonical long-format DataFrame with columns: donor, day, clonotype, count.
    scale : float
        Scaling factor. Default 1.0 give raw fraction.
        Use 100.0 for percentage, 1e6 for counts-per-million (CPM).
    output_col : str
        Name of the new column added to hold normalized values.

    Returns
    -------
    pd.DataFrame
        Copy of the input with an additional column `output_col`.
    """
    result = df.copy()
    lib_sizes = result.groupby(["patient", "day"])["count"].transform("sum")
    result[output_col] = result["count"] / lib_sizes 
    return result


# ---------------------------------------------------------------------------
# CLR normalization
# ---------------------------------------------------------------------------

def clr_normalize(
    df: pd.DataFrame,
    zero_strategy: Literal["pseudocount", "multiplicative"] = "pseudocount",
    pseudocount: float = 0.5,
    output_col: str = "clr",
) -> pd.DataFrame:
    """
    Apply Centered Log-Ratio (CLR) normalization to raw counts within each
    (donor, day) group.

    CLR formula:
        clr(x_i) = log( x_i / geometric_mean(x) )

    This removes the compositional constraint by expressing each clonotype
    relative to the geometric mean of the sample, which is robust to a few
    dominant clonotypes.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical long-format DataFrame with columns: donor, day, clonotype, count.
    zero_strategy : {'pseudocount', 'multiplicative'}
        How to handle zero counts before log-transformation.

        'pseudocount'    — add `pseudocount` to all counts before CLR.
                           Simple but the choice of pseudocount affects rare
                           clonotypes more than abundant ones.

        'multiplicative' — replace zeros with delta = min_nonzero * 0.65,
                           then rescale non-zero values to preserve the total
                           sum (Martín-Fernández et al.). More principled than
                           a flat pseudocount.
    pseudocount : float
        Value added to all counts when zero_strategy == 'pseudocount'.
        Must be > 0. Ignored when zero_strategy == 'multiplicative'.
    output_col : str
        Name of the new column added to hold CLR values.

    Returns
    -------
    pd.DataFrame
        Copy of the input with an additional column `output_col`.
        CLR values are centered around 0 within each (patient, day) group.

    Raises
    ------
    ValueError
        If zero_strategy is not recognized, or pseudocount <= 0.
    """
    if zero_strategy not in ("pseudocount", "multiplicative"):
        raise ValueError(
            f"zero_strategy must be 'pseudocount' or 'multiplicative', got '{zero_strategy}'."
        )
    if zero_strategy == "pseudocount" and pseudocount <= 0:
        raise ValueError(f"pseudocount must be > 0, got {pseudocount}.")

    result = df.copy()
    clr_values = np.empty(len(result), dtype=float)

    for (patient, day), group_idx in result.groupby(["patient", "day"]).groups.items():
        counts = result.loc[group_idx, "count"].values.astype(float)

        if zero_strategy == "pseudocount":
            adjusted = counts + pseudocount
        else:
            adjusted = _multiplicative_zero_replacement(counts)

        log_adjusted = np.log(adjusted)
        clr_values[group_idx] = log_adjusted - np.mean(log_adjusted)

    result[output_col] = clr_values
    return result


def _multiplicative_zero_replacement(counts: np.ndarray) -> np.ndarray:
    """
    Replace zeros using the multiplicative strategy (Martín-Fernández et al.):
    zeros -> min_nonzero * 0.65, then non-zeros rescaled to preserve the sum.
    """
    adjusted = counts.copy()
    zero_mask = adjusted == 0
    n_zeros = zero_mask.sum()

    if n_zeros == 0:
        return adjusted
    if n_zeros == len(adjusted):
        return np.ones_like(adjusted, dtype=float)

    delta = adjusted[~zero_mask].min() * 0.65
    total = adjusted.sum()
    adjusted[zero_mask] = delta

    nonzero_sum = adjusted[~zero_mask].sum()
    target = total - n_zeros * delta
    if target > 0:
        adjusted[~zero_mask] *= target / nonzero_sum

    return adjusted

def tmm_normalize(
    df: pd.DataFrame,
    trim_m: float = 0.3,
    trim_a: float = 0.05,
    factor_col: str = "tmm_factor",
) -> pd.DataFrame:
    """
    Compute TMM (Trimmed Mean of M-values) normalization factors within each
    patient, using that patient's timepoints as the set of samples.

    Each timepoint's factor is computed relative to a reference timepoint
    (the one whose total count is closest to the mean across all timepoints
    of that patient). The reference timepoint gets factor = 1.0.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical long-format DataFrame with columns: patient, day, clono, count.
    trim_m : float
        Fraction of features to trim from each tail of the M-value
        (log-ratio) distribution. Default 0.3 (30% each tail, as in edgeR).
    trim_a : float
        Fraction of features to trim from each tail of the A-value
        (mean log-count) distribution. Default 0.05.
    factor_col : str
        Name of the new column added to hold the raw TMM factor.

    Returns
    -------
    pd.DataFrame
        Copy of the input with an additional column `factor_col`, the raw
        per-(patient, day) multiplicative correction factor (centered near
        1). This is the correction factor itself — NOT a normalized count —
        and is what should be passed to _attach_lib's `norm_factors`, used
        as `lib = raw_lib * factor`. It must be multiplied onto raw library
        size, never used to divide or rescale counts directly.

    References
    ----------
    Robinson, M.D. & Oshlack, A. (2010). A scaling normalization method for
    differential expression analysis of RNA-seq data. Genome Biology, 11, R25.
    """
    result = df.copy()
    result[factor_col] = np.nan

    for patient, patient_df in result.groupby("patient"):
        pivot = patient_df.pivot_table(
            index="clono", columns="day", values="count", fill_value=0
        )
        lib_sizes = pivot.sum(axis=0)
        mean_lib = lib_sizes.mean()
        ref_day = (lib_sizes - mean_lib).abs().idxmin()

        norm_factors = {}
        for day in pivot.columns:
            if day == ref_day:
                norm_factors[day] = 1.0
                continue
            counts_t   = pivot[day].values.astype(float)
            counts_ref = pivot[ref_day].values.astype(float)
            lib_t   = lib_sizes[day]
            lib_ref = lib_sizes[ref_day]

            keep = (counts_t > 0) & (counts_ref > 0)
            if keep.sum() < 3:
                norm_factors[day] = 1.0
                continue

            ct, cr = counts_t[keep], counts_ref[keep]
            log_t, log_ref = np.log2(ct / lib_t), np.log2(cr / lib_ref)
            m = log_t - log_ref
            a = 0.5 * (log_t + log_ref)

            m_lo, m_hi = np.quantile(m, [trim_m, 1 - trim_m])
            a_lo, a_hi = np.quantile(a, [trim_a, 1 - trim_a])
            trimmed = (m >= m_lo) & (m <= m_hi) & (a >= a_lo) & (a <= a_hi)
            if trimmed.sum() < 3:
                norm_factors[day] = 1.0
                continue

            weights = 1.0 / ct[trimmed] + 1.0 / cr[trimmed]
            norm_factors[day] = 2 ** np.average(m[trimmed], weights=weights)

        for day, factor in norm_factors.items():
            mask = (result["patient"] == patient) & (result["day"] == day)
            result.loc[mask, factor_col] = factor

    return result
 
