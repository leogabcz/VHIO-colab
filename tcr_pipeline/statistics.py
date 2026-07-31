"""
statistics.py

Functions for identifying non-neutral clonotype trajectories by comparing
per-patient rates of change against a reference noise model.

Pipeline
--------
1. Abundance filter: remove clonotypes that never rise above a minimum
   count threshold at enough timepoints (avoids testing noise-dominated
   sparse clonotypes).
2. Rate computation: compute log2 fold-change per day between every pair
   of consecutive timepoints for each clonotype.
3. P-value computation: two-tailed test of each rate against the reference
   noise model distribution (Normal or Student-t).
4. BH correction: adjust p-values across all rates of the patient together
   (Benjamini-Hochberg FDR control).
5. Flagging: a clonotype is marked non-neutral if any of its
   BH-adjusted rates is <= alpha.

Differences from the R pipeline
--------------------------------
- Two-tailed test: the R pipeline uses a one-directional formula that only
  detects expansions. Here we use 2 * min(CDF, 1-CDF) which symmetrically
  detects both expansions and contractions.
- Absolute days: rates are computed in absolute days, not pseudotime.
- Distribution-aware: p-values use Normal or Student-t depending on what
  was used when fitting the noise model.

Available functions
-------------------
find_non_neutral : identify non-neutral clonotypes for a single patient
"""

import numpy as np
import pandas as pd
from scipy import stats


def find_non_neutral(
    model: dict,
    norm_col: str,
    df: pd.DataFrame,
    patient: str,
    alpha: float = 0.05,
    min_timepoints: int = 2,
    abundance_filter: bool = True,
    abundance_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Identify non-neutral clonotypes for a single patient by comparing
    their rates of change against a reference noise model.

    Parameters
    ----------
    model : dict
        Output of compute_slope_noise_model or loaded via load_noise_model.
        Used to obtain the pooled reference distribution parameters
        (mu, sigma, and optionally df for Student-t).
    norm_col : str
        Normalized count column to use for rate computation and abundance
        filtering. Must match the column used when fitting the noise model.
    df : pd.DataFrame
        Canonical long-format DataFrame containing at least the patient
        specified. Must contain: patient, day, clono, count, and `norm_col`.
    patient : str
        Patient ID to extract from `df`. Only this patient is processed.
    alpha : float
        FDR threshold for BH-corrected p-values. Default 0.05.
    min_timepoints : int
        Minimum number of timepoints at which a clonotype must exceed the
        abundance threshold to pass the abundance filter. Default 2.
    abundance_filter : bool
        Whether to apply the abundance filter. Default True.
        If False, all clonotypes with at least one valid rate are tested.
    abundance_multiplier : float
        A clonotype must exceed `abundance_multiplier * mean(norm_col)`
        at >= `min_timepoints` timepoints to pass the abundance filter.
        Default 2.0 (matches R pipeline).

    Returns
    -------
    pd.DataFrame
        One row per clonotype x consecutive timepoint pair. Columns:
            patient     : str
            clono       : str
            day_t1      : int   - earlier timepoint
            day_t2      : int   - later timepoint
            rate        : float - log2 fold-change per day
            pval        : float - two-tailed raw p-value
            pval_adj    : float - BH-adjusted p-value
            non_neutral : bool  - True if any rate for this clonotype
                                  has pval_adj <= alpha

    Raises
    ------
    ValueError
        If patient is not found in df.
        If norm_col is not found in df.

    Examples
    --------
    results = find_non_neutral(model, norm_col="clr",
                               df=long_df, patient="NOUS_18")

    # Non-neutral clonotypes
    non_neutral_ids = results[results["non_neutral"]]["clono"].unique()

    # Inspect the strongest jumps
    results[results["non_neutral"]].sort_values("pval_adj").head(20)
    """
    if norm_col not in df.columns:
        raise ValueError(
            f"Column '{norm_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )
    if patient not in df["patient"].values:
        raise ValueError(
            f"Patient '{patient}' not found in df. "
            f"Available patients: {df['patient'].unique().tolist()}"
        )

    patient_df = df[df["patient"] == patient].copy()

    # --- abundance filter ---
    if abundance_filter:
        patient_df = _apply_abundance_filter(
            patient_df, norm_col, min_timepoints, abundance_multiplier
        )
    if patient_df.empty:
        return pd.DataFrame(columns=[
            "patient", "clono", "day_t1", "day_t2",
            "rate", "pval", "pval_adj", "non_neutral"
        ])

    # --- compute rates ---
    rate_rows = []
    for clono, clono_df in patient_df.groupby("clono"):
        clono_df = clono_df.sort_values("day")
        days   = clono_df["day"].values
        counts = clono_df[norm_col].values.astype(float)

        for i in range(len(days) - 1):
            c1, c2 = counts[i], counts[i + 1]
            d1, d2 = days[i], days[i + 1]
            interval = (d2 - d1)/np.max(days)

            if interval == 0 or c1 <= 0 or c2 <= 0:
                continue

            rate = np.log2(c2 / c1) / interval
            rate_rows.append({
                "patient": patient,
                "clono":   clono,
                "day_t1":  int(d1),
                "day_t2":  int(d2),
                "rate":    rate,
            })

    if not rate_rows:
        return pd.DataFrame(columns=[
            "patient", "clono", "day_t1", "day_t2",
            "rate", "pval", "pval_adj", "non_neutral"
        ])

    rate_df = pd.DataFrame(rate_rows)

    # --- two-tailed p-values ---
    rate_df["pval"] = rate_df["rate"].apply(
        lambda r: _two_tailed_pval(r, model)
    )

    # --- BH correction across all rates of this patient ---
    rate_df["pval_adj"] = _bh_correct(rate_df["pval"].values)

    # --- flag non-neutral clonotypes ---
    sig_clonos = set(
        rate_df.loc[rate_df["pval_adj"] <= alpha, "clono"].unique()
    )
    rate_df["non_neutral"] = rate_df["clono"].isin(sig_clonos)

    return rate_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_abundance_filter(
    patient_df: pd.DataFrame,
    norm_col: str,
    min_timepoints: int,
    multiplier: float,
) -> pd.DataFrame:
    """
    Remove clonotypes that do not exceed `multiplier * mean(norm_col)`
    at >= `min_timepoints` timepoints.
    Mean is computed excluding zeros (imputed/absent timepoints).
    """
    nonzero = patient_df.loc[patient_df[norm_col] > 0, norm_col]
    if nonzero.empty:
        return patient_df

    threshold = multiplier * nonzero.mean()

    keep = (
        patient_df
        .groupby("clono")[norm_col]
        .apply(lambda x: (x > threshold).sum() >= min_timepoints)
    )
    keep_clonos = keep[keep].index
    return patient_df[patient_df["clono"].isin(keep_clonos)]


def _two_tailed_pval(rate: float, model: dict) -> float:
    """
    Compute a two-tailed p-value for a single rate against the noise model.
    p = 2 * min(CDF(rate), SF(rate)) — symmetrically detects both
    expansions (high rates) and contractions (low rates).
    """
    dist  = model.get("fit_distribution", "normal")
    mu    = model["pooled"]["mu"]
    sigma = model["pooled"]["sigma"]

    if dist == "t":
        df_param = model["pooled"]["df"]
        cdf = stats.t.cdf(rate, df_param, loc=mu, scale=sigma)
    else:
        cdf = stats.norm.cdf(rate, loc=mu, scale=sigma)

    return float(2 * min(cdf, 1 - cdf))


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to an array of p-values.
 
    For each p-value ranked i (1 = smallest):
        adjusted[i] = min(1, raw_p[i] * n / i)
    Monotonicity is enforced by taking the cumulative minimum from
    largest to smallest rank.
    """
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adjusted = np.minimum(1.0, pvals * n / ranks)
    adjusted[order] = np.minimum.accumulate(adjusted[order[::-1]])[::-1]
    return adjusted

