import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Optional


def _pooled_mu(sub: pd.DataFrame, count_col: str) -> np.ndarray:
    """
    Per-observation expected mean count for one clonotype, under a shared
    relative-frequency assumption across its timepoints.

    The clonotype's pooled frequency (total counts / total library size,
    across all its timepoints) is projected onto each timepoint's own
    library size to give mu_i = pooled_freq * lib_i. This is the "expected
    count under the null of constant frequency" used as the NB mean when
    fitting dispersion.

    Requires `sub` to already have a 'lib' column (see add_library_sizes).
    """
    pooled_freq = sub[count_col].sum() / sub['lib'].sum()
    return pooled_freq * sub['lib'].values.astype(float)


def _nb_neg_loglik(log_phi: float, x: np.ndarray, mu: np.ndarray) -> float:
    """
    Negative log-likelihood of x ~ NB(mu, phi), summed over observations.

    Uses the scipy.stats.nbinom parameterization n=phi (size), p=phi/(phi+mu),
    equivalent to the Gamma-Poisson mixture with mean mu and dispersion phi
    (variance = mu + mu^2/phi). Optimized in log-space (log_phi) so that
    phi = exp(log_phi) is guaranteed positive regardless of the search bounds.
    """
    phi = np.exp(log_phi)
    p = phi / (phi + mu)
    return -np.sum(stats.nbinom.logpmf(x, phi, p))


def nb_glm_trajectory_test(
    df: pd.DataFrame,
    phi: float,
    count_col: str = 'count',
    time_col: str = 'day',
    min_obs: int = 3,
    test: str = 'lrt',
) -> pd.DataFrame:
    """
    Per-clonotype negative binomial GLM trajectory test.

    For each (patient, clonotype) the model is:

        x_ci ~ NB(mu_ci, phi)
        log mu_ci = alpha_c + beta_c * t_i + log N_i

    where:
        log N_i  : fixed offset (library size) -> turns counts into rates
        alpha_c  : baseline log-frequency (intercept)
        beta_c   : log-frequency slope per unit time (the quantity of interest)
        phi      : shared dispersion (e.g. from fit_nb_dispersion_mle)

    Neutrality is H0: beta_c = 0. The slope is tested either by a likelihood-
    ratio test against the intercept-only (+offset) model (default, robust at
    low counts) or a Wald test.

    Operates on RAW COUNTS, grid-completed with explicit zeros (a zero is an
    informative observation under NB, not missing data). No imputation, no CLR.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format, grid-completed with explicit zeros.
        Columns: patient, day (time), clono, count.
    phi : float
        Shared NB dispersion (size). statsmodels uses alpha = 1/phi internally.
    count_col, time_col : str
        Column names for raw counts and the time covariate.
    min_obs : int
        Minimum timepoints a clonotype needs to fit a slope. Default 3
        (need >=2 for a line, >=3 to have any residual signal).
    test : str
        'lrt' (likelihood-ratio, recommended) or 'wald'.

    Returns
    -------
    pd.DataFrame
        One row per (patient, clono):
            beta      : fitted log-frequency slope per unit time
            se_beta   : standard error of beta
            stat      : test statistic (LRT chi2 or Wald z)
            p_value   : two-sided p-value for H0: beta = 0
            n_obs     : timepoints used
            direction : 'expansion' / 'contraction' / 'flat'
            converged : whether the GLM fit converged
    """
    alpha = 1.0 / phi  # statsmodels NB dispersion parameterisation
    fam = sm.families.NegativeBinomial(alpha=alpha)

    rows = []
    for (patient, clono), sub in df.groupby(['patient', 'clono']):
        sub = sub.sort_values(time_col)
        n = len(sub)
        if n < min_obs:
            continue

        y = sub[count_col].values.astype(float)
        t = sub[time_col].values.astype(float)
        lib = sub['lib'].values.astype(float)

        # need variation in time and at least one nonzero count to fit a slope
        if np.ptp(t) == 0 or y.sum() == 0:
            continue

        offset = np.log(lib)
        # center time for numerical stability of the intercept (slope unchanged)
        t_c = t - t.mean()

        X_full = sm.add_constant(t_c)                  # intercept + slope
        X_null = np.ones((n, 1))                       # intercept only

        beta = se = stat = pval = np.nan
        converged = False
        try:
            full = sm.GLM(y, X_full, family=fam, offset=offset).fit()
            beta = float(full.params[1])
            se = float(full.bse[1])
            converged = bool(full.converged)

            if test == 'wald':
                stat = beta / se if se > 0 else np.nan
                pval = 2.0 * stats.norm.sf(abs(stat)) if np.isfinite(stat) else np.nan
            else:  # likelihood-ratio test vs intercept-only
                null = sm.GLM(y, X_null, family=fam, offset=offset).fit()
                lr = 2.0 * (full.llf - null.llf)
                lr = max(lr, 0.0)
                stat = lr
                pval = float(stats.chi2.sf(lr, df=1))
        except Exception:
            pass

        if beta > 0:
            direction = 'expansion'
        elif beta < 0:
            direction = 'contraction'
        else:
            direction = 'flat'

        rows.append({
            'patient': patient, 'clono': clono,
            'beta': beta, 'se_beta': se, 'stat': stat, 'p_value': pval,
            'n_obs': n, 'direction': direction, 'converged': converged,
        })

    return pd.DataFrame(rows)


def add_library_sizes(df: pd.DataFrame, count_col: str = 'count') -> pd.DataFrame:
    """Attach per (patient, day) library size as column 'lib'."""
    lib = (
        df.groupby(['patient', 'day'])[count_col]
        .sum().rename('lib').reset_index()
    )
    return df.merge(lib, on=['patient', 'day'], how='left')


def bh_correct(pvals: pd.Series) -> pd.Series:
    """
    Manual Benjamini-Hochberg FDR correction (Python 3.8 compatible;
    no scipy.stats.false_discovery_control). NaNs pass through as NaN.
    Returns q-values aligned to the input index.
    """
    p = pvals.copy()
    mask = p.notna()
    pv = p[mask].values
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p downward
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.full(len(pv), np.nan)
    out[order] = q
    result = p.copy()
    result[mask] = out
    return result

def fit_nb_dispersion_mle(
    df: pd.DataFrame,
    count_col: str = 'count',
    min_obs: int = 2,
    floor_mu: float = 1e-8,
) -> float:
    """
    Maximum-likelihood estimate of a single shared NB dispersion (phi) from
    healthy-donor data.

    Each clonotype contributes its own pooled mean frequency (projected onto
    per-timepoint library sizes); phi is shared and found by maximising the
    total NB log-likelihood over all clonotype-timepoint observations.

    Operates on RAW COUNTS, grid-completed with explicit zeros (a zero is an
    informative observation under NB, not missing data).

    min_obs gates on REAL DETECTIONS (count > 0), not row count. On
    grid-completed data every clonotype has the same number of rows
    (n_timepoints), so filtering on len(sub) is a no-op; the actual
    information content of a clonotype's contribution to dispersion
    estimation depends on how many nonzero counts it has, not how many
    grid rows exist.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format, grid-completed with explicit zeros.
        Columns: patient, day, clono, count.
    count_col : str
        Raw count column.
    min_obs : int
        Minimum number of REAL (nonzero) detections a clonotype must have
        to contribute. Default 2 (need at least 2 detections to say
        anything about variance/dispersion).
    floor_mu : float
        Lower floor on per-observation mean to keep the likelihood finite
        when a clonotype's pooled frequency is ~0.

    Returns
    -------
    float
        MLE of the shared dispersion phi (size). Larger -> closer to Poisson.
    """
    lib = df.groupby(['patient', 'day'])[count_col].sum().rename('lib').reset_index()
    df = df.merge(lib, on=['patient', 'day'], how='left')

    xs, mus = [], []
    for _, sub in df.groupby(['patient', 'clono']):
        n_nonzero = int((sub[count_col] > 0).sum())
        if n_nonzero < min_obs:
            continue
        mu = _pooled_mu(sub, count_col)
        x = sub[count_col].values.astype(float)
        # drop observations where mu is essentially zero (clonotype never seen)
        keep = mu > floor_mu
        if keep.sum() < min_obs:
            continue
        xs.append(x[keep])
        mus.append(mu[keep])

    if not xs:
        raise ValueError("No clonotypes with enough observations to fit phi.")

    x_all = np.concatenate(xs)
    mu_all = np.concatenate(mus)

    res = minimize_scalar(
        _nb_neg_loglik,
        args=(x_all, mu_all),
        bounds=(np.log(1e-3), np.log(1e6)),
        method='bounded',
    )
    return float(np.exp(res.x))


def fit_nb_dispersion_trended(
    df: pd.DataFrame,
    count_col: str = 'count',
    n_bins: int = 10,
    min_obs: int = 2,
    floor_mu: float = 1e-8,
) -> dict:
    """
    Mean-dependent (trended) NB dispersion: phi estimated by MLE within
    quantile bins of mean frequency, capturing the fact that overdispersion
    varies with abundance (the NB analogue of the stratified slope null).

    Returns a model dict with bin edges (on log mean-frequency) and the per-bin
    phi, plus a lookup helper to map any mean frequency to its phi.

    min_obs gates on REAL DETECTIONS (count > 0), not row count -- see
    fit_nb_dispersion_mle docstring for why this matters on grid-completed
    data.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format, grid-completed with explicit zeros.
    count_col : str
        Raw count column.
    n_bins : int
        Number of mean-frequency quantile bins.
    min_obs : int
        Minimum number of REAL (nonzero) detections per clonotype to contribute.
    floor_mu : float
        Floor on per-observation mean.

    Returns
    -------
    dict
        {
          'logfreq_edges' : bin edges on log10 mean frequency,
          'phi_per_bin'   : list of per-bin phi (None if a bin had too few obs),
          'global_phi'    : fallback shared phi for empty/failed bins,
        }
    """
    lib = df.groupby(['patient', 'day'])[count_col].sum().rename('lib').reset_index()
    df = df.merge(lib, on=['patient', 'day'], how='left')

    # collect per-clonotype observations plus a mean-frequency for binning
    clono_obs = []   # (x_array, mu_array, mean_freq)
    mean_freqs = []
    for _, sub in df.groupby(['patient', 'clono']):
        n_nonzero = int((sub[count_col] > 0).sum())
        if n_nonzero < min_obs:
            continue
        mu = _pooled_mu(sub, count_col)
        x = sub[count_col].values.astype(float)
        keep = mu > floor_mu
        if keep.sum() < min_obs:
            continue
        total_lib = sub['lib'].sum()
        mean_freq = sub[count_col].sum() / total_lib if total_lib > 0 else 0.0
        if mean_freq <= 0:
            continue
        clono_obs.append((x[keep], mu[keep], mean_freq))
        mean_freqs.append(mean_freq)

    if not clono_obs:
        raise ValueError("No clonotypes with enough observations to fit phi.")

    mean_freqs = np.array(mean_freqs)
    log_mf = np.log10(mean_freqs)

    edges = np.quantile(log_mf, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    actual_bins = len(edges) - 1

    bin_idx = np.clip(np.digitize(log_mf, edges[1:-1]), 0, actual_bins - 1)

    # global fallback phi (pool everything)
    x_all = np.concatenate([c[0] for c in clono_obs])
    mu_all = np.concatenate([c[1] for c in clono_obs])
    res = minimize_scalar(
        _nb_neg_loglik, args=(x_all, mu_all),
        bounds=(np.log(1e-3), np.log(1e6)), method='bounded',
    )
    global_phi = float(np.exp(res.x))

    phi_per_bin = []
    for b in range(actual_bins):
        members = [clono_obs[i] for i in range(len(clono_obs)) if bin_idx[i] == b]
        if len(members) < 5:
            phi_per_bin.append(None)
            continue
        xb = np.concatenate([m[0] for m in members])
        mub = np.concatenate([m[1] for m in members])
        rb = minimize_scalar(
            _nb_neg_loglik, args=(xb, mub),
            bounds=(np.log(1e-3), np.log(1e6)), method='bounded',
        )
        phi_per_bin.append(float(np.exp(rb.x)))

    return {
        'logfreq_edges': edges,
        'phi_per_bin': phi_per_bin,
        'global_phi': global_phi,
    }


def fit_nb_dispersion_trended_smooth(
    df: pd.DataFrame,
    count_col: str = 'count',
    n_bins: int = 25,
    min_obs: int = 2,
    floor_mu: float = 1e-8,
    min_bin_members: int = 5,
    boundary_tol: float = 0.01,
    lowess_frac: float = 0.6,
    lowess_it: int = 3,
) -> dict:
    """
    Trended NB dispersion via binned MLE + loess smoothing across bins
    (edgeR-style bin.loess trend), replacing the per-bin step function
    in fit_nb_dispersion_trended with a smooth interpolated curve.

    min_obs gates on REAL DETECTIONS (count > 0), not row count -- see
    fit_nb_dispersion_mle docstring. With this fix, single-detection
    clonotypes (structural zeros everywhere else) no longer silently
    inflate low-abundance bins with near-uninformative NB(1; mu~0, phi)
    observations, which was the likely driver of the bin-1 boundary
    artifact (MLE walled at the optimizer bound).

    Pipeline:
      1. Bin clonotypes into `n_bins` quantile bins of mean log-frequency.
      2. Fit per-bin MLE dispersion (same as fit_nb_dispersion_trended).
      3. Discard bins whose MLE sits within `boundary_tol` (relative) of the
         optimizer's search bounds -- underdetermined, not real signal.
      4. Loess-smooth log(phi) vs log-frequency bin center over the
         remaining bins. Robustness iterations (`it`) further downweight
         any residual outliers.

    Returns
    -------
    dict with:
        'bin_centers'    : log10 mean-freq bin centers used (reliable bins only)
        'bin_log_phi'    : raw per-bin log(phi) MLE, for diagnostic plotting
        'smooth_x'       : x-grid of the fitted loess curve
        'smooth_log_phi' : loess-smoothed log(phi) at smooth_x
        'global_phi'     : fallback phi, unchanged from fit_nb_dispersion_trended
        'discarded_bins' : [(log_center, phi)] excluded as boundary-hugging
        'n_clono_used'   : number of clonotypes that passed the min_obs filter
        'n_clono_total'  : total (patient, clono) groups considered
    """
    lo_bound, hi_bound = 1e-3, 1e6

    lib = df.groupby(['patient', 'day'])[count_col].sum().rename('lib').reset_index()
    df = df.merge(lib, on=['patient', 'day'], how='left')

    clono_obs, mean_freqs = [], []
    n_total = 0
    for _, sub in df.groupby(['patient', 'clono']):
        n_total += 1
        n_nonzero = int((sub[count_col] > 0).sum())
        if n_nonzero < min_obs:
            continue
        mu = _pooled_mu(sub, count_col)
        x = sub[count_col].values.astype(float)
        keep = mu > floor_mu
        if keep.sum() < min_obs:
            continue
        total_lib = sub['lib'].sum()
        mean_freq = sub[count_col].sum() / total_lib if total_lib > 0 else 0.0
        if mean_freq <= 0:
            continue
        clono_obs.append((x[keep], mu[keep], mean_freq))
        mean_freqs.append(mean_freq)

    if not clono_obs:
        raise ValueError("No clonotypes with enough observations to fit phi.")

    mean_freqs = np.array(mean_freqs)
    log_mf = np.log10(mean_freqs)
    edges = np.unique(np.quantile(log_mf, np.linspace(0, 1, n_bins + 1)))
    actual_bins = len(edges) - 1
    bin_idx = np.clip(np.digitize(log_mf, edges[1:-1]), 0, actual_bins - 1)

    bin_widths = np.diff(edges)
    median_width = np.median(bin_widths)
    max_width_ratio = 4.0
    
    x_all = np.concatenate([c[0] for c in clono_obs])
    mu_all = np.concatenate([c[1] for c in clono_obs])
    res = minimize_scalar(_nb_neg_loglik, args=(x_all, mu_all),
                           bounds=(np.log(lo_bound), np.log(hi_bound)), method='bounded')
    global_phi = float(np.exp(res.x))

    reliable_centers, reliable_log_phi, discarded = [], [], []
    for b in range(actual_bins):
        members = [clono_obs[i] for i in range(len(clono_obs)) if bin_idx[i] == b]
        if len(members) < min_bin_members:
            continue
        
        if bin_widths[b] > max_width_ratio * median_width:
            log_center = float(np.log10(np.mean([m[2] for m in members])))
            discarded.append((log_center, None))  # None marks "too wide", not boundary-hugging
            continue
        
        xb = np.concatenate([m[0] for m in members])
        mub = np.concatenate([m[1] for m in members])
        rb = minimize_scalar(_nb_neg_loglik, args=(xb, mub),
                              bounds=(np.log(lo_bound), np.log(hi_bound)), method='bounded')
        phi_b = float(np.exp(rb.x))
        log_center = float(np.log10(np.mean([m[2] for m in members])))

        if phi_b <= lo_bound * (1 + boundary_tol) or phi_b >= hi_bound * (1 - boundary_tol):
            discarded.append((log_center, phi_b))
            continue

        reliable_centers.append(log_center)
        reliable_log_phi.append(np.log(phi_b))

    if len(reliable_centers) < 3:
        raise ValueError(
            "Fewer than 3 reliable bins after discarding boundary-hugging "
            "estimates -- increase n_bins, raise floor_mu, or check data density."
        )

    reliable_centers = np.array(reliable_centers)
    reliable_log_phi = np.array(reliable_log_phi)
    smoothed = lowess(reliable_log_phi, reliable_centers,
                       frac=lowess_frac, it=lowess_it, return_sorted=True)

    return {
        'bin_centers': reliable_centers,
        'bin_log_phi': reliable_log_phi,
        'smooth_x': smoothed[:, 0],
        'smooth_log_phi': smoothed[:, 1],
        'global_phi': global_phi,
        'discarded_bins': discarded,
        'n_clono_used': len(clono_obs),
        'n_clono_total': n_total,
    }
    
def pool_dispersion_trends_union(models: dict, n_grid: int = 100) -> dict:
    """
    Equal-weight pooling over the UNION of donors' ranges: each grid point
    averages only over the donors whose fitted curve actually covers it,
    rather than requiring all donors to cover the full range.
    """
    lo = min(m['smooth_x'].min() for m in models.values())
    hi = max(m['smooth_x'].max() for m in models.values())
    grid = np.linspace(lo, hi, n_grid)

    curves = np.full((len(models), n_grid), np.nan)
    for i, m in enumerate(models.values()):
        in_range = (grid >= m['smooth_x'].min()) & (grid <= m['smooth_x'].max())
        curves[i, in_range] = np.interp(grid[in_range], m['smooth_x'], m['smooth_log_phi'])

    n_covering = np.sum(~np.isnan(curves), axis=0)
    mean_curve = np.nanmean(curves, axis=0)

    return {
        'smooth_x': grid,
        'smooth_log_phi': mean_curve,
        'per_donor_log_phi': curves,   # NaN where a donor doesn't cover that x
        'n_donors_covering': n_covering,  # how many donors informed each grid point
        'global_phi': float(np.mean([m['global_phi'] for m in models.values()])),
    }
