import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar  # was missing from the current file
from typing import Optional, Union
import statsmodels.api as sm

def _attach_lib(
    df: pd.DataFrame,
    count_col: str,
    norm_factors: Optional[Union[pd.Series, pd.DataFrame, str]] = None,
) -> pd.DataFrame:
    """
    Attach per-(patient, day) library size to `df` as column 'lib'.

    lib = raw read total, optionally scaled by a normalization factor:
        lib_i = raw_lib_i * norm_factor_i

    raw_lib_i is always computed from `count_col` (must be RAW counts).
    norm_factors corrects for composition effects (a few hyperexpanded
    clonotypes skewing raw depth) — same role as DESeq2's/edgeR's size
    factors, applied on top of raw depth, never in place of it. The
    correction method itself (TMM, median-of-ratios, etc.) is up to the
    caller — this function just applies whatever factor it's given.

    Parameters
    ----------
    norm_factors : one of
        - Series indexed by (patient, day) with the normalization factor
        - DataFrame with columns ['patient', 'day', 'norm_factor']
        - str: name of a column ALREADY present in `df` holding the
          per-row normalization factor (no merge needed)
        - None: lib is just the raw per-(patient, day) count sum

    Any (patient, day) group left with non-finite or non-positive lib
    is dropped, with a note on how many groups were removed.
    """
    raw_lib = (
        df.groupby(['patient', 'day'])[count_col]
        .sum().rename('raw_lib').reset_index()
    )
    df = df.merge(raw_lib, on=['patient', 'day'], how='left')

    if norm_factors is None:
        df['lib'] = df['raw_lib']
    elif isinstance(norm_factors, str):
        if norm_factors not in df.columns:
            raise ValueError(f"norm_factors='{norm_factors}' is not a column in df")
        df['lib'] = df['raw_lib'] * df[norm_factors]
    else:
        if isinstance(norm_factors, pd.Series):
            factor_df = norm_factors.rename('norm_factor').reset_index()
        else:
            factor_df = norm_factors.rename(columns={norm_factors.columns[-1]: 'norm_factor'}) \
                if 'norm_factor' not in norm_factors.columns else norm_factors
        df = df.merge(factor_df, on=['patient', 'day'], how='left')
        df['lib'] = df['raw_lib'] * df['norm_factor']

    bad = ~np.isfinite(df['lib']) | (df['lib'] <= 0)
    if bad.any():
        n_bad = df.loc[bad, ['patient', 'day']].drop_duplicates().shape[0]
        print(f"_attach_lib: dropping {n_bad} (patient, day) groups "
              f"with non-finite/zero lib")
        df = df.loc[~bad].copy()

    drop_cols = ['raw_lib']
    if 'norm_factor' in df.columns:
        drop_cols.append('norm_factor')
    return df.drop(columns=drop_cols)

def _pooled_mu(sub: pd.DataFrame, count_col: str) -> np.ndarray:
    """
    Per-observation expected mean count for one clonotype, under a shared
    relative-frequency assumption across its timepoints.

    mu_i = pooled_freq * lib_i, where pooled_freq = sum(count)/sum(lib).
    Requires `sub` to already have a 'lib' column (see _attach_lib).
    """
    pooled_freq = sub[count_col].sum() / sub['lib'].sum()
    return pooled_freq * sub['lib'].values.astype(float)


def _nb_neg_loglik(log_phi: float, x: np.ndarray, mu: np.ndarray) -> float:
    """
    Negative log-likelihood of x ~ NB(mu, phi), summed over observations.
    x MUST be raw counts — this is derived from the NB pmf, not valid on
    normalized/frequency data.
    """
    phi = np.exp(log_phi)
    p = phi / (phi + mu)
    return -np.sum(stats.nbinom.logpmf(x, phi, p))


def fit_nb_dispersion_mle(
    df: pd.DataFrame,
    count_col: str = 'count',
    norm_factors: Optional[Union[pd.Series, pd.DataFrame, str]] = None,
    min_obs: int = 2,
    floor_mu: float = 1e-8,
) -> dict:
    """
    Maximum-likelihood estimate of a single shared NB dispersion (phi) from
    healthy-donor data.

    ... (docstring unchanged) ...

    Returns
    -------
    dict
        'phi'    : float, MLE of the shared dispersion (size).
        'x_all'  : np.ndarray, the raw counts actually used in the fit.
        'mu_all' : np.ndarray, the corresponding fitted means used in the fit.
    """
    df = _attach_lib(df, count_col, norm_factors)

    xs, mus = [], []
    for _, sub in df.groupby(['patient', 'clono']):
        if len(sub) < min_obs:
            continue
        mu = _pooled_mu(sub, count_col)
        x = sub[count_col].values.astype(float)
        keep = np.isfinite(mu) & (mu > floor_mu)
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

    return {
        'phi': float(np.exp(res.x)),
        'x_all': x_all,
        'mu_all': mu_all,
    }

def nb_glm_trajectory_test(
    df: pd.DataFrame,
    dispersion,
    count_col: str = 'count',
    time_col: str = 'day',
    min_obs: int = 3,
    test: str = 'lrt',
) -> pd.DataFrame:
    """
    Per-clonotype negative binomial GLM trajectory test.

    For each (patient, clonotype) the model is:

        x_ci ~ NB(mu_ci, phi_c)
        log mu_ci = alpha_c + beta_c * t_i + log N_i

    where:
        log N_i  : fixed offset (library size) -> turns counts into rates
        alpha_c  : baseline log-frequency (intercept)
        beta_c   : log-frequency slope per unit time (the quantity of interest)
        phi_c    : NB dispersion for this clonotype — see `dispersion` below

    Neutrality is H0: beta_c = 0. The slope is tested either by a likelihood-
    ratio test against the intercept-only (+offset) model (default, robust at
    low counts) or a Wald test.

    Operates on RAW COUNTS, grid-completed with explicit zeros (a zero is an
    informative observation under NB, not missing data). No imputation, no CLR.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format, grid-completed with explicit zeros.
        Columns: patient, day (time), clono, count, lib
        ('lib' must already be attached — see add_library_sizes / _attach_lib).
    dispersion : float or dict
        float : a single shared NB dispersion (phi) used for every clonotype
                (e.g. from fit_nb_dispersion_mle).
        dict  : a trended dispersion model from fit_nb_dispersion_trended.
                Each clonotype's phi is looked up via phi_for_freq using its
                own mean frequency (sum(count)/sum(lib)), falling back to
                the model's global_phi for bins with too few clonotypes or
                a boundary-pinned fit.
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
            phi_used  : the dispersion actually used for this clonotype
            direction : 'expansion' / 'contraction' / 'flat' / 'failed'
                        ('failed' = fit did not produce a finite beta;
                        distinct from 'flat', which means beta == 0 exactly)
            converged : whether the GLM fit converged
    """
    rows = []

    for (patient, clono), sub in df.groupby(['patient', 'clono']):
        sub = sub.sort_values(time_col)
        n = len(sub)
        if n < min_obs:
            continue

        y = sub[count_col].values.astype(float)
        t = sub[time_col].values.astype(float)
        lib = sub['lib'].values.astype(float)

        if np.ptp(t) == 0 or y.sum() == 0:
            continue

        if isinstance(dispersion, dict):
            total_lib = lib.sum()
            mean_freq = y.sum() / total_lib if total_lib > 0 else 0.0
            phi = phi_for_freq(dispersion, mean_freq)
        else:
            phi = dispersion
        alpha = 1.0 / phi
        fam = sm.families.NegativeBinomial(alpha=alpha)

        offset = np.log(lib)
        t_c = t - t.mean()
        X_full = sm.add_constant(t_c)
        X_null = np.ones((n, 1))

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

        if not np.isfinite(beta):
            direction = 'failed'
        elif beta > 0:
            direction = 'expansion'
        elif beta < 0:
            direction = 'contraction'
        else:
            direction = 'flat'

        rows.append({
            'patient': patient, 'clono': clono,
            'beta': beta, 'se_beta': se, 'stat': stat, 'p_value': pval,
            'n_obs': n, 'phi_used': phi, 'direction': direction, 'converged': converged,
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

def fit_nb_dispersion_trended(
    df: pd.DataFrame,
    count_col: str = 'count',
    norm_factors: Optional[Union[pd.Series, pd.DataFrame, str]] = None,
    n_bins: int = 6,
    min_obs: int = 2,
    min_bin_n: int = 5,
    floor_mu: float = 1e-8,
) -> dict:
    """
    Mean-dependent (trended) NB dispersion: phi estimated by pooled MLE
    within quantile bins of per-clonotype mean frequency.

    count_col MUST point at RAW integer counts (default 'count') — see
    fit_nb_dispersion_mle for why. norm_factors scales `lib` (and therefore
    mu), exactly as in fit_nb_dispersion_mle; it never touches x.

    Bins are quantile-based (equal number of clonotypes per bin), not
    fixed-width in log-frequency — this keeps statistical power roughly
    balanced across bins, which matters a lot here: clonotype frequency is
    heavily right-skewed, so fixed decade bins put ~90% of clonotypes in
    one or two bins and leave the rest severely underpowered (confirmed
    empirically: quantile bins removed the boundary-pinning artifacts that
    fixed-decade bins produced at the sparse end).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format, grid-completed with explicit zeros.
        Columns: patient, day, clono, and `count_col` (raw counts).
    count_col : str
        RAW count column. Default 'count'.
    norm_factors : optional normalization factor, see fit_nb_dispersion_mle
        / _attach_lib for accepted shapes (Series, DataFrame, or column name).
    n_bins : int
        Number of quantile bins of mean frequency. Default 6 (validated
        empirically for this dataset — revisit if clonotype counts change
        substantially).
    min_obs : int
        Minimum timepoints a clonotype must have to contribute at all.
    min_bin_n : int
        Minimum clonotypes required in a bin to fit that bin's own phi;
        below this, the bin falls back to global_phi via phi_for_freq.
    floor_mu : float
        Lower floor on per-observation mean to keep the likelihood finite.

    Returns
    -------
    dict
        'logfreq_edges' : np.ndarray, quantile bin edges in log10(p_hat)
        'phi_per_bin'   : list of float or None (None = fell back to
                          global_phi, either too few clonotypes or the
                          fit pinned at the optimizer's search bounds)
        'bin_n'         : list of int, clonotypes contributing to each bin
        'bin_at_bound'  : list of bool, True if that bin's fit was
                          boundary-pinned (unreliable, treated as None)
        'global_phi'    : float, pooled MLE across all clonotypes
    """
    df = _attach_lib(df, count_col, norm_factors)
    lo, hi = np.log(1e-3), np.log(1e6)

    p_hats = []
    clono_xmu = []  # (x, mu) per clonotype, aligned with p_hats

    for _, sub in df.groupby(['patient', 'clono']):
        if len(sub) < min_obs:
            continue

        p_hat = sub[count_col].sum() / sub['lib'].sum()
        if p_hat <= 0 or not np.isfinite(p_hat):
            continue

        mu = p_hat * sub['lib'].values.astype(float)
        x = sub[count_col].values.astype(float)
        keep = np.isfinite(mu) & (mu > floor_mu)
        if keep.sum() < min_obs:
            continue

        p_hats.append(p_hat)
        clono_xmu.append((x[keep], mu[keep]))

    if not p_hats:
        raise ValueError("No clonotypes with enough observations to fit phi.")

    p_hats = np.array(p_hats)
    log_p = np.log10(p_hats)

    # --- global phi, pooled across everything ---
    x_all = np.concatenate([xm[0] for xm in clono_xmu])
    mu_all = np.concatenate([xm[1] for xm in clono_xmu])
    if np.any(~np.isfinite(mu_all)) or np.any(mu_all <= 0):
        print("Warning: mu_all contains non-finite or non-positive values "
              "after filtering — check norm_factors scale.")
    res_global = minimize_scalar(_nb_neg_loglik, args=(x_all, mu_all),
                                  bounds=(lo, hi), method='bounded')
    global_phi = float(np.exp(res_global.x))

    # --- quantile bin edges (equal n per bin) ---
    bin_edges = np.quantile(log_p, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)  # guard against duplicate quantiles
    n_bins_actual = len(bin_edges) - 1
    bin_idx = np.clip(np.digitize(log_p, bin_edges[1:-1]), 0, n_bins_actual - 1)

    phi_per_bin, bin_n, bin_at_bound = [], [], []
    for b in range(n_bins_actual):
        in_bin = bin_idx == b
        n_in_bin = int(in_bin.sum())
        bin_n.append(n_in_bin)

        if n_in_bin < min_bin_n:
            phi_per_bin.append(None)
            bin_at_bound.append(False)
            continue

        idx = np.where(in_bin)[0]
        x_bin = np.concatenate([clono_xmu[j][0] for j in idx])
        mu_bin = np.concatenate([clono_xmu[j][1] for j in idx])

        res = minimize_scalar(_nb_neg_loglik, args=(x_bin, mu_bin),
                               bounds=(lo, hi), method='bounded')
        at_bound = np.isclose(res.x, lo, atol=1e-3) or np.isclose(res.x, hi, atol=1e-3)
        bin_at_bound.append(at_bound)

        if at_bound:
            print(f"Warning: bin {b} ({bin_edges[b]:.2f}, {bin_edges[b+1]:.2f}) "
                  f"pinned at optimizer bound (n={n_in_bin}) — falling back to global_phi")
            phi_per_bin.append(None)
        else:
            phi_per_bin.append(float(np.exp(res.x)))

    return {
        'logfreq_edges': bin_edges,
        'phi_per_bin': phi_per_bin,
        'bin_n': bin_n,
        'bin_at_bound': bin_at_bound,
        'global_phi': global_phi,
    }
    
def phi_for_freq(model: dict, mean_freq: float) -> float:
    """Look up trended dispersion phi for a given mean frequency."""
    if mean_freq <= 0:
        return model['global_phi']
    edges = model['logfreq_edges']
    actual_bins = len(edges) - 1
    b = int(np.clip(np.digitize(np.log10(mean_freq), edges[1:-1]), 0, actual_bins - 1))
    phi = model['phi_per_bin'][b]
    return phi if phi is not None else model['global_phi']

