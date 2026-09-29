"""
nb_glm_v2 -- negative binomial trajectory testing with a censored observation model.

Replaces the legacy ``nb_glm`` module (kept unchanged for comparison).

Observation model
-----------------
Counts are RAW SEQTR read counts, filtered upstream (see ``io.load_anonymized_rds``):
in-frame only, Count > 1, and frequency >= 1e-6 of the pre-filter depth D_t.
So for every sample the read cutoff is c_t = ceil(1e-6 * D_t), and a clonotype
missing from a sampled timepoint had a count somewhere in {0, ..., c_t - 1}.

    x_ct ~ NB(mu_ct, phi),   log mu_ct = log lib_t + alpha_c + beta_c * t

    lib_t = D_t * tmm_factor_t       (true depth, times TMM composition correction)

Design decisions (see methodology notes for the evidence behind each)
-------------------------------------------------------------------
* Offset uses the true pre-filter depth D_t, not the post-filter read sum, which
  understates depth by 3.5-12.5% differently per sample. TMM is computed on D_t.
* Dispersion (phi) is estimated from healthy donors on DETECTED timepoints only,
  with a Cox-Reid adjustment. Without it, estimating each clone's mean from the same
  3-5 points inflates phi by ~1.5x (Neyman-Scott; more clones do not fix it).
* Dispersion is fitted only on clones well above the cutoff. Near the cutoff the
  counts are shaped by truncation and phi is not identifiable by any estimator
  tested (plain, zero-truncated, left-truncated). ``min_freq_mult`` must be
  calibrated by simulation.
* The patient test is CENSORED: detected timepoints contribute the NB pmf, missing
  timepoints contribute P(X <= c_t - 1). No zero-filling (anticonservative, biased
  toward contractions), no cutoff-filling (conservative, attenuates slopes), no
  dropping (conservative, loses power). Validated by simulation against all three.
* phi used in the test must come from a fit that used the SAME offset construction.

Typical use
-----------
    from tcr_pipeline.io import load_anonymized_rds
    from tcr_pipeline.normalization import tmm_normalize
    from tcr_pipeline import nb_glm_v2 as nb

    healthy = nb.attach_offset(tmm_normalize(load_anonymized_rds(H), lib_col='D_t'))
    patients = nb.attach_offset(tmm_normalize(load_anonymized_rds(P), lib_col='D_t'))

    disp = nb.fit_nb_dispersion_cr(healthy, min_freq_mult=5.0)
    res = nb.nb_trajectory_test_censored(patients, disp['phi'])
    res = nb.add_qvalues(res)
"""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

CUTOFF_FRAC = 1e-6                               # upstream filter: freq >= 1e-6 of D_t
PHI_BOUNDS = (np.log(1e-3), np.log(1e6))         # search range for log(phi)
KEYS = ['patient', 'clono', 'day']


# =============================================================================
# Input checks and offsets
# =============================================================================

def _require(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")


def attach_offset(df: pd.DataFrame,
                  depth_col: str = 'D_t',
                  tmm_col: Optional[str] = 'tmm_factor',
                  out_col: str = 'lib') -> pd.DataFrame:
    """Attach the GLM library size: lib = D_t * tmm_factor.

    If ``tmm_col`` is None or absent, lib = D_t (with a warning when absent).
    Donors and patients must be built with the same call, since phi is only
    meaningful relative to the offset it was fitted with.
    """
    _require(df, [depth_col])
    d = df.copy()
    if tmm_col is not None and tmm_col in d.columns:
        d[out_col] = d[depth_col] * d[tmm_col]
    else:
        if tmm_col is not None:
            warnings.warn(f"'{tmm_col}' not found; using {depth_col} alone as the offset.")
        d[out_col] = d[depth_col].astype(float)
    bad = ~np.isfinite(d[out_col]) | (d[out_col] <= 0)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have non-finite or non-positive {out_col}")
    return d


def _check_per_sample_constant(df: pd.DataFrame, cols):
    for c in cols:
        n = df.groupby(['patient', 'day'])[c].nunique()
        if (n > 1).any():
            raise ValueError(f"'{c}' is not constant within (patient, day) samples")


# =============================================================================
# Multiple testing
# =============================================================================

def bh_correct(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values. NaNs are preserved."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def add_qvalues(res: pd.DataFrame, p_col: str = 'p_value',
                by: Optional[str] = 'patient', out_col: str = 'q_value') -> pd.DataFrame:
    """BH q-values, computed within each patient by default."""
    res = res.copy()
    if by is None:
        res[out_col] = bh_correct(res[p_col].values)
    else:
        res[out_col] = res.groupby(by)[p_col].transform(lambda s: bh_correct(s.values))
    return res


# =============================================================================
# Dispersion: Cox-Reid, detected timepoints only, identifiable clones only
# =============================================================================

def _nb_neg_loglik_cr(log_phi: float, x: np.ndarray, mu: np.ndarray,
                      starts: np.ndarray) -> float:
    """Cox-Reid adjusted NB negative log-likelihood.

    x, mu are concatenated over clones; ``starts`` gives each clone's first index.
    For an intercept-only log-link NB with offset, the information for the mean
    is sum_t w_ct with w_ct = mu_ct * phi / (phi + mu_ct); CR subtracts
    0.5 * log of it per clone.
    """
    phi = np.exp(log_phi)
    p = phi / (phi + mu)
    ll = stats.nbinom.logpmf(x, phi, p).sum()
    return -(ll - 0.5 * np.log(np.add.reduceat(mu * p, starts)).sum())


def _fit_phi(x, mu, starts):
    res = minimize_scalar(_nb_neg_loglik_cr, args=(x, mu, starts),
                          bounds=PHI_BOUNDS, method='bounded')
    at_bound = (np.isclose(res.x, PHI_BOUNDS[0], atol=1e-3)
                or np.isclose(res.x, PHI_BOUNDS[1], atol=1e-3))
    return float(np.exp(res.x)), bool(at_bound)


def _sparse_blocks(df, count_col, lib_col, depth_col, min_detect):
    """Detected rows only, one row per (patient, clono, day), clone-contiguous."""
    cols = list(dict.fromkeys([lib_col, depth_col]))
    agg = {count_col: 'sum', **{c: 'first' for c in cols}}
    d = (df[df[count_col] > 0]
         .groupby(KEYS, as_index=False, sort=True).agg(agg))
    d['n_det'] = d.groupby(['patient', 'clono'])['day'].transform('size')
    d = d[d['n_det'] >= min_detect].reset_index(drop=True)
    if len(d) == 0:
        return (np.array([]),) * 3 + (np.array([], int),) * 2
    key = d['patient'].astype(str) + '|' + d['clono'].astype(str)
    starts = np.flatnonzero((key != key.shift()).values)
    n_per = np.diff(np.append(starts, len(d)))
    return (d[count_col].to_numpy(float), d[lib_col].to_numpy(float),
            d[depth_col].to_numpy(float), starts, n_per)


def fit_nb_dispersion_cr(df: pd.DataFrame,
                         count_col: str = 'count',
                         lib_col: str = 'lib',
                         depth_col: str = 'D_t',
                         min_detect: int = 2,
                         min_freq_mult: float = 5.0,
                         cutoff_frac: float = CUTOFF_FRAC,
                         n_bins: int = 6,
                         min_bin_clones: int = 50) -> dict:
    """Shared NB dispersion from healthy donors (sparse, Cox-Reid, identifiable range).

    Parameters
    ----------
    df : long-format donor data, detected rows (sparse). Needs patient, day, clono,
        count_col, lib_col (see ``attach_offset``) and depth_col.
    min_detect : minimum detected timepoints per clone.
    min_freq_mult : keep only clones whose pooled frequency (relative to D_t) is at
        least ``min_freq_mult * cutoff_frac``. Calibrate by simulation: the smallest
        value for which phi is recovered within ~5%.
    n_bins : quantile bins over the retained frequency range, reported to check that
        a single phi is justified (flat per-bin estimates).

    Returns
    -------
    dict with 'phi', 'phi_at_bound', 'n_clones', 'n_obs', 'bins' (DataFrame), and the
    settings used, for provenance.
    """
    _require(df, ['patient', 'day', 'clono', count_col, lib_col, depth_col])
    x, L, Dd, starts, n_per = _sparse_blocks(df, count_col, lib_col, depth_col, min_detect)
    if len(n_per) == 0:
        raise ValueError("no clones with enough detections")

    sx = np.add.reduceat(x, starts)
    p_hat = sx / np.add.reduceat(L, starts)          # frequency on the offset scale
    p_depth = sx / np.add.reduceat(Dd, starts)       # frequency relative to true depth
    keep = p_depth >= min_freq_mult * cutoff_frac
    if not keep.any():
        raise ValueError("no clones above min_freq_mult * cutoff")

    obs = np.repeat(keep, n_per)
    x, L = x[obs], L[obs]
    n_per, p_hat, p_depth = n_per[keep], p_hat[keep], p_depth[keep]
    starts = np.concatenate([[0], np.cumsum(n_per)[:-1]])
    mu = np.repeat(p_hat, n_per) * L

    phi, at_bound = _fit_phi(x, mu, starts)
    if at_bound:
        warnings.warn("global phi hit the search bound: not identifiable in this subset")

    lp = np.log10(p_depth)
    edges = np.unique(np.quantile(lp, np.linspace(0, 1, n_bins + 1)))
    b_of = np.clip(np.digitize(lp, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = b_of == b
        row = {'bin_log10_freq': f"[{edges[b]:.2f},{edges[b + 1]:.2f})",
               'n_clones': int(m.sum()), 'phi': np.nan, 'at_bound': False}
        if m.sum() >= min_bin_clones:
            o = np.repeat(m, n_per)
            npb = n_per[m]
            sb = np.concatenate([[0], np.cumsum(npb)[:-1]])
            row['phi'], row['at_bound'] = _fit_phi(x[o], mu[o], sb)
        rows.append(row)

    return {'phi': phi, 'phi_at_bound': at_bound,
            'n_clones': int(keep.sum()), 'n_obs': int(len(x)),
            'bins': pd.DataFrame(rows),
            'settings': dict(min_detect=min_detect, min_freq_mult=min_freq_mult,
                             cutoff_frac=cutoff_frac, lib_col=lib_col)}


# =============================================================================
# Censored trajectory test
# =============================================================================

def _censored_ll(mu, x, det, c, phi):
    """Per-timepoint log-likelihood: pmf where detected, P(X <= c-1) where missing."""
    p = phi / (phi + mu)
    return np.where(det, stats.nbinom.logpmf(x, phi, p),
                    stats.nbinom.logcdf(c - 1, phi, p))


def _num_hessian(f, th, h=1e-3):
    k = len(th)
    H = np.zeros((k, k))
    E = np.eye(k) * h
    for i in range(k):
        for j in range(k):
            H[i, j] = (f(th + E[i] + E[j]) - f(th + E[i] - E[j])
                       - f(th - E[i] + E[j]) + f(th - E[i] - E[j])) / (4 * h * h)
    return H


def _test_one_clone(x, det, L, c, t_s, span, phi, beta_bound):
    """LRT for the slope of one clone. Time is rescaled (t_s) for conditioning."""
    a0 = np.log(x[det].sum() / L[det].sum())

    def nll_full(th):
        return -_censored_ll(L * np.exp(th[0] + th[1] * t_s), x, det, c, phi).sum()

    def nll_null(a):
        return -_censored_ll(L * np.exp(a), x, det, c, phi).sum()

    null = minimize_scalar(nll_null, bounds=(a0 - 20, a0 + 20), method='bounded')
    full = minimize(nll_full, np.array([null.x, 0.0]), method='L-BFGS-B',
                    bounds=[(a0 - 20, a0 + 20), (-beta_bound, beta_bound)])
    lr = max(2.0 * (null.fun - full.fun), 0.0)
    b_s = float(full.x[1])
    at_bound = abs(abs(b_s) - beta_bound) < 1e-6

    se = np.nan
    try:
        H = _num_hessian(nll_full, full.x)
        cov = np.linalg.inv(H)
        if cov[1, 1] > 0:
            se = float(np.sqrt(cov[1, 1])) / span
    except np.linalg.LinAlgError:
        pass

    return b_s / span, se, lr, float(stats.chi2.sf(lr, 1)), bool(full.success), at_bound


def _run_tasks(tasks, n_jobs):
    if n_jobs == 1:
        return [_test_one_clone(*t) for t in tasks]
    try:
        from joblib import Parallel, delayed
    except ImportError:
        warnings.warn("joblib not available; running sequentially")
        return [_test_one_clone(*t) for t in tasks]
    return Parallel(n_jobs=n_jobs, batch_size=256)(delayed(_test_one_clone)(*t) for t in tasks)


def nb_trajectory_test_censored(df: pd.DataFrame,
                                phi: Union[float, dict],
                                count_col: str = 'count',
                                lib_col: str = 'lib',
                                cutoff_col: str = 'c_t',
                                time_col: str = 'day',
                                min_detect: int = 3,
                                beta_bound: float = 20.0,
                                n_jobs: int = 1) -> pd.DataFrame:
    """Per-clone NB-GLM slope test with censoring below each sample's cutoff.

    Each patient's clones are laid out over that patient's sampled timepoints.
    Detected timepoints contribute the NB pmf; missing ones contribute
    P(X <= c_t - 1). H0: beta = 0, tested by likelihood ratio.

    Parameters
    ----------
    df : long-format, detected rows only (sparse is fine; the grid is built here).
        Needs patient, clono, time_col, count_col, lib_col, cutoff_col.
    phi : float, or the dict returned by ``fit_nb_dispersion_cr``.
    min_detect : minimum DETECTED timepoints (not rows) to test a clone.
    beta_bound : bound on the slope in rescaled time units (|beta| * span); hitting
        it signals a separation-like fit, flagged in 'direction'.
    n_jobs : parallel workers via joblib (1 = sequential).

    Returns
    -------
    DataFrame, one row per tested (patient, clono): beta (per unit time), se_beta
    (numerical Hessian, for display), stat (LRT), p_value, n_detected, n_timepoints,
    phi_used, direction ('expansion'/'contraction'/'flat'/'separation'/'failed'),
    converged.
    """
    if isinstance(phi, dict):
        phi = phi['phi']
    phi = float(phi)
    _require(df, ['patient', 'clono', time_col, count_col, lib_col, cutoff_col])
    _check_per_sample_constant(df.rename(columns={time_col: 'day'}), [lib_col, cutoff_col])

    frames = []
    for patient, g in df.groupby('patient'):
        days = np.sort(g[time_col].unique())
        per_day = g.groupby(time_col)[[lib_col, cutoff_col]].first().reindex(days)
        L = per_day[lib_col].to_numpy(float)
        c = per_day[cutoff_col].to_numpy(float)
        wide = (g.pivot_table(index='clono', columns=time_col, values=count_col,
                              aggfunc='sum', fill_value=0)
                  .reindex(columns=days, fill_value=0))
        X = wide.to_numpy(float)
        DET = X > 0
        n_det = DET.sum(axis=1)
        sel = np.flatnonzero(n_det >= min_detect)
        if len(sel) == 0:
            continue

        t = days.astype(float)
        span = float(np.ptp(t))
        if span == 0:
            continue
        t_s = (t - t.mean()) / span

        tasks = [(X[i], DET[i], L, c, t_s, span, phi, beta_bound) for i in sel]
        out = _run_tasks(tasks, n_jobs)
        beta, se, lr, pval, ok, atb = map(np.array, zip(*out))

        direction = np.where(~ok, 'failed',
                    np.where(atb, 'separation',
                    np.where(beta > 0, 'expansion',
                    np.where(beta < 0, 'contraction', 'flat'))))
        frames.append(pd.DataFrame({
            'patient': patient, 'clono': wide.index[sel],
            'beta': beta, 'se_beta': se, 'stat': lr, 'p_value': pval,
            'n_detected': n_det[sel], 'n_timepoints': len(days),
            'phi_used': phi, 'direction': direction, 'converged': ok}))

    if not frames:
        return pd.DataFrame(columns=['patient', 'clono', 'beta', 'se_beta', 'stat', 'p_value',
                                     'n_detected', 'n_timepoints', 'phi_used',
                                     'direction', 'converged'])
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Calibration: leave-one-donor-out null test
# =============================================================================

def leave_one_donor_out(healthy: pd.DataFrame,
                        phi_kwargs: Optional[dict] = None,
                        test_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """Fit phi without donor k, run the censored test on donor k, for every donor.

    Healthy donors are the declared null, so p-values should be close to uniform.
    """
    phi_kwargs = phi_kwargs or {}
    test_kwargs = test_kwargs or {}
    out = []
    for held in healthy['patient'].unique():
        disp = fit_nb_dispersion_cr(healthy[healthy['patient'] != held], **phi_kwargs)
        res = nb_trajectory_test_censored(healthy[healthy['patient'] == held],
                                          disp['phi'], **test_kwargs)
        out.append(res.assign(held_out=held))
    return pd.concat(out, ignore_index=True)


def calibration_summary(res: pd.DataFrame, alpha: float = 0.05,
                        by: str = 'held_out') -> pd.DataFrame:
    """Per held-out donor: false-positive rate, KS vs uniform, direction split."""
    def _one(g):
        p = g['p_value'].dropna()
        sig = g['p_value'] < alpha
        return pd.Series({
            'n_tested': len(g),
            'phi': g['phi_used'].iloc[0],
            'FPR': sig.mean(),
            'KS_vs_uniform': stats.kstest(p, 'uniform').statistic if len(p) else np.nan,
            'sig_expansion': int((sig & (g['beta'] > 0)).sum()),
            'sig_contraction': int((sig & (g['beta'] < 0)).sum()),
            'separation': int((g['direction'] == 'separation').sum()),
        })
    return res.groupby(by).apply(_one).round(3)


# =============================================================================
# Diagnostic: mean-variance decomposition
# =============================================================================

def mean_variance_decomposition(df: pd.DataFrame,
                                count_col: str = 'count',
                                depth_col: str = 'D_t',
                                cutoff_col: str = 'c_t',
                                min_ratio: float = 3.0,
                                n_bins: int = 20,
                                min_per_bin: int = 30) -> pd.DataFrame:
    """Consecutive-pair Var(f1 - f2) vs mean frequency: fit Var/m = a + b*m.

    Uses clones at least ``min_ratio`` times the cutoff in both samples.
    - eff_reads_per_cell = a / (1/D1 + 1/D2): ~1 means reads behave like independent
      draws (read-level NB is structurally right); >>1 means cell sampling dominates
      and the NB needs a cell layer. For RNA data this is kappa * (1 + CV^2).
    - C_eff_if_equal = 2/a: effective number of sampled cells.
    - phi_from_quadratic = 2/b: dispersion implied by the quadratic term.
    Donor timepoints weeks apart include biological change, so treat as bounds.

    LIMITATION (found by simulation): with strong overdispersion (phi ~ 2.5) the
    quadratic term exceeds the linear one by ~60x even at the lowest usable
    frequencies, and requiring both samples above the cutoff removes large downward
    differences at low m. The intercept 'a' is then imprecise and biased low: on pure
    NB data (true eff_reads_per_cell = 1) it came out between -26 and +0.6. So:
    * only values far above that noise band (say > 50) are evidence of a cell layer;
    * values near zero or negative mean the linear term is NOT DETECTABLE here, not
      that it is absent;
    * phi_from_quadratic ran ~12-20% above the true phi, from the same selection.
    A decisive test of the cell layer needs technical replicates (same draw), where
    the biological quadratic term is absent and the linear term is visible.
    """
    _require(df, ['patient', 'day', 'clono', count_col, depth_col, cutoff_col])
    out = []
    for patient, g in df.groupby('patient'):
        days = np.sort(g['day'].unique())
        D = g.groupby('day')[depth_col].first()
        cth = g.groupby('day')[cutoff_col].first()
        wide = g.pivot_table(index='clono', columns='day', values=count_col,
                             aggfunc='sum', fill_value=0)
        for d1, d2 in zip(days[:-1], days[1:]):
            x1, x2 = wide[d1].values, wide[d2].values
            ok = (x1 >= min_ratio * cth[d1]) & (x2 >= min_ratio * cth[d2])
            if ok.sum() < n_bins * min_per_bin:
                continue
            f1, f2 = x1[ok] / D[d1], x2[ok] / D[d2]
            m, v = (f1 + f2) / 2, (f1 - f2) ** 2
            lm = np.log10(m)
            edges = np.quantile(lm, np.linspace(0, 1, n_bins + 1))
            b = np.clip(np.digitize(lm, edges[1:-1]), 0, n_bins - 1)
            gb = (pd.DataFrame({'m': m, 'v': v, 'b': b}).groupby('b')
                    .agg(m=('m', 'mean'), v=('v', 'mean'), n=('m', 'size')))
            gb = gb[gb['n'] >= min_per_bin]
            if len(gb) < 3:
                continue
            A = np.column_stack([np.ones(len(gb)), gb['m']])
            (a, bq), *_ = np.linalg.lstsq(A, gb['v'] / gb['m'], rcond=None)
            inv_R = 1 / D[d1] + 1 / D[d2]
            out.append({'patient': patient, 'pair': f"{d1}-{d2}",
                        'n_clones': int(ok.sum()),
                        'eff_reads_per_cell': a / inv_R,
                        'C_eff_if_equal': 2 / a if a > 0 else np.nan,
                        'phi_from_quadratic': 2 / bq if bq > 0 else np.nan})
    return pd.DataFrame(out)
