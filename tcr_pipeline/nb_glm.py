"""
nb_glm_v2 -- negative binomial trajectory testing for longitudinal TCR repertoires.

Replaces the legacy ``nb_glm`` module (kept unchanged for comparison).

Observation model
-----------------
Counts are RAW SEQTR read counts, filtered upstream (see ``io.load_anonymized_rds``):
in-frame only, Count > 1, and frequency >= 1e-6 of the pre-filter depth D_t.
So for every sample the read cutoff is c_t = ceil(1e-6 * D_t), and a clonotype
missing from a sampled timepoint either had no cells in the sample (cell-sampling
dropout) or had a read count somewhere in {0, ..., c_t - 1}.

    x_ct ~ NB(mu_ct, phi),   log mu_ct = log lib_t + alpha_c + beta_c * t
    lib_t = D_t * tmm_factor_t       (true depth, times TMM composition correction)
    P(clone absent from sample t) = exp(-N_t * lambda_ct),  lambda_ct = mu_ct / lib_t

Design decisions (see methodology notes for the evidence behind each)
-------------------------------------------------------------------
* Offset uses the true pre-filter depth D_t, not the post-filter read sum.
* Sample QC (relative to the same person's other samples) is applied before anything
  else; failed libraries are removed.
* Dispersion phi is estimated on clones well above the cutoff (min_freq_mult ~ 10),
  Cox-Reid adjusted, with each clone's mean re-solved at every trial phi. Near the
  cutoff no estimator tested recovers phi.
* Donors differ in phi (about 1.75-4.0). Patients are tested with a CONSERVATIVE
  external null: the noisiest healthy donor's phi, or the patient's own phi
  (per-clone intercept + slope design, ``fit_phi_per_patient``) if that is lower.
* The patient test uses the DROPOUT likelihood (``likelihood='dropout'``): censoring
  at c_t plus per-sample cell-sampling dropout with effective input N_t estimated
  within each person (``estimate_dropout_N``). The plain censored test over-calls
  (misses are far more frequent than the NB predicts); the truncated test fails near
  the cutoff. Validated by leave-one-donor-out calibration: <= nominal for all donors.
* phi used in the test must come from a fit that used the SAME offset construction.

Typical use
-----------
    from tcr_pipeline import nb_glm_v2 as nb

    healthy = nb.attach_offset(tmm_normalize(load_anonymized_rds(H), lib_col='D_t'))
    patients = nb.attach_offset(tmm_normalize(load_anonymized_rds(P), lib_col='D_t'))

    donor_phi = nb.fit_phi_per_patient(healthy, min_freq_mult=10)
    phi_ref = donor_phi['phi_intercept_only'].min()          # conservative external null

    res = []
    for pat, sub in patients.groupby('patient'):
        Nt = nb.estimate_dropout_N(sub, phi=phi_ref)
        res.append(nb.nb_trajectory_test(sub, phi=phi_ref, likelihood='dropout',
                                         dropout_N=Nt))
    res = nb.add_qvalues(pd.concat(res))
"""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar, brentq

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
#
# Model (healthy donors, neutral):
#     x_ct ~ NB(mu_ct, phi),   mu_ct = L_t * lambda_c   (L_t = D_t * tmm_factor)
# A count is observed only if x_ct >= c_t, the per-sample detection threshold
# (c_t = ceil(cutoff_frac * D_t) for this data). Only detected rows are used, so
# each is modelled with the NB truncated at c_t, i.e. conditional on the clone's
# detection pattern. This makes the min_detect selection ignorable.
#
# For each trial phi, every clone's lambda_c is re-solved by Fisher scoring
# (true profile likelihood), and the Cox-Reid term uses the (truncated) Fisher
# information for log(lambda_c):
#     I_c = sum_t w_ct^2 Var(X | X >= c_t),   w_ct = phi / (phi + mu_ct)
# which reduces to sum_t mu_ct * phi / (phi + mu_ct) without truncation.
#


def _trunc_moments(mu: np.ndarray, phi: float, c: np.ndarray):
    """Mean, variance and tail probability of X | X >= c, X ~ NB(mu, phi) (size phi).

    c is an integer array (>= 1), one threshold per row. Uses the pmf recurrence
    P(k) = P(k-1) * (k - 1 + phi) / k * mu / (phi + mu).
    """
    p = phi / (phi + mu)
    q = mu / (phi + mu)
    pk = np.exp(phi * np.log(p))                 # P(X = 0)
    s1 = np.zeros_like(mu)
    s2 = np.zeros_like(mu)
    for k in range(1, int(c.max())):             # k = 0 adds nothing to s1, s2
        pk = pk * (k - 1 + phi) / k * q
        below = k < c
        s1 += np.where(below, k * pk, 0.0)
        s2 += np.where(below, k * k * pk, 0.0)
    tail = np.maximum(stats.nbinom.sf(c - 1, phi, p), 1e-300)   # P(X >= c)
    ex = (mu - s1) / tail
    ex2 = (mu + mu * mu / phi + mu * mu - s2) / tail
    var = np.maximum(ex2 - ex * ex, 1e-12)
    return ex, var, tail


def _nb_mu_given_phi(x: np.ndarray, L: np.ndarray, starts: np.ndarray,
                     n_per: np.ndarray, phi: float, lam0: np.ndarray,
                     c: Optional[np.ndarray] = None,
                     max_iter: int = 25, tol: float = 1e-8, max_step: float = 2.0):
    """NB MLE of each clone's log-frequency at fixed phi, optionally truncated at c.

    Fisher scoring on beta_c = log(lambda_c), warm-started at lam0 (pooled ratio).
    Score: U_c = sum_t w_ct (x_ct - E[X_ct | X_ct >= c_t]),  w_ct = phi / (phi + mu_ct).
    Info:  I_c = sum_t w_ct^2 Var(X_ct | X_ct >= c_t).
    Without truncation these are the usual NB-GLM score and Fisher information.

    Returns (mu, I, step_max): fitted means (concatenated over clones), per-clone
    information at the fit, and the largest final step (convergence check).
    """
    def moments(mu):
        if c is None:
            return mu, mu * (phi + mu) / phi
        ex, var, _ = _trunc_moments(mu, phi, c)
        return ex, var

    beta = np.log(lam0)
    step_max = np.inf
    for _ in range(max_iter):
        mu = np.repeat(np.exp(beta), n_per) * L
        w = phi / (phi + mu)
        ex, var = moments(mu)
        U = np.add.reduceat(w * (x - ex), starts)
        I = np.add.reduceat(w * w * var, starts)
        step = np.clip(U / I, -max_step, max_step)
        beta = beta + step
        step_max = float(np.max(np.abs(step)))
        if step_max < tol:
            break
    mu = np.repeat(np.exp(beta), n_per) * L
    w = phi / (phi + mu)
    _, var = moments(mu)
    I = np.add.reduceat(w * w * var, starts)
    return mu, I, step_max


def _nb_neg_loglik_cr(log_phi: float, x: np.ndarray, L: np.ndarray,
                      starts: np.ndarray, n_per: np.ndarray, lam0: np.ndarray,
                      resolve_mean: bool = True,
                      c: Optional[np.ndarray] = None,
                      cox_reid: bool = True) -> float:
    """Cox-Reid adjusted (optionally truncated) NB negative profile log-likelihood.

    resolve_mean=True : each clone's lambda_c is re-solved at this phi (true profile).
    resolve_mean=False: the fixed pooled ratio lam0 is plugged in (plug-in approximation).
    c                 : per-row detection threshold; None = no truncation.
    cox_reid          : include the Cox-Reid term (False = plain profile likelihood).
    """
    phi = np.exp(log_phi)
    if resolve_mean:
        mu, I, _ = _nb_mu_given_phi(x, L, starts, n_per, phi, lam0, c)
    else:
        mu = np.repeat(lam0, n_per) * L
        w = phi / (phi + mu)
        var = mu * (phi + mu) / phi if c is None else _trunc_moments(mu, phi, c)[1]
        I = np.add.reduceat(w * w * var, starts)
    p = phi / (phi + mu)
    ll = stats.nbinom.logpmf(x, phi, p)
    if c is not None:
        ll = ll - stats.nbinom.logsf(c - 1, phi, p)
    penalty = 0.5 * np.log(I).sum() if cox_reid else 0.0
    return -(ll.sum() - penalty)


def _fit_phi(x, L, starts, n_per, lam0, resolve_mean: bool = True,
             c: Optional[np.ndarray] = None, cox_reid: bool = True):
    res = minimize_scalar(_nb_neg_loglik_cr,
                          args=(x, L, starts, n_per, lam0, resolve_mean, c, cox_reid),
                          bounds=PHI_BOUNDS, method='bounded')
    phi = float(np.exp(res.x))
    at_bound = (np.isclose(res.x, PHI_BOUNDS[0], atol=1e-3)
                or np.isclose(res.x, PHI_BOUNDS[1], atol=1e-3))
    if resolve_mean:
        _, _, step_max = _nb_mu_given_phi(x, L, starts, n_per, phi, lam0, c)
        if step_max > 1e-6:
            warnings.warn(f"inner NB mean fit not converged at phi={phi:.3g} "
                          f"(max step {step_max:.2e})")
    return phi, bool(at_bound)


def _sparse_blocks(df, count_col, lib_col, depth_col, min_detect, c_col=None):
    """Detected rows only, one row per (patient, clono, day), clone-contiguous.

    Returns x, L, D, starts, n_per, c (c is None when c_col is None).
    """
    cols = list(dict.fromkeys([lib_col, depth_col] + ([c_col] if c_col else [])))
    agg = {count_col: 'sum', **{col: 'first' for col in cols}}
    d = (df[df[count_col] > 0]
         .groupby(KEYS, as_index=False, sort=True).agg(agg))
    d['n_det'] = d.groupby(['patient', 'clono'])['day'].transform('size')
    d = d[d['n_det'] >= min_detect].reset_index(drop=True)
    if len(d) == 0:
        return (np.array([]),) * 3 + (np.array([], int),) * 2 + (None,)
    if c_col and (d[count_col] < d[c_col]).any():
        raise ValueError("detected counts below the per-sample threshold c: check c_col")
    key = d['patient'].astype(str) + '|' + d['clono'].astype(str)
    starts = np.flatnonzero((key != key.shift()).values)
    n_per = np.diff(np.append(starts, len(d)))
    c = d[c_col].to_numpy(int) if c_col else None
    return (d[count_col].to_numpy(float), d[lib_col].to_numpy(float),
            d[depth_col].to_numpy(float), starts, n_per, c)


def fit_nb_dispersion_cr(df: pd.DataFrame,
                         count_col: str = 'count',
                         lib_col: str = 'lib',
                         depth_col: str = 'D_t',
                         min_detect: int = 2,
                         min_freq_mult: float = 5.0,
                         cutoff_frac: float = CUTOFF_FRAC,
                         n_bins: int = 6,
                         min_bin_clones: int = 50,
                         resolve_mean: bool = True,
                         truncate: bool = True,
                         c_col: str = 'c',
                         cox_reid: bool = True) -> dict:
    """Shared NB dispersion from healthy donors (sparse, truncated, Cox-Reid).

    Parameters
    ----------
    df : long-format donor data, detected rows (sparse). Needs patient, day, clono,
        count_col, lib_col (see ``attach_offset``), depth_col and, if truncate,
        c_col: the per-sample detection threshold (here ceil(cutoff_frac * D_t),
        equal to the per-sample minimum detected count).
    min_detect : minimum detected timepoints per clone. Ignorable under truncation.
    min_freq_mult : keep only clones whose pooled frequency (relative to D_t) is at
        least ``min_freq_mult * cutoff_frac``. Calibrate by simulation.
    n_bins : quantile bins over the retained frequency range, reported to check that
        a single phi is justified (flat per-bin estimates).
    resolve_mean : re-solve each clone's mean at every trial phi (true profile).
        False plugs in the fixed pooled ratio (previous behaviour).
    truncate : model detected counts as NB truncated at c (conditional on detection).
        False uses the untruncated pmf (previous behaviour).
    c_col : column holding the per-sample detection threshold.
    cox_reid : include the Cox-Reid adjustment (False = plain profile, for comparison).

    Returns
    -------
    dict with 'phi', 'phi_at_bound', 'n_clones', 'n_obs', 'bins' (DataFrame), and the
    settings used, for provenance.
    """
    _require(df, ['patient', 'day', 'clono', count_col, lib_col, depth_col]
             + ([c_col] if truncate else []))
    x, L, Dd, starts, n_per, c = _sparse_blocks(
        df, count_col, lib_col, depth_col, min_detect, c_col if truncate else None)
    if len(n_per) == 0:
        raise ValueError("no clones with enough detections")

    sx = np.add.reduceat(x, starts)
    p_hat = sx / np.add.reduceat(L, starts)          # warm start, offset scale
    p_depth = sx / np.add.reduceat(Dd, starts)       # frequency relative to true depth
    keep = p_depth >= min_freq_mult * cutoff_frac
    if not keep.any():
        raise ValueError("no clones above min_freq_mult * cutoff")

    obs = np.repeat(keep, n_per)
    x, L = x[obs], L[obs]
    if c is not None:
        c = c[obs]
    n_per, p_hat, p_depth = n_per[keep], p_hat[keep], p_depth[keep]
    starts = np.concatenate([[0], np.cumsum(n_per)[:-1]])

    phi, at_bound = _fit_phi(x, L, starts, n_per, p_hat, resolve_mean, c, cox_reid)
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
            row['phi'], row['at_bound'] = _fit_phi(
                x[o], L[o], sb, npb, p_hat[m], resolve_mean,
                None if c is None else c[o], cox_reid)
        rows.append(row)

    return {'phi': phi, 'phi_at_bound': at_bound,
            'n_clones': int(keep.sum()), 'n_obs': int(len(x)),
            'bins': pd.DataFrame(rows),
            'settings': dict(min_detect=min_detect, min_freq_mult=min_freq_mult,
                             cutoff_frac=cutoff_frac, lib_col=lib_col,
                             resolve_mean=resolve_mean, truncate=truncate,
                             c_col=c_col, cox_reid=cox_reid)}


def _phi_vector(phi, patient, days, time_col: str = 'day') -> np.ndarray:
    """Resolve a phi specification to one value per sampled day of ``patient``.

    Accepted: a scalar (all samples); {patient: scalar}; {patient: sequence aligned to
    the sorted days}; {patient: {day: phi}}; or a DataFrame with columns patient,
    time_col and phi (e.g. from ``conservative_phi``).
    """
    days = np.asarray(days)
    if isinstance(phi, pd.DataFrame):
        m = (phi[phi['patient'] == patient].set_index(time_col)['phi']
             .reindex(days).to_numpy(float))
        if np.isnan(m).any():
            raise ValueError(f"phi table lacks samples for {patient}")
        return m
    v = phi[patient] if isinstance(phi, dict) else phi
    if isinstance(v, dict):
        return np.array([float(v[d]) for d in days])
    v = np.asarray(v, float)
    if v.ndim == 0:
        return np.full(len(days), float(v))
    if len(v) != len(days):
        raise ValueError(f"phi for {patient} has {len(v)} values for {len(days)} samples")
    return v


# =============================================================================
# Trajectory test: censored or truncated NB likelihood, bounded slope
# =============================================================================
#


def _traj_censored_ll(mu, x, det, c, phi):
    """Per-timepoint log-likelihood: pmf where detected, P(X <= c-1) where missing."""
    p = phi / (phi + mu)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(det, stats.nbinom.logpmf(x, phi, p),
                        stats.nbinom.logcdf(c - 1, phi, p))


def _traj_truncated_ll(mu, x, det, c, phi):
    """Per-timepoint log-likelihood, detected rows only: log P(X = x | X >= c).
    Missing timepoints contribute 0, i.e. they carry no information."""
    p = phi / (phi + mu)
    with np.errstate(divide='ignore', invalid='ignore'):
        ll = stats.nbinom.logpmf(x, phi, p) - stats.nbinom.logsf(c - 1, phi, p)
    return np.where(det, ll, 0.0)


def _traj_dropout_ll(mu, x, det, c, phi, N, L):
    """Per-timepoint log-likelihood with cell-sampling dropout.

    A clone is absent from sample t with probability pi_t = exp(-N_t * lambda_t),
    lambda_t = mu_t / L_t (frequency on the offset scale), N_t = effective input of the
    sample. Otherwise counts are NB(mu_t, phi). Detected: (1 - pi_t) NB(x);
    missing: pi_t + (1 - pi_t) P(X <= c_t - 1).
    """
    p = phi / (phi + mu)
    pi = np.exp(-N * mu / L)
    with np.errstate(divide='ignore', invalid='ignore'):
        det_ll = np.log1p(-pi) + stats.nbinom.logpmf(x, phi, p)
        miss_ll = np.log(pi + (1 - pi) * stats.nbinom.cdf(c - 1, phi, p))
    return np.where(det, det_ll, miss_ll)


_TRAJ_LL = {'censored': _traj_censored_ll, 'truncated': _traj_truncated_ll,
            'dropout': _traj_dropout_ll}


def _traj_fit_one(x, det, L, c, t_s, span, phi, beta_bound, likelihood, N=None):
    """Fit one clone: full model (alpha, b) and null (alpha, b = 0), LRT on 1 df.

    mu_t = L_t * exp(alpha + b * t_s), with t_s = (t - mean) / span, so b = beta * span
    is the log fold change over the whole series; |b| <= beta_bound.
    Returns (beta per unit time, se_beta, LR, p_value, converged, at_bound).
    """
    if likelihood == 'dropout':
        ll_fn = lambda mu, x_, d_, c_, ph: _traj_dropout_ll(mu, x_, d_, c_, ph, N, L)
    else:
        ll_fn = _TRAJ_LL[likelihood]
    a0 = np.log(max(x[det].sum(), 1.0) / L[det].sum())

    def nll(theta):
        mu = L * np.exp(theta[0] + theta[1] * t_s)
        v = -ll_fn(mu, x, det, c, phi).sum()
        return v if np.isfinite(v) else 1e300

    a_bounds = (a0 - 30.0, a0 + 30.0)

    # null: alpha only
    r0 = minimize_scalar(lambda a: nll((a, 0.0)), bounds=a_bounds, method='bounded')
    nll0 = r0.fun

    # full: alpha and b, a few starts so a bound-hugging optimum is not missed
    best = None
    for b_start in (0.0, 0.5 * beta_bound, -0.5 * beta_bound):
        r = minimize(nll, x0=np.array([r0.x, b_start]), method='L-BFGS-B',
                     bounds=[a_bounds, (-beta_bound, beta_bound)])
        if best is None or r.fun < best.fun:
            best = r
    nll1 = min(best.fun, nll0)                      # the full model nests the null
    lr = max(2.0 * (nll0 - nll1), 0.0)
    pval = float(stats.chi2.sf(lr, 1))
    a_hat, b_hat = best.x
    at_bound = bool(np.isclose(abs(b_hat), beta_bound, rtol=1e-3))
    converged = bool(best.success or best.fun <= nll0) and np.isfinite(nll0)

    # se of b from a numerical Hessian (display only; NaN at the bound or if not PD)
    se_b = np.nan
    if not at_bound:
        h = 1e-4
        th = np.array([a_hat, b_hat])
        H = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                ei, ej = np.eye(2)[i] * h, np.eye(2)[j] * h
                H[i, j] = (nll(th + ei + ej) - nll(th + ei - ej)
                           - nll(th - ei + ej) + nll(th - ei - ej)) / (4 * h * h)
        try:
            cov = np.linalg.inv(H)
            if cov[1, 1] > 0:
                se_b = float(np.sqrt(cov[1, 1]))
        except np.linalg.LinAlgError:
            pass
    return b_hat / span, se_b / span, lr, pval, converged, at_bound


def _traj_fit_quad(x, det, L, c, t_s, span, phi, beta_bound, likelihood, N=None,
                   shape_min_logfold=np.log(2)):
    """Fit one clone with a quadratic trajectory: full (alpha, b1, b2) vs null (alpha).

    log mu_t = log L_t + alpha + b1 * s_t + b2 * q_t,  s_t = (t - mean) / span,
    q_t = s_t^2 - mean(s^2) (centred, so b1 keeps its meaning as the overall trend).
    Bounds: |b1| <= beta_bound; |b2| * range(q) <= beta_bound (the curvature alone can
    move the curve by at most beta_bound in log). LRT on 2 df.

    Shape of the fitted curve over the sampled days:
      'peak' : interior maximum at least shape_min_logfold above both ends (rise and fall)
      'dip'  : interior minimum at least shape_min_logfold below both ends
      'up' / 'down' : otherwise, by the sign of last - first
    peak_support: for peaks/dips, number of DETECTED timepoints whose observed log
    frequency is within log(2) of the observed extreme (>= 2: the excursion is carried
    by more than one sample; 1: a single sample, lower confidence).

    Returns (beta1 per unit time, beta2, LR, p_value, converged, at_bound, shape,
    extreme_index, amplitude_logfold, peak_support).
    """
    if likelihood == 'dropout':
        ll_fn = lambda mu, x_, d_, c_, ph: _traj_dropout_ll(mu, x_, d_, c_, ph, N, L)
    else:
        ll_fn = _TRAJ_LL[likelihood]
    a0 = np.log(max(x[det].sum(), 1.0) / L[det].sum())
    q_s = t_s ** 2 - np.mean(t_s ** 2)
    b2_bound = beta_bound / max(np.ptp(q_s), 1e-12)

    def nll(theta):
        mu = L * np.exp(theta[0] + theta[1] * t_s + theta[2] * q_s)
        v = -ll_fn(mu, x, det, c, phi).sum()
        return v if np.isfinite(v) else 1e300

    a_bounds = (a0 - 30.0, a0 + 30.0)
    r0 = minimize_scalar(lambda a: nll((a, 0.0, 0.0)), bounds=a_bounds, method='bounded')
    nll0 = r0.fun
    best = None
    starts = [(0.0, 0.0), (0.5 * beta_bound, 0.0), (-0.5 * beta_bound, 0.0),
              (0.0, 0.5 * b2_bound), (0.0, -0.5 * b2_bound)]
    for b1s, b2s in starts:
        r = minimize(nll, x0=np.array([r0.x, b1s, b2s]), method='L-BFGS-B',
                     bounds=[a_bounds, (-beta_bound, beta_bound), (-b2_bound, b2_bound)])
        if best is None or r.fun < best.fun:
            best = r
    nll1 = min(best.fun, nll0)
    lr = max(2.0 * (nll0 - nll1), 0.0)
    pval = float(stats.chi2.sf(lr, 2))
    _, b1, b2 = best.x
    at_bound = bool(np.isclose(abs(b1), beta_bound, rtol=1e-3)
                    or np.isclose(abs(b2), b2_bound, rtol=1e-3))
    converged = bool(best.success or best.fun <= nll0) and np.isfinite(nll0)

    f = b1 * t_s + b2 * q_s                          # fitted log-curve (relative)
    kmax, kmin = int(np.argmax(f)), int(np.argmin(f))
    interior = lambda k: 0 < k < len(f) - 1
    rise = f[kmax] - max(f[0], f[-1])
    fall = min(f[0], f[-1]) - f[kmin]
    if interior(kmax) and rise >= shape_min_logfold and rise >= fall:
        shape, k_ext = 'peak', kmax
    elif interior(kmin) and fall >= shape_min_logfold:
        shape, k_ext = 'dip', kmin
    else:
        shape, k_ext = ('up' if f[-1] > f[0] else 'down'), (len(f) - 1 if f[-1] > f[0] else 0)
    amplitude = float(f.max() - f.min())

    support = np.nan
    if shape in ('peak', 'dip') and det.sum() > 0:
        lf = np.full(len(x), np.nan)
        lf[det] = np.log(x[det] / L[det])
        if shape == 'peak':
            support = int(np.sum(lf[det] >= np.nanmax(lf) - np.log(2)))
        else:
            support = int(np.sum(lf[det] <= np.nanmin(lf) + np.log(2)))
    return (b1 / span, b2, lr, pval, converged, at_bound, shape, k_ext, amplitude, support)


def _traj_run_tasks(tasks, n_jobs=1, fn=None):
    fn = _traj_fit_one if fn is None else fn
    if n_jobs == 1:
        return [fn(*t) for t in tasks]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(delayed(fn)(*t) for t in tasks)


def nb_trajectory_test(df: pd.DataFrame,
                       phi: Union[float, dict],
                       likelihood: str = 'truncated',
                       count_col: str = 'count',
                       lib_col: str = 'lib',
                       cutoff_col: str = 'c_t',
                       time_col: str = 'day',
                       min_detect: int = 3,
                       beta_bound: float = np.log(100),
                       dropout_N: Optional[pd.DataFrame] = None,
                       min_freq: Optional[float] = None,
                       depth_col: str = 'D_t',
                       design: str = 'slope',
                       n_jobs: int = 1) -> pd.DataFrame:
    """Per-clone NB trajectory test with a censored, truncated or dropout likelihood.

    Each patient's clones are laid out over that patient's sampled timepoints.
    H0: constant frequency, tested by likelihood ratio against
      design='slope'     : a log-linear trend (1 df), or
      design='quadratic' : a log-quadratic trajectory (2 df), which also detects a single
                           rise-and-fall ('peak') or fall-and-rise ('dip').
    The quadratic design adds columns beta2, shape ('peak'/'dip'/'up'/'down', or
    'failed'), at_bound (the fit reached beta_bound: typically a clone absent at one
    end, i.e. emerging or disappearing; its amplitude is then a lower bound),
    extreme_day, amplitude_fold (max/min of the fitted curve) and peak_support.

    Parameters
    ----------
    df : long-format, detected rows only (sparse is fine; the grid is built here).
        Needs patient, clono, time_col, count_col, lib_col, cutoff_col.
    phi : float, the dict returned by ``fit_nb_dispersion_cr``, or a dict
        {patient: phi} for per-patient values.
    likelihood : 'truncated' - only detected timepoints contribute, as
                     P(X = x | X >= c_t); misses carry no information (robust to dropout).
                 'censored'  - missing timepoints also contribute P(X <= c_t - 1).
                 'dropout'   - censored, plus cell-sampling dropout: a clone is absent from
                     sample t with probability exp(-N_t * lambda_t). Needs dropout_N.
    min_detect : minimum DETECTED timepoints to test a clone.
    beta_bound : bound on |beta| * span, i.e. the log fold change over the whole series
        (log(100) = up to 100-fold). Fits that reach it are flagged 'separation'.
    dropout_N : for likelihood='dropout': the table from ``estimate_dropout_N``
        (patient, time_col, N_t).
    min_freq : independent abundance filter (as edgeR's filterByExpr): test only clones
        whose average frequency over the person's sampled days,
        sum_t count / sum_t D_t (misses counted as 0), is >= min_freq. The filter
        uses overall abundance, not the trajectory, so it does not bias the test; it
        removes clones with essentially no power, which lightens the BH correction
        for the rest. None = no filter. depth_col (D_t) is used for the frequency.
    n_jobs : parallel workers via joblib (1 = sequential).

    Returns
    -------
    DataFrame, one row per tested (patient, clono): beta (per unit time), se_beta
    (numerical Hessian, display only), stat (LRT), p_value, n_detected, n_timepoints,
    phi_used, direction ('expansion'/'contraction'/'flat'/'separation'/'failed'),
    converged, likelihood.
    """
    if likelihood not in _TRAJ_LL:
        raise ValueError("likelihood must be 'censored', 'truncated' or 'dropout'")
    if likelihood == 'dropout' and dropout_N is None:
        raise ValueError("likelihood='dropout' needs dropout_N (see estimate_dropout_N)")
    if isinstance(phi, dict) and 'phi' in phi and not isinstance(phi['phi'], dict):
        phi = float(phi['phi'])                       # result of fit_nb_dispersion_cr
    _require(df, ['patient', 'clono', time_col, count_col, lib_col, cutoff_col])
    _check_per_sample_constant(df.rename(columns={time_col: 'day'}), [lib_col, cutoff_col])

    cols = ['patient', 'clono', 'beta', 'se_beta', 'stat', 'p_value', 'n_detected',
            'n_timepoints', 'phi_used', 'direction', 'converged', 'likelihood']
    frames = []
    for patient, g in df.groupby('patient'):
        days = np.sort(g[time_col].unique())
        phi_p = _phi_vector(phi, patient, days, time_col)
        dcols = [lib_col, cutoff_col] + ([depth_col] if depth_col in g.columns else [])
        per_day = g.groupby(time_col)[dcols].first().reindex(days)
        L = per_day[lib_col].to_numpy(float)
        c = per_day[cutoff_col].to_numpy(float)
        Dd = per_day[depth_col].to_numpy(float) if depth_col in g.columns else L
        wide = (g.pivot_table(index='clono', columns=time_col, values=count_col,
                              aggfunc='sum', fill_value=0)
                  .reindex(columns=days, fill_value=0))
        X = wide.to_numpy(float)
        DET = X > 0
        n_det = DET.sum(axis=1)
        keep = n_det >= min_detect
        if min_freq is not None:
            keep &= (X.sum(axis=1) / Dd.sum()) >= min_freq
        sel = np.flatnonzero(keep)
        if len(sel) == 0:
            continue
        t = days.astype(float)
        span = float(np.ptp(t))
        if span == 0:
            continue
        t_s = (t - t.mean()) / span

        N = None
        if likelihood == 'dropout':
            Np = dropout_N[dropout_N['patient'] == patient].set_index(time_col)['N_t']
            N = Np.reindex(days).to_numpy(float)
            if np.isnan(N).any():
                raise ValueError(f"dropout_N lacks samples for {patient}")
        tasks = [(X[i], DET[i], L, c, t_s, span, phi_p, beta_bound, likelihood, N)
                 for i in sel]
        if design == 'quadratic':
            out = _traj_run_tasks(tasks, n_jobs, fn=_traj_fit_quad)
            (beta, beta2, lr, pval, ok, atb, shape, k_ext, amp, supp) = map(np.array, zip(*out))
            ok, atb = ok.astype(bool), atb.astype(bool)
            fr = pd.DataFrame({
                'patient': patient, 'clono': wide.index[sel],
                'beta': beta.astype(float), 'beta2': beta2.astype(float),
                'stat': lr.astype(float), 'p_value': pval.astype(float),
                'n_detected': n_det[sel], 'n_timepoints': len(days),
                'phi_used': float(np.exp(np.mean(np.log(phi_p)))),
                'shape': np.where(~ok, 'failed', shape),
                'at_bound': atb,
                'extreme_day': days[k_ext.astype(int)],
                'amplitude_fold': np.exp(amp.astype(float)),
                'peak_support': supp.astype(float),
                'converged': ok, 'likelihood': likelihood, 'design': design})
            frames.append(fr)
            continue
        if design != 'slope':
            raise ValueError("design must be 'slope' or 'quadratic'")
        out = _traj_run_tasks(tasks, n_jobs)
        beta, se, lr, pval, ok, atb = map(np.array, zip(*out))

        direction = np.where(~ok, 'failed',
                    np.where(atb, 'separation',
                    np.where(beta > 0, 'expansion',
                    np.where(beta < 0, 'contraction', 'flat'))))
        frames.append(pd.DataFrame({
            'patient': patient, 'clono': wide.index[sel],
            'beta': beta, 'se_beta': se, 'stat': lr, 'p_value': pval,
            'n_detected': n_det[sel], 'n_timepoints': len(days),
            'phi_used': float(np.exp(np.mean(np.log(phi_p)))),   # geometric mean over samples
            'direction': direction, 'converged': ok,
            'likelihood': likelihood}))

    if not frames:
        return pd.DataFrame(columns=cols)
    if design == 'quadratic':
        return pd.concat(frames, ignore_index=True)
    return pd.concat(frames, ignore_index=True)[cols]


def nb_trajectory_combined(df: pd.DataFrame, phi, alpha: float = 0.05,
                           **test_kwargs) -> pd.DataFrame:
    """Slope and quadratic tests on the same clones, combined into one call per clone.

    Runs ``nb_trajectory_test`` with design='slope' and design='quadratic' (same
    arguments otherwise) and combines the two p-values per clone by Bonferroni min-p,
        p_comb = min(1, 2 * min(p_slope, p_quad)),
    which is valid whatever the dependence between the two tests. BH is then applied to
    p_comb within each patient (q_comb); called = q_comb < alpha. This gives a single
    FDR-controlled list of non-neutral clones that includes monotone trends (where the
    1-df slope test is more powerful) and rise-and-fall / fall-and-rise clones (which the
    slope test misses). The shape, extreme_day and amplitude come from the quadratic fit.

    Returns one row per tested clone: patient, clono, n_detected, beta_slope, p_slope,
    q_slope, direction (slope fit), p_quad, q_quad, shape, at_bound, extreme_day,
    amplitude_fold, peak_support, p_comb, q_comb, called, phi_used.
    """
    rs = nb_trajectory_test(df, phi, design='slope', **test_kwargs)
    rq = nb_trajectory_test(df, phi, design='quadratic', **test_kwargs)
    rs = add_qvalues(rs, p_col='p_value', by='patient')
    rq = add_qvalues(rq, p_col='p_value', by='patient')
    m = (rs[['patient', 'clono', 'n_detected', 'beta', 'p_value', 'q_value', 'direction',
             'phi_used']]
         .rename(columns={'beta': 'beta_slope', 'p_value': 'p_slope', 'q_value': 'q_slope'})
         .merge(rq[['patient', 'clono', 'p_value', 'q_value', 'shape', 'at_bound',
                    'extreme_day', 'amplitude_fold', 'peak_support']]
                .rename(columns={'p_value': 'p_quad', 'q_value': 'q_quad'}),
                on=['patient', 'clono'], how='outer'))
    m['p_comb'] = np.minimum(1.0, 2.0 * np.fmin(m['p_slope'], m['p_quad']))
    m = add_qvalues(m, p_col='p_comb', by='patient', out_col='q_comb')
    m['called'] = m['q_comb'] < alpha
    return m


def estimate_dropout_N(df: pd.DataFrame,
                       phi: Union[float, dict],
                       count_col: str = 'count',
                       lib_col: str = 'lib',
                       cutoff_col: str = 'c_t',
                       time_col: str = 'day',
                       min_other: int = 2,
                       n_iter: int = 15,
                       flag_rel: float = 10.0,
                       fallback: Optional[str] = 'median') -> pd.DataFrame:
    """Effective input N_t per sample, for the cell-sampling dropout term.

    For each sample t of a person: take the clones detected in >= min_other of the
    person's OTHER samples, estimate their lambda from those other samples only
    (NB MLE truncated at c, at fixed phi), and find the N_t maximising
        sum_clones log P(detected at t or not | lambda, N_t),
        P(detected) = (1 - exp(-N_t lambda)) * P_NB(X >= c_t | L_t lambda, phi).
    Selection and lambda never use sample t, so the estimate is free of selection bias.

    Estimates are flagged in two directions, relative to the median of the person's
    OTHER samples (factor flag_rel) or near the search bounds:
    * HIGH (implausibly little dropout, e.g. misses that do not follow exp(-N lambda)):
      unreliable; fallback='median' replaces it by the median of the person's unflagged
      samples (conservative). fallback=None keeps it.
    * LOW (the sample captured very few of the person's clones): the estimate is real,
      so it is KEPT and flagged in 'low_capture' with a warning; such a sample is
      usually a failed library and a candidate for exclusion in QC.
    The raw estimate is always kept in N_t_raw.

    Returns patient, time_col, N_t, N_t_raw, flagged, low_capture, n_clones, observed and predicted
    detection rates at t (a check of the fit), and N_t relative to the person's median.
    """
    rows = []
    for patient, g in df.groupby('patient'):
        days = np.sort(g[time_col].unique())
        phi_v = _phi_vector(phi, patient, days, time_col)
        per_day = g.groupby(time_col)[[lib_col, cutoff_col]].first().reindex(days)
        L = per_day[lib_col].to_numpy(float)
        c = per_day[cutoff_col].to_numpy(float)
        X = (g.pivot_table(index='clono', columns=time_col, values=count_col,
                           aggfunc='sum', fill_value=0)
               .reindex(columns=days, fill_value=0).to_numpy(float))
        DET = X > 0
        for ti in range(len(days)):
            oth = np.delete(np.arange(len(days)), ti)
            Xo, Do = X[:, oth], DET[:, oth]
            keep = Do.sum(axis=1) >= min_other
            if keep.sum() == 0:
                continue
            Xo, Do = Xo[keep], Do[keep]
            Lo, co = L[oth], c[oth]
            # truncated NB MLE of lambda from the other samples' detected counts
            lam = Xo.sum(1) / (Do * Lo).sum(1)
            phi_o = phi_v[oth]
            for _ in range(n_iter):
                mu = lam[:, None] * Lo[None, :]
                ex, var, _ = _trunc_moments(mu.ravel(),
                                            np.broadcast_to(phi_o, mu.shape).ravel(),
                                            np.broadcast_to(co, mu.shape).ravel().astype(int))
                ex, var = ex.reshape(mu.shape), var.reshape(mu.shape)
                w = phi_o / (phi_o + mu)
                U = (Do * w * (Xo - ex)).sum(1)
                I = (Do * w * w * var).sum(1)
                lam = lam * np.exp(np.clip(U / np.maximum(I, 1e-300), -2, 2))
            det_t = DET[keep, ti]
            mu_t = lam * L[ti]
            q_nb = stats.nbinom.sf(c[ti] - 1, phi_v[ti], phi_v[ti] / (phi_v[ti] + mu_t))

            def nll(logN):
                pdet = (1 - np.exp(-np.exp(logN) * lam)) * q_nb
                pdet = np.clip(pdet, 1e-12, 1 - 1e-12)
                return -(det_t * np.log(pdet) + (~det_t) * np.log1p(-pdet)).sum()

            # the objective is flat for very large N (no dropout), which can trap a
            # bounded search: locate the minimum on a grid first, then refine
            grid = np.linspace(np.log(1e2), np.log(1e10), 33)
            vals = np.array([nll(v) for v in grid])
            k = int(np.argmin(vals))
            lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
            r = minimize_scalar(nll, bounds=(lo, hi), method='bounded')
            N_t = float(np.exp(r.x))
            pred = ((1 - np.exp(-N_t * lam)) * q_nb).mean()
            rows.append({'patient': patient, time_col: days[ti], 'N_t': N_t,
                         'n_clones': int(keep.sum()),
                         'obs_detect_rate': float(det_t.mean()),
                         'pred_detect_rate': float(pred),
                         'pred_detect_rate_nb_only': float(q_nb.mean())})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    lo_b, hi_b = 1e2, 1e10
    out['N_t_raw'] = out['N_t']
    near_bound = (out['N_t'] <= lo_b * 1.5) | (out['N_t'] >= hi_b / 1.5)
    # relative to the median of the person's OTHER samples
    def _rel(exclude):
        r = []
        for p, t, v in zip(out['patient'], out[time_col], out['N_t']):
            ref = out.loc[(out['patient'] == p) & (out[time_col] != t) & ~exclude, 'N_t']
            r.append(v / ref.median() if len(ref) else 1.0)
        return np.asarray(r, float)
    # pass 1: HIGH flags (search bound, or far above the person's other samples)
    at_hi_bound = (out['N_t'] >= hi_b / 1.5).to_numpy()
    hi_mask = at_hi_bound | (_rel(at_hi_bound) > flag_rel)
    # pass 2: LOW flags, relative to the person's samples that are NOT high-flagged, so
    # a runaway estimate cannot make the other samples look like low capture
    rel_other = _rel(hi_mask)
    # HIGH: implausibly little dropout (fit ran towards 'no dropout') -> unreliable,
    #       replaced by the median of the person's other samples (conservative).
    # LOW:  the sample captured very few of the person's clones -> the estimate is
    #       real (a nearly empty library); kept as estimated, flagged for QC.
    hi_flag = pd.Series(hi_mask, index=out.index)
    lo_flag = ((out['N_t'] <= lo_b * 1.5) | (rel_other < 1.0 / flag_rel)) & ~hi_flag
    out['flagged'] = hi_flag | lo_flag
    out['low_capture'] = lo_flag
    if lo_flag.any():
        bad = out.loc[lo_flag, ['patient', time_col, 'N_t_raw', 'obs_detect_rate']]
        warnings.warn("very low capture (kept as estimated; consider excluding in QC): "
                      + ", ".join(f"{p} {time_col} {t} (N_t={v:.3g}, detect rate={r:.3f})"
                                  for p, t, v, r in bad.values))
    if hi_flag.any():
        bad = out.loc[hi_flag, ['patient', time_col, 'N_t_raw']]
        warnings.warn("unreliable dropout N_t (implausibly little dropout) for: "
                      + ", ".join(f"{p} {time_col} {t} (N_t={v:.3g})" for p, t, v in bad.values)
                      + ("; replaced by the median of the person's other samples"
                         if fallback == 'median' else ""))
        if fallback == 'median':
            for idx in out.index[hi_flag]:
                pat = out.at[idx, 'patient']
                ok = out[(out['patient'] == pat) & ~out['flagged']]['N_t']
                if len(ok):
                    out.at[idx, 'N_t'] = float(ok.median())
    out['N_t_rel'] = out['N_t'] / out.groupby('patient')['N_t'].transform('median')
    return out


def nb_trajectory_test_censored(df: pd.DataFrame,
                                phi: Union[float, dict],
                                count_col: str = 'count',
                                lib_col: str = 'lib',
                                cutoff_col: str = 'c_t',
                                time_col: str = 'day',
                                min_detect: int = 3,
                                beta_bound: float = 20.0,
                                n_jobs: int = 1) -> pd.DataFrame:
    """Plain censored test (kept for comparison; over-calls on real data, see the
    module notes). Same as ``nb_trajectory_test(..., likelihood='censored')`` with the
    old default slope bound."""
    return nb_trajectory_test(df, phi, likelihood='censored', count_col=count_col,
                              lib_col=lib_col, cutoff_col=cutoff_col, time_col=time_col,
                              min_detect=min_detect, beta_bound=beta_bound, n_jobs=n_jobs)


# =============================================================================
# Per-patient dispersion: per-clone intercept + slope, Cox-Reid (2x2), optional prior
# =============================================================================
#
#
# Model, per patient p and retained clone c (detected timepoints only):
#     x_ct ~ NB(mu_ct, phi_p),   log mu_ct = log L_t + a_c + b_c * s_t,
#     s_t = (t - mean of the patient's sampled days) / span   (as in nb_trajectory_test)
# Each clone's level a_c and trend b_c are nuisance parameters, re-solved at every
# trial phi; phi only measures scatter AROUND each clone's own trend, so monotone
# dynamics are not counted as noise. The Cox-Reid term is -0.5 * log det of the
# clone's 2x2 information for (a_c, b_c).


def _slope_blocks(df, count_col, lib_col, depth_col, time_col, min_detect,
                  min_freq_mult, cutoff_frac, c_col=None):
    """Detected rows of retained clones, clone-contiguous, with rescaled time s per row.

    Retained: >= min_detect detected timepoints and pooled frequency (relative to
    depth) >= min_freq_mult * cutoff_frac. Time is rescaled per patient over that
    patient's sampled days.
    """
    cols = list(dict.fromkeys([lib_col, depth_col] + ([c_col] if c_col else [])))
    agg = {count_col: 'sum', **{col: 'first' for col in cols}}
    days_all = (df[[ 'patient', time_col]].drop_duplicates()
                .groupby('patient')[time_col].agg(['min', 'max', 'mean']))
    d = (df[df[count_col] > 0]
         .groupby(['patient', 'clono', time_col], as_index=False, sort=True).agg(agg))
    d['n_det'] = d.groupby(['patient', 'clono'])[time_col].transform('size')
    d = d[d['n_det'] >= min_detect]
    g = d.groupby(['patient', 'clono'])
    p_depth = g[count_col].transform('sum') / g[depth_col].transform('sum')
    d = d[p_depth >= min_freq_mult * cutoff_frac].reset_index(drop=True)
    if len(d) == 0:
        return None
    span = (days_all['max'] - days_all['min']).reindex(d['patient']).to_numpy(float)
    mean = days_all['mean'].reindex(d['patient']).to_numpy(float)
    if np.any(span <= 0):
        raise ValueError("a patient has a single sampled day: slope not identifiable")
    s = (d[time_col].to_numpy(float) - mean) / span
    key = d['patient'].astype(str) + '|' + d['clono'].astype(str)
    starts = np.flatnonzero((key != key.shift()).values)
    n_per = np.diff(np.append(starts, len(d)))
    x = d[count_col].to_numpy(float)
    L = d[lib_col].to_numpy(float)
    lam0 = np.add.reduceat(x, starts) / np.add.reduceat(L, starts)
    c = d[c_col].to_numpy(int) if c_col else None
    return dict(x=x, L=L, s=s, starts=starts, n_per=n_per, lam0=lam0, c=c)


def _nb_ab_given_phi(x, L, s, starts, n_per, phi, lam0, c=None, b_bound=np.log(100),
                     max_iter=30, tol=1e-8, max_step=2.0):
    """Per-clone NB MLE of (a_c, b_c) at fixed phi, optionally truncated at c.

    Fisher scoring on (a, b), warm start a = log(pooled ratio), b = 0; b is kept in
    [-b_bound, b_bound]. Returns mu (per row), the 2x2 information per clone as
    (I00, I01, I11), and the largest final step.
    """
    a = np.log(lam0)
    b = np.zeros_like(a)
    step_max = np.inf

    def moments(mu):
        if c is None:
            return mu, mu * (phi + mu) / phi
        ex, var, _ = _trunc_moments(mu, phi, c)
        return ex, var

    def info(mu):
        w = phi / (phi + mu)
        _, var = moments(mu)
        h = w * w * var
        return (np.add.reduceat(h, starts), np.add.reduceat(h * s, starts),
                np.add.reduceat(h * s * s, starts))

    for _ in range(max_iter):
        mu = L * np.exp(np.repeat(a, n_per) + np.repeat(b, n_per) * s)
        w = phi / (phi + mu)
        ex, _ = moments(mu)
        g = w * (x - ex)
        U0 = np.add.reduceat(g, starts)
        U1 = np.add.reduceat(g * s, starts)
        I00, I01, I11 = info(mu)
        da_a = U0 / np.maximum(I00, 1e-300)               # a-only step (b held fixed)
        if b_bound == 0:                                  # intercept-only design
            da = da_a
            db = np.zeros_like(b)
        else:
            det = np.maximum(I00 * I11 - I01 * I01, 1e-300)
            da = (I11 * U0 - I01 * U1) / det
            db = (I00 * U1 - I01 * U0) / det
            # where the slope would leave the bound, hold it AT the bound and update
            # a conditionally on it (a joint step would push a in the wrong direction)
            out = np.abs(b + db) > b_bound
            if out.any():
                db = np.where(out, np.sign(b + db) * b_bound - b, db)
                da = np.where(out, da_a, da)
        da = np.clip(np.nan_to_num(da), -max_step, max_step)
        db = np.clip(np.nan_to_num(db), -max_step, max_step)
        a = a + da
        b = np.clip(b + db, -b_bound, b_bound)
        step_max = float(max(np.max(np.abs(da)), np.max(np.abs(db))))
        if step_max < tol:
            break
    mu = L * np.exp(np.repeat(a, n_per) + np.repeat(b, n_per) * s)
    return mu, info(mu), step_max


def _nb_neg_loglik_cr_slope(log_phi, blk, cox_reid=True, b_bound=np.log(100),
                            log_phi_prior=None, slope=True):
    """Negative CR-adjusted profile log-likelihood for the intercept + slope design
    (slope=False: intercept only, scalar CR term), plus an optional Gaussian prior on
    log(phi) given as (mean, sd)."""
    phi = np.exp(log_phi)
    c = blk['c']
    mu, (I00, I01, I11), _ = _nb_ab_given_phi(blk['x'], blk['L'], blk['s'], blk['starts'],
                                              blk['n_per'], phi, blk['lam0'], c,
                                              b_bound if slope else 0.0)
    p = phi / (phi + mu)
    with np.errstate(divide='ignore', invalid='ignore'):
        ll = stats.nbinom.logpmf(blk['x'], phi, p)
        if c is not None:
            ll = ll - stats.nbinom.logsf(c - 1, phi, p)
    obj = ll.sum()
    if cox_reid:
        det = I00 * I11 - I01 * I01 if slope else I00
        obj -= 0.5 * np.log(np.maximum(det, 1e-300)).sum()
    if log_phi_prior is not None:
        m, sd = log_phi_prior
        obj -= 0.5 * ((log_phi - m) / sd) ** 2
    return -obj if np.isfinite(obj) else 1e300


def fit_nb_dispersion_slope(df: pd.DataFrame,
                            count_col: str = 'count',
                            lib_col: str = 'lib',
                            depth_col: str = 'D_t',
                            time_col: str = 'day',
                            min_detect: int = 3,
                            min_freq_mult: float = 10.0,
                            cutoff_frac: float = CUTOFF_FRAC,
                            truncate: bool = False,
                            c_col: str = 'c_t',
                            cox_reid: bool = True,
                            b_bound: float = np.log(100),
                            log_phi_prior: Optional[tuple] = None,
                            slope: bool = True,
                            ci_level: float = 0.95) -> dict:
    """NB dispersion with a per-clone intercept AND slope (Cox-Reid adjusted).

    Intended per patient (pass one patient's rows), for the trajectory test's phi.

    min_detect    : clones need >= 3 detected timepoints; with 2 the clone is fitted
                    exactly and carries no information about phi.
    min_freq_mult : estimation-only floor (clones well above the detection threshold,
                    where the estimator is reliable). Every clone is still TESTED.
    truncate      : model detected counts as truncated at c (negligible above a 10x floor).
    log_phi_prior : optional (mean, sd) of a Gaussian prior on log(phi), e.g. from donors;
                    a guardrail for patients with little data. None = no prior.
    slope         : True = per-clone intercept + slope (default); False = intercept only
                    (same clones), for comparison.

    Returns dict: phi, ci (profile, on phi), at_bound, n_clones, n_obs, resid_df
    (sum of n_det - 2), and the settings.
    """
    _require(df, ['patient', 'clono', time_col, count_col, lib_col, depth_col]
             + ([c_col] if truncate else []))
    blk = _slope_blocks(df, count_col, lib_col, depth_col, time_col, min_detect,
                        min_freq_mult, cutoff_frac, c_col if truncate else None)
    if blk is None:
        raise ValueError("no clones pass min_detect and the frequency floor")
    args = (blk, cox_reid, b_bound, log_phi_prior, slope)
    res = minimize_scalar(_nb_neg_loglik_cr_slope, args=args,
                          bounds=PHI_BOUNDS, method='bounded')
    lp = float(res.x)
    at_bound = (np.isclose(lp, PHI_BOUNDS[0], atol=1e-3)
                or np.isclose(lp, PHI_BOUNDS[1], atol=1e-3))
    if at_bound:
        warnings.warn("phi hit the search bound: not identifiable in this subset")

    crit = stats.chi2.ppf(ci_level, 1) / 2
    f0 = res.fun
    g = lambda v: _nb_neg_loglik_cr_slope(v, *args) - f0 - crit
    ci = []
    for edge in (max(lp - 3, PHI_BOUNDS[0]), min(lp + 3, PHI_BOUNDS[1])):
        try:
            ci.append(float(np.exp(brentq(g, *sorted((edge, lp)), xtol=1e-3))))
        except ValueError:
            ci.append(np.nan)

    return {'phi': float(np.exp(lp)), 'ci': tuple(ci), 'at_bound': bool(at_bound),
            'n_clones': int(len(blk['n_per'])), 'n_obs': int(len(blk['x'])),
            'resid_df': int((blk['n_per'] - (2 if slope else 1)).sum()),
            'settings': dict(min_detect=min_detect, min_freq_mult=min_freq_mult,
                             truncate=truncate, cox_reid=cox_reid, b_bound=b_bound,
                             log_phi_prior=log_phi_prior, slope=slope)}


def fit_phi_per_patient(df: pd.DataFrame,
                        reference_phis: Optional[dict] = None,
                        log_phi_prior: Optional[tuple] = None,
                        flag_factor: float = 1.5,
                        **kwargs) -> pd.DataFrame:
    """Per-patient phi (intercept + slope design) for every patient in df.

    reference_phis : optional {name: phi} of reference individuals (e.g. donors); a
                     patient is flagged if its phi lies outside
                     [min / flag_factor, max * flag_factor] of those values.
    log_phi_prior  : optional guardrail prior, passed to fit_nb_dispersion_slope.
    kwargs         : passed to fit_nb_dispersion_slope (min_freq_mult, min_detect, ...).

    Also reports the intercept-only phi (same clones, same floor): the difference
    estimates how much monotone dynamics would have been counted as noise.

    Returns one row per patient; use dict(zip(out.patient, out.phi)) as the
    trajectory test's per-patient phi.
    """
    rows = []
    for pat, sub in df.groupby('patient'):
        try:
            r = fit_nb_dispersion_slope(sub, log_phi_prior=log_phi_prior, **kwargs)
            r0 = fit_nb_dispersion_slope(sub, log_phi_prior=log_phi_prior,
                                         slope=False, **kwargs)         # intercept only
        except ValueError as e:
            rows.append({'patient': pat, 'error': str(e)})
            continue
        rows.append({'patient': pat, 'phi': r['phi'], 'ci_lo': r['ci'][0], 'ci_hi': r['ci'][1],
                     'phi_intercept_only': r0['phi'],
                     'n_clones': r['n_clones'], 'resid_df': r['resid_df'],
                     'at_bound': r['at_bound']})
    out = pd.DataFrame(rows)
    if reference_phis and 'phi' in out:
        lo = min(reference_phis.values()) / flag_factor
        hi = max(reference_phis.values()) * flag_factor
        out['outside_reference'] = (out['phi'] < lo) | (out['phi'] > hi)
    return out


# =============================================================================
# Calibration: leave-one-donor-out null test
# =============================================================================

def leave_one_donor_out(healthy: pd.DataFrame,
                        likelihood: str = 'dropout',
                        reference: str = 'min_others',
                        phi_kwargs: Optional[dict] = None,
                        test_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """Test each donor as if it were a patient, with phi from the OTHER donors.

    Healthy donors are the declared null, so p-values should be close to uniform
    (or conservative).

    likelihood : passed to nb_trajectory_test ('dropout', 'censored' or 'truncated');
                 for 'dropout', N_t is estimated within the held-out donor.
    reference  : 'min_others' - the minimum of the other donors' own phi (intercept
                                only), i.e. the conservative external null used for
                                patients;
                 'pooled'     - fit_nb_dispersion_cr on the other donors pooled.
    phi_kwargs : passed to the phi fit (default min_freq_mult=10; for 'pooled' also
                 truncate=False).
    test_kwargs: passed to nb_trajectory_test.
    """
    phi_kwargs = {'min_freq_mult': 10, **(phi_kwargs or {})}
    test_kwargs = test_kwargs or {}
    if reference == 'min_others':
        own = fit_phi_per_patient(healthy, **phi_kwargs)
        own = dict(zip(own['patient'], own['phi_intercept_only']))
    out = []
    for held in healthy['patient'].unique():
        if reference == 'min_others':
            phi_ref = min(v for p, v in own.items() if p != held)
        elif reference == 'pooled':
            kw = {'truncate': False, **phi_kwargs}
            phi_ref = fit_nb_dispersion_cr(healthy[healthy['patient'] != held], **kw)['phi']
        else:
            raise ValueError("reference must be 'min_others' or 'pooled'")
        test_df = healthy[healthy['patient'] == held]
        kw = dict(test_kwargs)
        if likelihood == 'dropout':
            kw['dropout_N'] = estimate_dropout_N(
                test_df, phi=phi_ref,
                **{k: v for k, v in test_kwargs.items()
                   if k in ('count_col', 'lib_col', 'cutoff_col', 'time_col')})
        res = nb_trajectory_test(test_df, phi_ref, likelihood=likelihood, **kw)
        out.append(res.assign(held_out=held, phi_ref=phi_ref))
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



# =============================================================================
# Power: spike-in of known trends on each patient's real design
# =============================================================================

def simulate_like_patient(df: pd.DataFrame,
                          phi: Union[float, dict],
                          dropout_N: pd.DataFrame,
                          frac_spike: float = 0.05,
                          fold_changes=(2, 4, 10, 30),
                          max_clones: Optional[int] = None,
                          seed: int = 0,
                          shape: str = 'trend',
                          count_col: str = 'count',
                          lib_col: str = 'lib',
                          depth_col: str = 'D_t',
                          cutoff_col: str = 'c_t',
                          time_col: str = 'day'):
    """Simulate counts on each person's REAL design, with known trends in a subset.

    Per person: the real sampled days, lib_t, D_t, c_t and N_t (from ``dropout_N``),
    and the person's phi. Each real clone (>= 1 detection) gets its abundance
    lambda_c = pooled ratio over its detected rows. A random fraction ``frac_spike``
    gets a log-linear trend over the series: fold change F (drawn from
    ``fold_changes``), up or down at random, centred on mid-series; the rest are
    constant. Then, per clone and timepoint:
        present ~ Bernoulli(1 - exp(-N_t * lambda_ct))      (cell sampling)
        x ~ NB(lib_t * lambda_ct, phi)                      (reads)
        observed if present and x >= c_t                    (detection threshold)

    Returns (sim, truth): sim in the usual long format (detected rows only, with the
    real per-sample columns); truth has one row per simulated clone with fold,
    sign, log_fold_true (over the series, or peak / baseline for shape='peak'),
    peak_day_true (shape='peak') and log10_lambda.

    shape='peak' : spiked clones instead rise and fall back: a Gaussian bump (width =
    span / 4) of height F centred on a random interior sampled day.
    """
    rng = np.random.default_rng(seed)
    fold_changes = np.asarray(fold_changes, float)
    sims, truths = [], []
    for pat, g in df.groupby('patient'):
        days = np.sort(g[time_col].unique())
        phi_p = _phi_vector(phi, pat, days, time_col)[None, :]
        per_day = (g.groupby(time_col)[[lib_col, depth_col, cutoff_col]].first()
                    .reindex(days))
        L = per_day[lib_col].to_numpy(float)
        Dd = per_day[depth_col].to_numpy(float)
        c = per_day[cutoff_col].to_numpy(float)
        N = (dropout_N[dropout_N['patient'] == pat].set_index(time_col)['N_t']
             .reindex(days).to_numpy(float))
        if np.isnan(N).any():
            raise ValueError(f"dropout_N lacks samples for {pat}")
        det = g[g[count_col] > 0]
        lam = (det.groupby('clono')[count_col].sum()
               / det.groupby('clono')[lib_col].sum()).to_numpy(float)
        if max_clones is not None and len(lam) > max_clones:
            lam = rng.choice(lam, size=max_clones, replace=False)
        n = len(lam)
        s = (days - days.mean()) / np.ptp(days)
        spiked = rng.random(n) < frac_spike
        fold = np.where(spiked, rng.choice(fold_changes, size=n), 1.0)
        sign = np.where(spiked, rng.choice([-1, 1], size=n), 0)
        b = sign * np.log(fold)                                # log fold over the series
        t_pk = np.full(n, np.nan)
        if shape == 'trend':
            prof_t = b[:, None] * s[None, :]
        elif shape == 'peak':
            # transient: rise to peak and fall back; peak at a random interior sampled day,
            # Gaussian bump with width = span / 4; 'fold' = peak / baseline
            inner = days[1:-1] if len(days) > 2 else days
            t_pk = np.where(spiked, rng.choice(inner, size=n), np.nan)
            width = np.ptp(days) / 4
            bump = np.nan_to_num(np.exp(-0.5 * ((days[None, :] - t_pk[:, None]) / width) ** 2))
            prof_t = np.abs(b)[:, None] * bump
            sign = np.where(spiked, 1, 0)
            b = np.abs(b)
        else:
            raise ValueError("shape must be 'trend' or 'peak'")
        lam_t = lam[:, None] * np.exp(prof_t)
        mu = lam_t * L[None, :]
        present = rng.random(mu.shape) >= np.exp(-N[None, :] * lam_t)
        x = rng.negative_binomial(phi_p, phi_p / (phi_p + mu)).astype(float)
        obs = present & (x >= c[None, :])
        ci, ti = np.nonzero(obs)
        sims.append(pd.DataFrame({'patient': pat, 'clono': ci, time_col: days[ti],
                                  count_col: x[ci, ti], lib_col: L[ti],
                                  depth_col: Dd[ti], cutoff_col: c[ti]}))
        truths.append(pd.DataFrame({'patient': pat, 'clono': np.arange(n),
                                    'spiked': spiked, 'fold': fold, 'sign': sign,
                                    'log_fold_true': b, 'peak_day_true': t_pk,
                                    'log10_lambda': np.log10(lam)}))
    return pd.concat(sims, ignore_index=True), pd.concat(truths, ignore_index=True)


def spike_in_power(df: pd.DataFrame,
                   phi: Union[float, dict],
                   frac_spike: float = 0.05,
                   fold_changes=(2, 4, 10, 30),
                   abundance_edges=(-np.inf, -5.0, -4.0, np.inf),
                   alpha: float = 0.05,
                   max_clones: Optional[int] = None,
                   seed: int = 0,
                   n_jobs: int = 1,
                   cutoff_col: str = 'c_t',
                   time_col: str = 'day',
                   test_kwargs: Optional[dict] = None,
                   shape: str = 'trend') -> dict:
    """Power of the dropout trajectory test, by fold change and abundance, per person.

    For each person: estimate N_t on the real data, simulate with known trends on the
    real design (``simulate_like_patient``), then run the SAME pipeline on the
    simulated data (N_t re-estimated, dropout test, BH within person).

    abundance_edges : log10(lambda) band edges for the power table (lambda on the
                      offset scale, about the clone's frequency).
    test_kwargs     : extra arguments for nb_trajectory_test (e.g. {'min_freq': 1e-5}),
                      so the simulated data go through exactly the real pipeline.

    Returns dict with
      'power'   : per person x fold x abundance band: n spiked, fraction tested
                  (>= min_detect detections), power among ALL spiked clones and among
                  TESTED ones (q < alpha); among calls: shape_correct (slope design:
                  right sign; quadratic: classified 'up'/'down' for trends, 'peak' for
                  peaks), median estimated / true log fold (quadratic: fitted max/min
                  over the series), and for peaks peak_day_correct (extreme_day equals
                  the simulated peak day);
      'fdr'     : per person: calls, false calls (unspiked clones), observed FDR;
      'realism' : real vs simulated number of testable clones and detection rate;
      'per_clone': truth merged with test results.
    """
    power_rows, fdr_rows, real_rows, per_clone = [], [], [], []
    for pat, sub in df.groupby('patient'):
        phi_p = phi
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Nt_real = estimate_dropout_N(sub, phi=phi_p, cutoff_col=cutoff_col,
                                         time_col=time_col)
            sim, truth = simulate_like_patient(sub, phi_p, Nt_real, frac_spike,
                                               fold_changes, max_clones, seed,
                                               cutoff_col=cutoff_col, time_col=time_col,
                                               shape=shape)
            Nt_sim = estimate_dropout_N(sim, phi=phi_p, cutoff_col=cutoff_col,
                                        time_col=time_col)
            res = nb_trajectory_test(sim, phi_p, likelihood='dropout', dropout_N=Nt_sim,
                                     cutoff_col=cutoff_col, time_col=time_col,
                                     n_jobs=n_jobs, **(test_kwargs or {}))
        res = add_qvalues(res, p_col='p_value', by='patient')
        span = float(np.ptp(np.sort(sub[time_col].unique())))
        keepc = ['patient', 'clono', 'beta', 'p_value', 'q_value'] + \
                [k for k in ('direction', 'shape', 'extreme_day', 'amplitude_fold')
                 if k in res.columns]
        m = truth.merge(res[keepc], on=['patient', 'clono'], how='left')
        m['tested'] = m['p_value'].notna()
        m['called'] = m['q_value'] < alpha
        quad = 'shape' in m.columns
        if quad:                                   # fitted max / min over the series
            m['log_fold_est'] = np.log(m['amplitude_fold'])
            exp_shape = (np.where(m['sign'] > 0, 'up', 'down') if shape == 'trend'
                         else np.full(len(m), 'peak'))
            m['shape_correct'] = m['shape'] == exp_shape
            m['est_over_true'] = m['log_fold_est'] / m['log_fold_true'].abs()
        else:
            m['log_fold_est'] = m['beta'] * span
            m['shape_correct'] = (np.sign(m['beta']) == m['sign']) if shape == 'trend' \
                else np.nan
            m['est_over_true'] = (m['log_fold_est'] / m['log_fold_true']) if shape == 'trend' \
                else np.nan
        m['band'] = pd.cut(m['log10_lambda'], list(abundance_edges))
        per_clone.append(m)

        sp = m[m['spiked']]
        for (fold, band), gg in sp.groupby(['fold', 'band'], observed=True):
            calls = gg[gg['called']]
            power_rows.append({
                'patient': pat, 'fold': fold, 'band': str(band), 'n_spiked': len(gg),
                'frac_tested': gg['tested'].mean(),
                'power_all': gg['called'].mean(),
                'power_tested': gg.loc[gg['tested'], 'called'].mean() if gg['tested'].any() else np.nan,
                'shape_correct': calls['shape_correct'].astype(float).mean()
                                 if len(calls) else np.nan,
                'est_over_true_logfold': calls['est_over_true'].median()
                                         if len(calls) else np.nan,
                'peak_day_correct': (calls['extreme_day'] == calls['peak_day_true']).mean()
                                    if len(calls) and quad and shape == 'peak' else np.nan})
        calls = m[m['called']]
        fdr_rows.append({'patient': pat, 'n_calls': len(calls),
                         'false_calls': int((~calls['spiked']).sum()),
                         'observed_FDR': (~calls['spiked']).mean() if len(calls) else np.nan})

        def testable(d):
            n_det = d[d['count'] > 0].groupby('clono')[time_col].nunique()
            return int((n_det >= 3).sum())
        real_rows.append({'patient': pat,
                          'real_testable_clones': testable(sub),
                          'sim_testable_clones': testable(sim),
                          'real_mean_detect_rate': float(Nt_real['obs_detect_rate'].mean()),
                          'sim_mean_detect_rate': float(Nt_sim['obs_detect_rate'].mean())})

    return {'power': pd.DataFrame(power_rows), 'fdr': pd.DataFrame(fdr_rows),
            'realism': pd.DataFrame(real_rows),
            'per_clone': pd.concat(per_clone, ignore_index=True)}


def plot_power(power: pd.DataFrame, path: Optional[str] = None, which: str = 'power_all'):
    """Power against fold change, one panel per person, one line per abundance band."""
    import matplotlib.pyplot as plt
    pats = sorted(power['patient'].unique())
    ncol = min(3, len(pats))
    nrow = int(np.ceil(len(pats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             sharey=True, squeeze=False)
    for ax, pat in zip(axes.ravel(), pats):
        g = power[power['patient'] == pat]
        for band, gb in g.groupby('band', sort=False):
            gb = gb.sort_values('fold')
            ax.plot(gb['fold'], gb[which], marker='o', label=band)
        ax.set_xscale('log')
        ax.set_xticks(sorted(power['fold'].unique()))
        ax.set_xticklabels([f"{f:g}x" for f in sorted(power['fold'].unique())])
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.8, color='grey', lw=0.6, ls='--')
        ax.set_title(pat, fontsize=10)
        ax.set_xlabel('fold change over the series')
    for ax in axes[:, 0]:
        ax.set_ylabel('power' if which == 'power_all' else which)
    for ax in axes.ravel()[len(pats):]:
        ax.axis('off')
    axes.ravel()[0].legend(title='log10 frequency', fontsize=7, title_fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


# =============================================================================
# Empirical-Bayes shrinkage of clone slopes (spike-and-slab on the log fold change)
# =============================================================================
#
# For each clone c the slope enters as b_c = beta_c * span (log fold change over the
# person's series). Stage 1: the profile log-likelihood l_c(b) = max_alpha l_c(alpha, b)
# is computed on a grid of b, with the SAME observation model as the trajectory test
# (NB + optional dropout + threshold c_t; phi and N_t fixed). Stage 2: across the
# person's clones,
#     g(b) = (1 - w) * delta_0(b) + w * Slab(b; tau)       (Laplace or normal slab)
# with w (fraction of non-neutral clones) and tau (typical |log fold|) fitted by
# maximising sum_c log[(1 - w) L_c(0) + w * int L_c(b) Slab(b) db]. Stage 3: per-clone
# posterior: lfdr = P(b = 0 | data), shrunken log fold change and credible interval.


def _slope_profile_matrix(X, DET, L, c, s, phi, likelihood, N, b_grid,
                          max_iter=40, tol=1e-7, h=1e-3, cox_reid=True):
    """Profile log-likelihood over alpha, for every clone (rows of X) at every b.

    Vectorised Newton in alpha (finite-difference derivatives), warm-started along the
    grid outwards from b = 0. With cox_reid=True the profile is adjusted by
    -0.5 * log(observed information for alpha) at each b (approximately integrating
    alpha out): the plain profile is over-confident with 3-6 timepoints per clone, which
    makes neutral data look like a narrow slab of small changes.
    Returns l (n_clones x n_grid) and alpha_hat at b = 0.
    """
    if likelihood == 'dropout':
        Nr, Lr = N[None, :], L[None, :]
        ll_fn = lambda mu: _traj_dropout_ll(mu, X, DET, c[None, :], phi, Nr, Lr)
    else:
        f0 = _TRAJ_LL[likelihood]
        ll_fn = lambda mu: f0(mu, X, DET, c[None, :], phi)

    def total(a, b):
        mu = L[None, :] * np.exp(a[:, None] + b * s[None, :])
        v = ll_fn(mu).sum(axis=1)
        return np.where(np.isfinite(v), v, -1e300)

    n = X.shape[0]
    a_start = np.log(np.maximum((X * DET).sum(1), 1.0) / (DET * L[None, :]).sum(1))
    lo_a, hi_a = a_start - 30.0, a_start + 30.0

    def newton(a, b):
        fa = total(a, b)
        for _ in range(max_iter):
            fp, fm = total(a + h, b), total(a - h, b)
            g = (fp - fm) / (2 * h)
            H = (fp - 2 * fa + fm) / (h * h)
            step = np.where(H < -1e-10, -g / np.where(H < -1e-10, H, -1.0), np.sign(g) * 0.5)
            step = np.clip(step, -2.0, 2.0)
            for _half in range(6):                       # vectorised backtracking
                a_new = np.clip(a + step, lo_a, hi_a)
                f_new = total(a_new, b)
                bad = f_new < fa - 1e-12
                if not bad.any():
                    break
                step = np.where(bad, step / 2, step)
            a_new = np.where(f_new >= fa - 1e-12, a_new, a)
            f_new = np.maximum(f_new, fa)
            done = np.max(np.abs(a_new - a)) < tol
            a, fa = a_new, f_new
            if done:
                break
        if cox_reid:
            fp, fm = total(a + h, b), total(a - h, b)
            info = -(fp - 2 * fa + fm) / (h * h)
            fa = fa - 0.5 * np.log(np.maximum(info, 1e-8))
        return a, fa

    K = len(b_grid)
    k0 = int(np.argmin(np.abs(b_grid)))
    l = np.empty((n, K))
    a0, l[:, k0] = newton(a_start, b_grid[k0])
    a = a0
    for k in range(k0 + 1, K):
        a, l[:, k] = newton(a, b_grid[k])
    a = a0
    for k in range(k0 - 1, -1, -1):
        a, l[:, k] = newton(a, b_grid[k])
    return l, a0


def slope_profiles(df: pd.DataFrame,
                   phi: Union[float, dict],
                   likelihood: str = 'dropout',
                   dropout_N: Optional[pd.DataFrame] = None,
                   b_max: float = np.log(1000),
                   n_grid: int = 81,
                   min_detect: int = 3,
                   min_freq: Optional[float] = None,
                   cox_reid: bool = True,
                   count_col: str = 'count',
                   lib_col: str = 'lib',
                   cutoff_col: str = 'c_t',
                   depth_col: str = 'D_t',
                   time_col: str = 'day') -> dict:
    """Stage 1: per-clone (Cox-Reid adjusted) profile log-likelihood of b on a grid.

    Clone selection and observation model are those of ``nb_trajectory_test`` (same
    min_detect, min_freq, likelihood, phi, dropout_N). b = beta * span is the log fold
    change over the person's sampled series; the grid is symmetric in [-b_max, b_max]
    with an odd number of points, so b = 0 is on it.

    Returns {patient: dict(clono, b_grid, loglik (n x K), span, phi, n_detected)}.
    """
    if likelihood not in _TRAJ_LL:
        raise ValueError("likelihood must be 'censored', 'truncated' or 'dropout'")
    if likelihood == 'dropout' and dropout_N is None:
        raise ValueError("likelihood='dropout' needs dropout_N (see estimate_dropout_N)")
    n_grid = n_grid + (1 - n_grid % 2)
    b_grid = np.linspace(-b_max, b_max, n_grid)
    out = {}
    for patient, g in df.groupby('patient'):
        days = np.sort(g[time_col].unique())
        phi_p = _phi_vector(phi, patient, days, time_col)
        span = float(np.ptp(days.astype(float)))
        if span == 0:
            continue
        dcols = [lib_col, cutoff_col] + ([depth_col] if depth_col in g.columns else [])
        per_day = g.groupby(time_col)[dcols].first().reindex(days)
        L = per_day[lib_col].to_numpy(float)
        c = per_day[cutoff_col].to_numpy(float)
        Dd = per_day[depth_col].to_numpy(float) if depth_col in g.columns else L
        wide = (g.pivot_table(index='clono', columns=time_col, values=count_col,
                              aggfunc='sum', fill_value=0)
                  .reindex(columns=days, fill_value=0))
        X = wide.to_numpy(float)
        DET = X > 0
        keep = DET.sum(1) >= min_detect
        if min_freq is not None:
            keep &= (X.sum(1) / Dd.sum()) >= min_freq
        if not keep.any():
            continue
        N = None
        if likelihood == 'dropout':
            N = (dropout_N[dropout_N['patient'] == patient].set_index(time_col)['N_t']
                 .reindex(days).to_numpy(float))
            if np.isnan(N).any():
                raise ValueError(f"dropout_N lacks samples for {patient}")
        s = (days.astype(float) - days.mean()) / span
        with np.errstate(all='ignore'):
            l, _ = _slope_profile_matrix(X[keep], DET[keep], L, c, s, phi_p,
                                         likelihood, N, b_grid, cox_reid=cox_reid)
        out[patient] = dict(clono=wide.index[keep].to_numpy(), b_grid=b_grid, loglik=l,
                            span=span, phi=phi_p, n_detected=DET[keep].sum(1))
    return out


def _slab_weights(b_grid, tau, slab):
    """Slab density on the grid (excluding b = 0), normalised to sum to 1 over the grid."""
    if slab == 'laplace':
        d = np.exp(-np.abs(b_grid) / tau)
    elif slab == 'normal':
        d = np.exp(-0.5 * (b_grid / tau) ** 2)
    else:
        raise ValueError("slab must be 'laplace' or 'normal'")
    d = np.where(np.isclose(b_grid, 0.0), 0.0, d)
    return d / d.sum()


def fit_slope_prior(prof: dict, slab: str = 'laplace', hyper: Optional[tuple] = None,
                    tau_min: float = np.log(2)):
    """Stage 2: fit (w, tau) by marginal likelihood over one person's clones.

    prof    : one person's entry from ``slope_profiles``.
    hyper   : optional fixed (w, tau) (e.g. donor-derived); then nothing is fitted.
    tau_min : smallest slab scale (log fold over the series). Keeps 'non-neutral' meaning
              a real change: without it a very narrow slab can absorb small residual
              misfit of the noise model and inflate w with no biological meaning.
    Returns (w, tau, log marginal likelihood).
    """
    l, b = prof['loglik'], prof['b_grid']
    k0 = int(np.argmin(np.abs(b)))
    m = l.max(axis=1, keepdims=True)
    R = np.exp(l - m)                                    # relative likelihood, <= 1
    R0 = R[:, k0]

    def nll(theta):
        w = 1.0 / (1.0 + np.exp(-theta[0]))
        sw = _slab_weights(b, np.exp(theta[1]), slab)
        mix = (1 - w) * R0 + w * (R @ sw)
        return -(np.log(np.maximum(mix, 1e-300)) + m[:, 0]).sum()

    if hyper is not None:
        w, tau = hyper
        return float(w), float(tau), -nll([np.log(w / (1 - w)), np.log(tau)])
    best = None
    for w0 in (0.01, 0.1, 0.3):
        for t0 in (0.5, 1.5, 3.0):
            r = minimize(nll, x0=[np.log(w0 / (1 - w0)), np.log(t0)], method='L-BFGS-B',
                         bounds=[(-12.0, 6.0), (np.log(tau_min), np.log(prof['b_grid'].max()))])
            if best is None or r.fun < best.fun:
                best = r
    w = 1.0 / (1.0 + np.exp(-best.x[0]))
    return float(w), float(np.exp(best.x[1])), float(-best.fun)


def slope_posteriors(prof: dict, w: float, tau: float, slab: str = 'laplace',
                     level: float = 0.95) -> pd.DataFrame:
    """Stage 3: per-clone posterior under g(b) = (1 - w) delta_0 + w Slab(tau).

    Columns (b = log fold change over the series; fold = exp(b)):
      lfdr            P(b = 0 | data)
      p_up            P(b > 0 | data)
      b_mle           grid maximiser of the profile likelihood (for comparison)
      b_mle_at_edge   MLE at the grid edge (separation-type clone)
      b_post_mean     posterior mean of b (mixture, includes the spike at 0)
      b_slab_median, b_slab_lo, b_slab_hi
                      median and credible interval of b GIVEN the clone is non-neutral
    """
    l, b = prof['loglik'], prof['b_grid']
    k0 = int(np.argmin(np.abs(b)))
    m = l.max(axis=1, keepdims=True)
    R = np.exp(l - m)
    sw = _slab_weights(b, tau, slab)
    slab_post = R * sw[None, :] * w                      # unnormalised, per grid point
    spike = (1 - w) * R[:, k0]
    tot = spike + slab_post.sum(1)
    lfdr = spike / tot
    p_up = slab_post[:, b > 0].sum(1) / tot
    b_post_mean = (slab_post @ b) / tot
    cs = np.cumsum(slab_post, axis=1) / np.maximum(slab_post.sum(1, keepdims=True), 1e-300)
    q = lambda p: b[np.minimum((cs < p).sum(1), len(b) - 1)]
    a = (1 - level) / 2
    k_mle = l.argmax(1)
    return pd.DataFrame({
        'clono': prof['clono'], 'n_detected': prof['n_detected'],
        'lfdr': lfdr, 'p_up': p_up,
        'b_mle': b[k_mle], 'b_mle_at_edge': (k_mle == 0) | (k_mle == len(b) - 1),
        'b_post_mean': b_post_mean,
        'b_slab_median': q(0.5), 'b_slab_lo': q(a), 'b_slab_hi': q(1 - a),
    })


def lfdr_qvalues(lfdr) -> np.ndarray:
    """Bayesian FDR q-values: for each clone, the mean lfdr of all clones at least as
    significant. Calling q < alpha keeps the expected FDR of the called set <= alpha."""
    lfdr = np.asarray(lfdr, float)
    order = np.argsort(lfdr)
    q = np.empty_like(lfdr)
    q[order] = np.cumsum(lfdr[order]) / np.arange(1, len(lfdr) + 1)
    return q


def eb_trajectory(df: pd.DataFrame,
                  phi: Union[float, dict],
                  likelihood: str = 'dropout',
                  dropout_N: Optional[pd.DataFrame] = None,
                  slab: str = 'laplace',
                  hyper: Optional[dict] = None,
                  alpha: float = 0.05,
                  tau_min: float = np.log(2),
                  w_max: float = 0.9,
                  **profile_kwargs):
    """Empirical-Bayes trajectory analysis, per person.

    Same clone selection and observation model as ``nb_trajectory_test``; instead of a
    per-clone LRT, fits a spike-and-slab prior on the log fold change per person and
    reports posteriors.

    hyper : optional {patient: (w, tau)} to use fixed hyperparameters (e.g. the donors'
            values) instead of fitting them per person.
    profile_kwargs : passed to ``slope_profiles`` (b_max, n_grid, min_detect, min_freq,
            column names).

    Returns (clones, summary):
      clones  : one row per clone: patient, clono, lfdr, q_lfdr, called (q_lfdr < alpha),
                p_up, b_* (log fold over the series), fold_post_mean = exp(b_post_mean),
                fold_slab_lo / fold_slab_hi.
      summary : per person: w (slab weight), tau, frac_gt_2fold / frac_gt_4fold (estimated
                fraction of clones changing > 2x / > 4x over the series; more robust than
                w), expected number of non-neutral clones sum(1 - lfdr), calls, n_tested,
                log marginal likelihood.
    """
    profs = slope_profiles(df, phi, likelihood=likelihood, dropout_N=dropout_N,
                           **profile_kwargs)
    rows, summ = [], []
    for pat, prof in profs.items():
        hp = hyper.get(pat) if hyper else None
        w, tau, lml = fit_slope_prior(prof, slab=slab, hyper=hp, tau_min=tau_min)
        post = slope_posteriors(prof, w, tau, slab=slab)
        post['q_lfdr'] = lfdr_qvalues(post['lfdr'])
        degenerate = w > w_max
        if degenerate:
            warnings.warn(f"{pat}: slab weight w={w:.3f} > {w_max}: almost no clone fits the "
                          "neutral spike, so lfdr-based calls are not meaningful and are "
                          "suppressed. Usually means the neutral model (constant frequency) "
                          "does not describe this person's background, e.g. drift over a long "
                          "span or an unmodelled sample effect; check before interpreting.")
        post['called'] = (post['q_lfdr'] < alpha) & (not degenerate)
        post.insert(0, 'patient', pat)
        post['span_days'] = prof['span']
        rows.append(post)
        sw = _slab_weights(prof['b_grid'], tau, slab)
        absb = np.abs(prof['b_grid'])
        summ.append({'patient': pat, 'w': w, 'tau': tau,
                     'typical_fold': float(np.exp(tau)),
                     # robust summaries: fraction of clones changing by more than 2x / 4x
                     # over the series (w alone is weakly identified when the slab has
                     # mass near 0)
                     'frac_gt_2fold': float(w * sw[absb > np.log(2)].sum()),
                     'frac_gt_4fold': float(w * sw[absb > np.log(4)].sum()),
                     'expected_non_neutral': float((1 - post['lfdr']).sum()),
                     'n_called': int(post['called'].sum()),
                     'n_called_up': int((post['called'] & (post['p_up'] > 0.5)).sum()),
                     'n_called_down': int((post['called'] & (post['p_up'] <= 0.5)).sum()),
                     'n_tested': len(post), 'log_marginal_lik': lml,
                     'fixed_hyper': hp is not None, 'degenerate_w': bool(degenerate)})
    clones = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(clones):
        clones['fold_post_mean'] = np.exp(clones['b_post_mean'])
        clones['fold_slab_lo'] = np.exp(clones['b_slab_lo'])
        clones['fold_slab_hi'] = np.exp(clones['b_slab_hi'])
    return clones, pd.DataFrame(summ)



# =============================================================================
# Per-sample dispersion phi_t (sample-specific technical noise)
# =============================================================================
#
# Samples differ in noise (input material, library quality): the pairwise analysis of
# donors showed single samples up to ~5x noisier than the rest of the same person.
# With >= 3 samples per person, per-sample dispersions are identifiable without
# replicates: clones shared across samples triangulate each sample's noise. Model
# (detected rows of clones well above the cutoff, where truncation and dropout are
# negligible):
#     x_ct ~ NB(mu_ct, phi_t),  log mu_ct = log L_t + alpha_c
# Each clone's level is re-solved at every trial phi vector; Cox-Reid adjusted.


def _phi_sample_blocks(g, count_col, lib_col, depth_col, cutoff_col, time_col,
                       min_detect, min_freq_mult, cutoff_frac):
    days = np.sort(g[time_col].unique())
    d = g[g[count_col] > 0]
    d = (d.groupby(['clono', time_col], as_index=False, sort=True)
           .agg({count_col: 'sum', lib_col: 'first', depth_col: 'first'}))
    d['n_det'] = d.groupby('clono')[time_col].transform('size')
    d = d[d['n_det'] >= min_detect]
    gg = d.groupby('clono')
    f = gg[count_col].transform('sum') / gg[depth_col].transform('sum')
    d = d[f >= min_freq_mult * cutoff_frac].reset_index(drop=True)
    if len(d) == 0:
        return None
    key = d['clono'].to_numpy()
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    n_per = np.diff(np.r_[starts, len(key)])
    x = d[count_col].to_numpy(float)
    L = d[lib_col].to_numpy(float)
    ti = np.searchsorted(days, d[time_col].to_numpy())
    lam0 = np.add.reduceat(x, starts) / np.add.reduceat(L, starts)
    return dict(days=days, x=x, L=L, ti=ti, starts=starts, n_per=n_per, lam0=lam0)


def _neg_cr_loglik_phi_vec(log_phi_vec, blk, cox_reid=True):
    phi_row = np.exp(log_phi_vec)[blk['ti']]
    mu, I, _ = _nb_mu_given_phi(blk['x'], blk['L'], blk['starts'], blk['n_per'],
                                phi_row, blk['lam0'])
    p = phi_row / (phi_row + mu)
    with np.errstate(divide='ignore', invalid='ignore'):
        ll = stats.nbinom.logpmf(blk['x'], phi_row, p).sum()
    if cox_reid:
        ll -= 0.5 * np.log(np.maximum(I, 1e-300)).sum()
    return -ll if np.isfinite(ll) else 1e300


def fit_phi_per_sample(df: pd.DataFrame,
                       count_col: str = 'count',
                       lib_col: str = 'lib',
                       depth_col: str = 'D_t',
                       cutoff_col: str = 'c_t',
                       time_col: str = 'day',
                       min_detect: int = 2,
                       min_freq_mult: float = 10.0,
                       cutoff_frac: float = CUTOFF_FRAC,
                       cox_reid: bool = True) -> pd.DataFrame:
    """Per-sample NB dispersion phi_t for every person in df (intercept-only design).

    Clones: >= min_detect detections and pooled frequency >= min_freq_mult * cutoff
    (estimation only; every clone is still tested). For each person, a single phi is
    fitted first (phi_person), then one phi per sample by L-BFGS on log phi, starting
    from it. Persons with fewer than 3 samples get phi_person for every sample
    (per-sample values are not identifiable from 2 samples).

    Note: genuinely trending clones inflate the apparent noise of the first and last
    samples (intercept-only design), which makes those phi_t lower, i.e. conservative.

    Returns one row per (patient, day): phi_sample, phi_person, rel (phi_person /
    phi_sample: > 1 means noisier than the person's average), n_rows (detected rows
    contributing at that sample), identifiable.
    """
    _require(df, ['patient', 'clono', time_col, count_col, lib_col, depth_col])
    out = []
    for pat, g in df.groupby('patient'):
        blk = _phi_sample_blocks(g, count_col, lib_col, depth_col, cutoff_col, time_col,
                                 min_detect, min_freq_mult, cutoff_frac)
        if blk is None:
            warnings.warn(f"{pat}: no clones pass the filters; skipped")
            continue
        T = len(blk['days'])
        r1 = minimize_scalar(lambda v: _neg_cr_loglik_phi_vec(np.full(T, v), blk, cox_reid),
                             bounds=PHI_BOUNDS, method='bounded')
        lp = np.full(T, r1.x)
        ident = T >= 3
        if ident:
            r = minimize(_neg_cr_loglik_phi_vec, x0=lp, args=(blk, cox_reid),
                         method='L-BFGS-B', bounds=[PHI_BOUNDS] * T)
            lp = r.x
        n_rows = np.bincount(blk['ti'], minlength=T)
        for t in range(T):
            out.append({'patient': pat, time_col: blk['days'][t],
                        'phi_sample': float(np.exp(lp[t])),
                        'phi_person': float(np.exp(r1.x)),
                        'rel': float(np.exp(r1.x - lp[t])),
                        'n_rows': int(n_rows[t]), 'identifiable': ident})
    return pd.DataFrame(out)


def conservative_phi(per_sample: pd.DataFrame, phi_ref: float,
                     time_col: str = 'day') -> pd.DataFrame:
    """Per-sample phi used in the test: min(donor reference, the sample's own phi).

    Keeps the external (donor) null as the floor: a sample is never treated as less
    noisy than the noisiest healthy donor, and a noisier sample gets its own, lower phi.
    Returns patient, time_col, phi (usable as the ``phi`` argument everywhere), and
    source ('donor reference' / 'own').
    """
    t = per_sample[['patient', time_col, 'phi_sample']].copy()
    t['phi'] = np.minimum(float(phi_ref), t['phi_sample'])
    t['source'] = np.where(t['phi_sample'] < phi_ref, 'own', 'donor reference')
    return t[['patient', time_col, 'phi', 'phi_sample', 'source']]
