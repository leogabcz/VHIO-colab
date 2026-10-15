"""
simulate -- synthetic repertoires for validating the NB pipeline.

Three generators, from closest to the pipeline's own model to furthest from it:

1. ``simulate_like_patient``            : each real clone's frequency, 1:1, on the person's
   real design (days, depths, offsets, cutoffs, N_t); cell-sampling dropout, NB reads,
   detection threshold. The observation model is the pipeline's own.
2. ``simulate_like_patient(frequencies=draw_powerlaw(...))`` : the same observation model,
   but clone frequencies drawn from the power law fitted to the person's clone-size
   distribution, so clones are not copies of the observed ones and include clones that
   were never observed.
3. ``simulate_two_step``                 : a different generator: cells, then reads.
   n_ct ~ Poisson(N f_ct) cells; x_ct ~ NB(L_t n_ct / N, phi_tech) reads. It is not the
   pipeline's model (counts given presence are a Poisson mixture of NBs), so it checks
   robustness to misspecification. ``calibrate_two_step`` fits it to a person's data.

Also here (moved from nb_glm): ``spike_in_power`` and ``plot_power``.
"""

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from tcr_pipeline.nb_glm import (PhiTrend, _phi_vector, estimate_dropout_N,
                                 nb_trajectory_test, add_qvalues, fit_phi_trend)


# =============================================================================
# Generators 1 and 2: the pipeline's observation model on a person's real design
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
                          time_col: str = 'day',
                          frequencies=None):
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

    frequencies : None (default) = each real clone's frequency, 1:1, as above. Otherwise
    an array of clone frequencies (or {patient: array}) used instead, e.g. from
    ``draw_powerlaw``; the design, phi and N_t are still the person's.

    shape='peak' : spiked clones instead rise and fall back: a Gaussian bump (width =
    span / 4) of height F centred on a random interior sampled day.
    """
    rng = np.random.default_rng(seed)
    fold_changes = np.asarray(fold_changes, float)
    sims, truths = [], []
    for pat, g in df.groupby('patient'):
        days = np.sort(g[time_col].unique())
        trend = isinstance(phi, PhiTrend)
        phi_p = None if trend else _phi_vector(phi, pat, days, time_col)[None, :]
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
        if frequencies is not None:
            lam = np.asarray(frequencies[pat] if isinstance(frequencies, dict)
                             else frequencies, float)
        if max_clones is not None and len(lam) > max_clones:
            lam = rng.choice(lam, size=max_clones, replace=False)
        n = len(lam)
        if trend:          # phi of each clone from its baseline frequency (lambda ~ f)
            phi_p = phi(lam)[:, None]
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
# Realistic clone frequencies: the power-law clone-size distribution
# =============================================================================

def pooled_frequency(g: pd.DataFrame, count_col: str = 'count', depth_col: str = 'D_t',
                     time_col: str = 'day') -> pd.Series:
    """Each clone's frequency over ALL the person's sampled days, misses counted as 0:
    sum_t x_ct / sum_t D_t (the definition of the test's min_freq filter)."""
    D = g.groupby(time_col)[depth_col].first().sum()
    return g.groupby('clono')[count_col].sum() / D


def fit_powerlaw(g: pd.DataFrame, f0: float = 3e-5, **kw) -> dict:
    """Tail of the clone-size distribution, P(F > f) = (f / f0)^-(alpha - 1) for f >= f0.

    Maximum likelihood (Hill estimator) on one person's pooled frequencies:
    alpha = 1 + n0 / sum log(f_c / f0), over the n0 clones with f_c >= f0. f0 = 3e-5 is
    30x the cutoff, where frequencies are barely affected by detection.
    Returns alpha, its SE, n0, f0 and f_max (the largest clone, used as a cap).
    """
    f = pooled_frequency(g, **kw).to_numpy()
    x = f[f >= f0]
    alpha = 1.0 + len(x) / np.log(x / f0).sum()
    return {'alpha': float(alpha), 'se': float((alpha - 1) / np.sqrt(len(x))),
            'n0': int(len(x)), 'f0': f0, 'f_max': float(f.max())}


def draw_powerlaw(pl: dict, f_lo: float = 1e-6, alpha_low: Optional[float] = None,
                  rng=None, seed: int = 0) -> np.ndarray:
    """Draw one repertoire of TRUE clone frequencies from a fitted power law.

    Above f0: exponent pl['alpha'], Poisson(n0) clones on average, capped at f_max.
    Below f0, down to f_lo: the same exponent (alpha_low=None) or another one
    (alpha_low; a broken power law), with the density continuous at f0.
    The number of clones below f0 follows from the density:
        n_lo = n0 (alpha - 1) / (alpha_low - 1) * ((f_lo / f0)^(1 - alpha_low) - 1).
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    a, n0, f0 = pl['alpha'], pl['n0'], pl['f0']
    a2 = a if alpha_low is None else alpha_low
    hi = f0 * rng.random(rng.poisson(n0)) ** (-1.0 / (a - 1.0))
    n_lo = n0 * (a - 1.0) / (a2 - 1.0) * ((f_lo / f0) ** (1.0 - a2) - 1.0)
    u = rng.random(rng.poisson(n_lo))
    e = 1.0 - a2                                  # inverse CDF of f^-a2 on [f_lo, f0]
    lo = (f_lo ** e - u * (f_lo ** e - f0 ** e)) ** (1.0 / e)
    return np.minimum(np.concatenate([hi, lo]), pl['f_max'])


# =============================================================================
# Generator 3: cells, then reads (a model different from the pipeline's)
# =============================================================================

def simulate_two_step(df: pd.DataFrame,
                      frequencies,
                      N,
                      phi_tech,
                      frac_spike: float = 0.0,
                      fold_changes=(2, 4, 10, 30),
                      seed: int = 0,
                      count_col: str = 'count',
                      lib_col: str = 'lib',
                      depth_col: str = 'D_t',
                      cutoff_col: str = 'c_t',
                      time_col: str = 'day'):
    """Simulate on each person's real design with a cells-then-reads generator.

    Per clone c and sample t (the person's real days, L_t, D_t, c_t):
        f_ct = f_c                       (or f_c * exp(b_c s_t) for spiked clones)
        n_ct ~ Poisson(N * f_ct)         cells of the clone in the sample
        x_ct ~ NB(L_t * n_ct / N, phi_tech) reads (x = 0 when n = 0)
        observed if x_ct >= c_t
    N is the number of cells (effective templates) behind a sample, phi_tech the
    dispersion of the reads given the cells (amplification and sequencing). Neither the
    NB nor the 'dropout' form exp(-N f) of the pipeline is assumed for the counts: a
    present clone's count is a Poisson mixture of NBs. For an abundant clone the
    implied dispersion approaches phi_tech; for N f of a few cells it is lower:
        1 / phi_eff(f) = 1 / phi_tech + (1 + 1 / phi_tech) / (N f)   (unconditional).

    frequencies, N, phi_tech : scalars/arrays, or {patient: value}.
    Spiked clones (frac_spike) follow a log-linear trend of fold F over the series.
    Returns (sim, truth) like ``simulate_like_patient``.
    """
    rng = np.random.default_rng(seed)
    get = lambda v, p: v[p] if isinstance(v, dict) else v
    fold_changes = np.asarray(fold_changes, float)
    sims, truths = [], []
    for pat, g in df.groupby('patient'):
        f = np.asarray(get(frequencies, pat), float)
        Np, ph = float(get(N, pat)), float(get(phi_tech, pat))
        days = np.sort(g[time_col].unique())
        per = g.groupby(time_col)[[lib_col, depth_col, cutoff_col]].first().reindex(days)
        s = (days - days.mean()) / np.ptp(days)
        n = len(f)
        spiked = np.flatnonzero(rng.random(n) < frac_spike)
        b = np.zeros(n)
        b[spiked] = rng.choice([-1, 1], len(spiked)) * np.log(rng.choice(fold_changes, len(spiked)))
        for k, t in enumerate(days):          # one sample at a time (memory)
            L, Dd, c = per.loc[t, lib_col], per.loc[t, depth_col], per.loc[t, cutoff_col]
            ft = f.copy()
            ft[spiked] *= np.exp(b[spiked] * s[k])
            cells = rng.poisson(Np * ft)
            pos = np.flatnonzero(cells > 0)
            m = L * cells[pos] / Np
            x = rng.negative_binomial(ph, ph / (ph + m))
            keep = x >= c
            sims.append(pd.DataFrame({'patient': pat, 'clono': pos[keep], time_col: t,
                                      count_col: x[keep].astype(float), lib_col: L,
                                      depth_col: Dd, cutoff_col: c}))
        truths.append(pd.DataFrame({'patient': pat, 'clono': np.arange(n),
                                    'spiked': b != 0, 'log_fold_true': b,
                                    'log10_f': np.log10(f)}))
    return pd.concat(sims, ignore_index=True), pd.concat(truths, ignore_index=True)


def repertoire_summary(g: pd.DataFrame, f_est: float = 1e-5, f0: float = 3e-5,
                       count_col: str = 'count', time_col: str = 'day') -> dict:
    """Statistics used to compare a simulated repertoire with the real one."""
    det = g[g[count_col] > 0]
    per_sample = det.groupby(time_col).size()
    n_det = det.groupby('clono')[time_col].nunique()
    f_det = det.groupby('clono')[count_col].sum() / det.groupby('clono')['D_t'].sum()
    return {'clones per sample (median)': int(per_sample.median()),
            'distinct clones': int(len(n_det)),
            'fraction seen once': float((n_det == 1).mean()),
            'clones used for phi': int(((n_det >= 2) & (f_det >= f_est)).sum()),
            'median count': float(det[count_col].median()),
            'power-law alpha (f >= 3e-5)': fit_powerlaw(det, f0=f0)['alpha']}


def calibrate_two_step(g: pd.DataFrame, f_lo: float = 1e-7, f0: float = 3e-5,
                       alpha_low_range=(1.3, 2.6), max_clones: float = 3e7,
                       seed: int = 0, **trend_kw) -> dict:
    """Fit the cells-then-reads generator to ONE person.

    * Tail of the clone sizes: power law above f0 (``fit_powerlaw``).
    * N and phi_tech: from the person's binned dispersion (``fit_phi_trend``), by
      weighted least squares of 1/phi_bin = 1/phi_tech + (1 + 1/phi_tech) / (N f_bin),
      the dependence the generator implies (see ``simulate_two_step``).
    * Clone sizes below f0 (never seen often enough to fit directly): a second exponent
      alpha_low, chosen so that the simulated median number of detected clones per
      sample equals the real one (bisection with a fixed seed; the number of clones is
      capped at max_clones for memory, which bounds alpha_low from above).

    Returns dict: pl (power-law fit), alpha_low, N, phi_tech, the 1/phi fit, and
    'at_cap' (True if the clone cap stopped the match: the simulated repertoire then has
    fewer detected clones than the real one).
    """
    pl = fit_powerlaw(g, f0=f0)
    bins = fit_phi_trend(g, **trend_kw)['bins']
    bins = bins[~bins['at_bound']]
    f = 10 ** bins['log10_freq'].to_numpy()
    y, w = 1 / bins['phi'].to_numpy(), np.sqrt(bins['n_clones'].to_numpy())
    A = np.column_stack([np.ones_like(f), 1 / f])
    (a, k), *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    if a <= 0 or k <= 0:
        raise ValueError(f"1/phi = a + k/f fit not usable (a={a:.3g}, k={k:.3g})")
    phi_tech, N = 1 / a, (1 + a) / k
    target = g[g['count'] > 0].groupby('day').size().median()

    def n_clones(a2):
        return pl['n0'] * (pl['alpha'] - 1) / (a2 - 1) * ((f_lo / f0) ** (1 - a2) - 1)

    def per_sample(a2):
        lam = draw_powerlaw(pl, f_lo=f_lo, alpha_low=a2, seed=seed)
        sim, _ = simulate_two_step(g, lam, N, phi_tech, seed=seed)
        return sim.groupby('day').size().median()

    lo, hi = alpha_low_range
    hi = min(hi, brentq(lambda a2: n_clones(a2) - max_clones, 1.01, 5.0))
    at_cap = per_sample(hi) < target
    if at_cap:
        a2 = hi
    else:
        for _ in range(12):                    # per_sample increases with alpha_low
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if per_sample(mid) < target else (lo, mid)
        a2 = 0.5 * (lo + hi)
    return {'pl': pl, 'alpha_low': float(a2), 'N': float(N), 'phi_tech': float(phi_tech),
            'f_lo': f_lo, 'n_clones': float(n_clones(a2)), 'at_cap': bool(at_cap),
            'inv_phi_fit': {'a': float(a), 'k': float(k)}}


def phi_eff(f, N: float, phi_tech: float):
    """Unconditional dispersion implied by the two-step generator at frequency f."""
    f = np.asarray(f, float)
    return 1.0 / (1.0 / phi_tech + (1.0 + 1.0 / phi_tech) / (N * f))
