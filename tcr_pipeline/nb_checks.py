"""
nb_checks -- checks and validation for the NB pipeline (Notebook 2).

Everything here takes the long-format donor table used in the notebook (patient, day,
clono, count, D_t, lib; lib from ``nb_glm.attach_size_factor_offset``) and the
``nb_glm`` module itself, so the notebook cells stay short.

Contents
--------
phi_recovery_simulation  known-phi recovery of the three estimators vs frequency floor
estimator_table          the three phi estimators on real donors, with / without floor
noise_by_donor           phi per frequency bin with 95% profile intervals, trend slope
reference_trend          pointwise minimum of donors' phi trends (the patient reference)
absence_diagnostic       observed vs model-expected misses of abundant clones
lodo / lodo_summary      leave-one-donor-out calibration and its verdict
fpr_by_frequency         where the false calls of a LODO run sit, by clone frequency
"""
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

FLOOR = 1e-5          # phi-estimation floor and test filter: pooled frequency / D_t
K_LOW = 3             # counts below K_LOW are not trusted as exact values in the test
ALPHA = 0.05


def short(p):
    return str(p).replace('_SEQTR', '').replace('_ReSeq', '')


def subset(d: pd.DataFrame, donors) -> pd.DataFrame:
    """Rows of the given persons, with unused categories dropped (pandas-2 safe)."""
    donors = [donors] if isinstance(donors, str) else list(donors)
    s = d[d['patient'].isin(donors)].copy()
    if hasattr(s['patient'], 'cat'):
        s['patient'] = s['patient'].cat.remove_unused_categories()
    return s


# =============================================================================
# 1. Known-phi recovery (why the frequency floor)
# =============================================================================

def simulate_neutral_donor(phi, n_clones=4000, n_days=5, lib_range=(5e4, 3e5),
                           f_min=3e-6, f_max=1e-2, seed=0):
    """Neutral donor with known phi: x_ct ~ NB(f_c * L_t, phi), sparse (count >= 1).
    Pareto clone frequencies, so most clones sit near the detection limit, as in TCR data."""
    rng = np.random.default_rng(seed)
    L = np.exp(rng.uniform(*np.log(lib_range), n_days))
    f = np.minimum(f_min * (1 + rng.pareto(1.0, n_clones)), f_max)
    mu = f[:, None] * L[None, :]
    x = rng.negative_binomial(phi, phi / (phi + mu))
    ci, ti = np.nonzero(x >= 1)
    return pd.DataFrame({'patient': 'sim', 'day': ti, 'clono': ci,
                         'count': x[ci, ti], 'lib': L[ti], 'D_t': L[ti]})


FITTERS = {'legacy':   dict(resolve_mean=False, truncate=False),
           'CR':       dict(resolve_mean=True,  truncate=False),
           'CR+trunc': dict(resolve_mean=True,  truncate=True)}


def phi_recovery_simulation(nb, phis=(0.5, 1.0, 4.0), floors_reads=(0, 5, 10, 20),
                            n_reps=3):
    """Median phi_hat / phi_true of the three estimators, by frequency floor (in reads)."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for phi in phis:
            for rep in range(n_reps):
                df = simulate_neutral_donor(phi, seed=rep)
                per_read = 1.0 / np.median(df['D_t'].unique())
                for fl in floors_reads:
                    for name, kw in FITTERS.items():
                        r = nb.fit_nb_dispersion_cr(df, min_freq_mult=fl, cutoff_frac=per_read,
                                                    c_val=1, **kw)
                        rows.append(dict(phi_true=phi, floor_reads=fl, fitter=name,
                                         ratio=r['phi'] / phi, n_clones=r['n_clones']))
    res = pd.DataFrame(rows)
    out = (res.groupby(['phi_true', 'floor_reads', 'fitter'])['ratio'].median()
              .unstack('fitter')[list(FITTERS)])
    out['clones kept'] = (res.groupby(['phi_true', 'floor_reads'])['n_clones']
                             .median().astype(int))
    return out


# =============================================================================
# 2. phi on real donors
# =============================================================================

def estimator_table(nb, d, donors, floor=FLOOR):
    """The three estimators per donor, with the frequency floor and without it."""
    mult = floor / nb.CUTOFF_FRAC
    rows = []
    for p in donors:
        g = subset(d, p)
        for label, m in [('with floor', mult), ('no floor', 0)]:
            fit = lambda **kw: nb.fit_nb_dispersion_cr(
                g, min_detect=2, min_freq_mult=m, c_val=1, n_bins=1,
                min_bin_clones=10**9, **kw)
            used = fit(truncate=True, cox_reid=True)
            rows.append({'donor': short(p), 'clones': label,
                         'truncated + Cox-Reid (used)': used['phi'],
                         'not truncated': fit(truncate=False, cox_reid=True)['phi'],
                         'no Cox-Reid': fit(truncate=True, cox_reid=False)['phi'],
                         'n_clones': used['n_clones']})
    return pd.DataFrame(rows).set_index(['clones', 'donor']).sort_index()


TREND_KW = dict(c_val=1, min_detect=2, min_freq_mult=10, n_bins=6, min_bin_clones=200,
                truncate=True, cox_reid=True)


def _bin_blocks(nb, g, kw):
    """Detected rows of one donor, split into the same frequency bins as fit_phi_trend."""
    x, L, Dd, starts, n_per, c = nb._sparse_blocks(
        g, 'count', 'lib', 'D_t', kw['min_detect'], kw['c_val'] if kw['truncate'] else None)
    sx = np.add.reduceat(x, starts)
    p_hat = sx / np.add.reduceat(L, starts)
    p_depth = sx / np.add.reduceat(Dd, starts)
    keep = p_depth >= kw['min_freq_mult'] * nb.CUTOFF_FRAC
    obs = np.repeat(keep, n_per)
    x, L = x[obs], L[obs]
    c = None if c is None else c[obs]
    n_per, p_hat, p_depth = n_per[keep], p_hat[keep], p_depth[keep]
    lp = np.log10(p_depth)
    edges = np.unique(np.quantile(lp, np.linspace(0, 1, kw['n_bins'] + 1)))
    b_of = np.clip(np.digitize(lp, edges[1:-1]), 0, len(edges) - 2)
    for k in range(len(edges) - 1):
        m = b_of == k
        if m.sum() < kw['min_bin_clones']:
            continue
        o = np.repeat(m, n_per)
        npb = n_per[m]
        yield (float(np.median(lp[m])), int(m.sum()), x[o], L[o],
               np.concatenate([[0], np.cumsum(npb)[:-1]]), npb, p_hat[m],
               None if c is None else c[o])


def _profile_ci(nb, x, L, starts, n_per, lam0, c, cox_reid, level=3.84 / 2):
    """phi_hat and 95% profile-likelihood interval (root-finding on each side)."""
    f = lambda lp: nb._nb_neg_loglik_cr(lp, x, L, starts, n_per, lam0, True, c, cox_reid)
    phi_hat, _ = nb._fit_phi(x, L, starts, n_per, lam0, True, c, cox_reid)
    l0 = np.log(phi_hat)
    g = lambda lp: f(lp) - (f(l0) + level)
    ends = []
    for sign in (-1, 1):
        step, prev = 0.02, l0
        while True:
            nxt = l0 + sign * step
            if g(nxt) > 0:
                ends.append(brentq(g, min(prev, nxt), max(prev, nxt), xtol=1e-4))
                break
            prev, step = nxt, step * 2
            if step > 4:
                ends.append(nxt)
                break
    return phi_hat, np.exp(ends[0]), np.exp(ends[1])


def noise_by_donor(nb, d, donors, null_donors, kw=TREND_KW):
    """phi per frequency bin (95% profile CI), log-linear trend slope, per donor.

    Returns (summary, bins, fits); fits[p] is the fit_phi_trend result of donor p.
    """
    rows, fits = [], {}
    for p in donors:
        g = subset(d, p)
        fits[p] = nb.fit_phi_trend(g, **kw)
        for lf, n_cl, x, L, st, npb, lam0, c in _bin_blocks(nb, g, kw):
            phi, lo, hi = _profile_ci(nb, x, L, st, npb, lam0, c, kw['cox_reid'])
            rows.append(dict(donor=p, null=p in null_donors, log10_freq=lf, n_clones=n_cl,
                             phi=phi, ci_lo=lo, ci_hi=hi))
    bins = pd.DataFrame(rows)
    summ = []
    for p, b in bins.groupby('donor', sort=False):
        y = np.log(b['phi'].to_numpy())
        se = (np.log(b['ci_hi']) - np.log(b['ci_lo'])).to_numpy() / 3.92
        w = 1 / np.maximum(se, 1e-3) ** 2
        X = np.column_stack([np.ones(len(b)), b['log10_freq'].to_numpy()])
        cov = np.linalg.inv(X.T @ (w[:, None] * X))
        coef = cov @ X.T @ (w * y)
        summ.append(dict(donor=p, null=b['null'].iloc[0], phi_global=fits[p]['phi_global'],
                         n_clones=fits[p]['n_clones'], slope_b=coef[1],
                         slope_se=np.sqrt(cov[1, 1]), slope_z=coef[1] / np.sqrt(cov[1, 1]),
                         line_chi2=float((w * (y - X @ coef) ** 2).sum()),
                         line_df=len(b) - 2))
    return pd.DataFrame(summ), bins, fits


def reference_trend(nb, fits, exclude=None):
    """Pointwise minimum of the donors' phi trends (the noisiest donor at each frequency)."""
    ex = set([exclude] if isinstance(exclude, str) else (exclude or []))
    return nb.PhiTrend.minimum([v['trend'] for k, v in fits.items() if k not in ex])


# =============================================================================
# 3. Absences of abundant clones: observed vs model-expected
# =============================================================================

_MISS_BINS = np.array([1e-5, 2e-5, 5e-5, 1e-4, 1e-3, 1.0])


def _donor_matrix(g, prefilter):
    days = np.sort(g['day'].unique())
    per_day = g.groupby('day', observed=True)[['lib', 'D_t']].first().reindex(days)
    L, D = per_day['lib'].to_numpy(float), per_day['D_t'].to_numpy(float)
    tot = g.groupby('clono', observed=True)['count'].sum()
    keep = tot.index[tot.to_numpy(float) / D.sum() >= prefilter]
    sub = g[g['clono'].isin(keep)]
    X = (sub.pivot_table(index='clono', columns='day', values='count', aggfunc='sum',
                         fill_value=0, observed=True)
            .reindex(columns=days, fill_value=0).to_numpy(float))
    return days, X, L, D


def _loso_records(X, L, D, phi, N, floor, min_other=2):
    det = X >= 1
    recs = []
    for t in range(X.shape[1]):
        o = np.delete(np.arange(X.shape[1]), t)
        ok = det[:, o].sum(1) >= min_other
        num = (X[ok][:, o] * det[ok][:, o]).sum(1)
        f = num / (det[ok][:, o] * D[o]).sum(1)
        lam = num / (det[ok][:, o] * L[o]).sum(1)
        sel = f >= floor
        f, lam = f[sel], lam[sel]
        mu = lam * L[t]
        p0 = stats.nbinom.cdf(0, phi, phi / (phi + mu))
        pi = np.exp(-N[t] * lam)
        recs.append(pd.DataFrame({'t': t, 'freq': f, 'missing': ~det[ok][sel, t],
                                  'pred_nb': p0, 'pred_dropout': pi + (1 - pi) * p0}))
    return pd.concat(recs, ignore_index=True)


def _simulate_like(X, L, phi, N, rng):
    det = X >= 1
    lam = (X * det).sum(1) / np.maximum((det * L).sum(1), 1e-300)
    lam = lam[det.sum(1) >= 1]
    mu = lam[:, None] * L[None, :]
    present = rng.random(mu.shape) >= np.exp(-N[None, :] * lam[:, None])
    return (rng.negative_binomial(phi, phi / (phi + mu)) * present).astype(float)


def absence_diagnostic(nb, d, phis: dict, n_sim=3, seed=0, floor=FLOOR):
    """Leave-one-sample-out miss rates of clones above the floor: observed vs the same
    procedure on donors simulated from the fitted model (NB + cell dropout, the donor's
    own phi and N_t). Returns (by frequency bin, by sample)."""
    rng = np.random.default_rng(seed)
    by_bin, by_sample = [], []
    for p, phi in phis.items():
        g = subset(d, p)
        Nt = nb.estimate_dropout_N(g, phi=float(phi), c_val=1)
        days, X, L, D = _donor_matrix(g, floor / 5)
        N = Nt.set_index('day').reindex(days)['N_t'].to_numpy(float)
        real = _loso_records(X, L, D, phi, N, floor)
        sims = pd.concat([_loso_records(_simulate_like(X, L, phi, N, rng), L, D, phi, N, floor)
                          for _ in range(n_sim)], ignore_index=True)
        for df_ in (real, sims):
            df_['bin'] = pd.cut(df_['freq'], _MISS_BINS, right=False)
            df_['day'] = days[df_['t']]
        for key, store in (('bin', by_bin), ('day', by_sample)):
            r = real.groupby(key, observed=True).agg(
                n=('missing', 'size'), obs_miss=('missing', 'mean'),
                pred_dropout=('pred_dropout', 'mean'), pred_nb=('pred_nb', 'mean'))
            r = r.join(sims.groupby(key, observed=True)['missing'].mean().rename('sim_miss'))
            r['excess'] = r['obs_miss'] - r['sim_miss']
            store.append(r.reset_index().assign(donor=short(p)))
    cols = ['donor', 'n', 'obs_miss', 'sim_miss', 'excess', 'pred_dropout', 'pred_nb']
    return (pd.concat(by_bin)[['donor', 'bin'] + cols[1:]],
            pd.concat(by_sample)[['donor', 'day'] + cols[1:]])


def plot_absence(by_bin):
    import matplotlib.pyplot as plt
    donors = by_bin['donor'].unique()
    fig, axes = plt.subplots(1, len(donors), figsize=(2.6 * len(donors), 2.9),
                             sharey=True, squeeze=False)
    for ax, dn in zip(axes[0], donors):
        b = by_bin[by_bin['donor'] == dn]
        x = np.array([np.sqrt(iv.left * min(iv.right, 1e-2)) for iv in b['bin']])
        pos = lambda s: s.where(s > 0)
        ax.plot(x, pos(b['obs_miss']), 'o-', color='k', label='observed')
        ax.plot(x, pos(b['sim_miss']), 's--', color='#2a78d6', label='expected (model)')
        ax.set(xscale='log', yscale='log', title=dn)
    axes[0, 0].set_ylabel('P(missing at a sample)')
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel('clone frequency, estimated from the other samples', fontsize=9)
    fig.tight_layout()
    return fig


# =============================================================================
# 4. Leave-one-donor-out calibration
# =============================================================================

def harmonic_mean_trend(nb, trends, n=80):
    """Average of PhiTrends on the 1/phi (excess-variance) scale."""
    lo = min(t.log10_f[0] for t in trends)
    hi = max(t.log10_f[-1] for t in trends)
    grid = np.linspace(lo, hi, n)
    inv = np.mean([1.0 / t(10.0 ** grid) for t in trends], axis=0)
    return nb.PhiTrend(grid, 1.0 / inv)


def test_person(nb, g, phi, likelihood='truncated', k=K_LOW, floor=FLOOR,
                beta_bound=np.log(100), n_jobs=1):
    """nb_trajectory_test on one person: counts below k are not detected (truncated at k)."""
    g = g.assign(c_t=int(k))
    kw = dict(likelihood=likelihood, min_freq=floor, beta_bound=beta_bound, n_jobs=n_jobs,
              cutoff_col='c_t')
    if likelihood == 'dropout':
        kw['dropout_N'] = nb.estimate_dropout_N(g, phi=phi, c_val=int(k))
    res = nb.add_qvalues(nb.nb_trajectory_test(g, phi, **kw), p_col='p_value', by='patient')
    res['n_miss'] = (res['n_timepoints'] - res['n_detected']).astype(int)
    return res


def lodo(nb, d, fits: dict, likelihood='truncated', reference='min', k=K_LOW,
         floor=FLOOR, n_jobs=1, verbose=True):
    """Hold out each donor in fits, test it like a patient against the others."""
    out = []
    for held in fits:
        others = [v['trend'] for kk, v in fits.items() if kk != held]
        ref = nb.PhiTrend.minimum(others) if reference == 'min' else \
            harmonic_mean_trend(nb, others)
        res = test_person(nb, subset(d, held), ref, likelihood, k, floor, n_jobs=n_jobs)
        out.append(res.assign(held_out=held, likelihood=likelihood, reference=reference, k=k))
        if verbose:
            print(f"{short(held):8s} {likelihood:9s} ref={reference:4s} k={k}: "
                  f"{len(res):,} clones, FPR = {(res['p_value'] < ALPHA).mean():.3f}")
    return pd.concat(out, ignore_index=True)


def lodo_summary(res, by=('likelihood', 'reference', 'k')):
    """Per held-out donor: FPR with its binomial limit, BH calls, direction; and a verdict.
    Pass = FPR <= 5% + 2 binomial SE and <= 1 BH call, for every held-out donor."""
    by = [b for b in by if b in res.columns]

    def one(g):
        s = g['p_value'] < ALPHA
        n = len(g)
        return pd.Series({'n_tested': n, 'FPR_05': s.mean(),
                          'limit': ALPHA + 2 * np.sqrt(ALPHA * (1 - ALPHA) / max(n, 1)),
                          'FPR_01': (g['p_value'] < 0.01).mean(),
                          'BH': int((g['q_value'] < ALPHA).sum()),
                          'frac_expanding_sig': (g.loc[s, 'beta'] > 0).mean(),
                          'frac_expanding_all': (g['beta'] > 0).mean()})
    per = res.groupby(by + ['held_out']).apply(one).reset_index()
    per['held_out'] = per['held_out'].map(short)
    per['passes'] = (per['FPR_05'] <= per['limit']) & (per['BH'] <= 1)
    pooled = res.groupby(by).apply(lambda g: (g['p_value'] < ALPHA).mean()).rename('pooled_FPR')
    verdict = (per.groupby(by).agg(all_pass=('passes', 'all'), max_FPR=('FPR_05', 'max'),
                                   total_BH=('BH', 'sum'))
                  .join(pooled).reset_index())
    verdict['failing'] = verdict.apply(
        lambda v: ', '.join(per[(per[by] == v[by]).all(axis=1) & ~per['passes']]['held_out'])
        or '-', axis=1)
    return per, verdict


def lodo_by_miss(res, by=('likelihood', 'reference', 'k')):
    """FPR by number of missing timepoints. DIAGNOSTIC ONLY: n_miss is part of the outcome,
    so p-values are skewed small for clones with a miss even under a correct model."""
    by = [b for b in by if b in res.columns]
    t = (res.groupby(by + ['held_out', 'n_miss'])
            .agg(n=('p_value', 'size'), FPR_05=('p_value', lambda p: (p < ALPHA).mean()))
            .reset_index())
    t['held_out'] = t['held_out'].map(short)
    return t


def fpr_by_frequency(nb, d, res, fits, edges=(1e-5, 2e-5, 5e-5, 1e-4, 1e-3, 1.0)):
    """FPR and direction per frequency bin of one LODO run, with the held-out donor's own
    phi relative to the reference it was tested with (own_vs_ref < 1: noisier than ref)."""
    edges = np.asarray(edges)
    rows = []
    for held, r in res.groupby('held_out'):
        g = subset(d, held)
        days = np.sort(g['day'].unique())
        D = g.groupby('day', observed=True)['D_t'].first().reindex(days).to_numpy(float)
        Xw = g.pivot_table(index='clono', columns='day', values='count', aggfunc='sum',
                           fill_value=0, observed=True).reindex(columns=days, fill_value=0)
        f = pd.Series(nb.clone_frequency(Xw.to_numpy(float), Xw.to_numpy() >= 1, D),
                      index=Xw.index.astype(str))
        rr = r.assign(freq=r['clono'].astype(str).map(f))
        ref = reference_trend(nb, fits, exclude=held)
        own = fits[held]['trend']
        rr['bin'] = pd.cut(rr['freq'], edges, right=False)
        for b, s in rr.groupby('bin', observed=True):
            sig = s['p_value'] < ALPHA
            fm = np.sqrt(b.left * min(b.right, 1e-2))
            rows.append(dict(donor=short(held), bin=str(b), n=len(s), FPR_05=sig.mean(),
                             frac_expanding_sig=(s.loc[sig, 'beta'] > 0).mean(),
                             own_vs_ref=float(own(fm) / ref(fm))))
    return pd.DataFrame(rows)
