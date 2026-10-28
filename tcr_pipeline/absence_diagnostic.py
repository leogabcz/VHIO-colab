# =============================================================================
# 2.6 Diagnostic: are absences of abundant donor clones as frequent as the model says?
#
# Leave-one-sample-out. For each donor and each sample t:
#   * take clones detected in >= MIN_OTHER of the donor's OTHER samples,
#   * estimate each clone's frequency from those other samples only,
#   * keep clones above the phi-estimation floor (same floor as Section 2.3),
#   * record whether the clone is missing (count < c = 1) at t.
# The observed miss rate is compared with:
#   * plug-in predictions: NB only, P(X = 0), and NB + cell dropout
#     (pi_t + (1 - pi_t) P(X = 0), pi_t = exp(-N_t lambda)), using N_t from
#     nb.estimate_dropout_N;
#   * the SAME diagnostic run on donors simulated from the fitted model (NB + dropout,
#     the donor's own phi and N_t). This is the fair reference: the clone frequency is
#     estimated from only 2-4 samples, and P(miss) is convex in it, so even a perfect
#     model shows observed > plug-in. Excess over the simulated rate is model misfit.
# =============================================================================
import numpy as np
import pandas as pd
from scipy import stats

FLOOR_FREQ = 1e-5      # phi-estimation floor (pooled frequency relative to D_t)
MIN_OTHER = 2          # detections required in the other samples
N_SIM = 3              # simulated donors per real donor
FREQ_BINS = np.array([1e-5, 2e-5, 5e-5, 1e-4, 1e-3, 1.0])


def _donor_matrix(g, count_col='count', prefilter=FLOOR_FREQ / 5):
    """Clone x sample count matrix (only clones that could pass the floor)."""
    days = np.sort(g['day'].unique())
    per_day = g.groupby('day', observed=True)[['lib', 'D_t']].first().reindex(days)
    L, D = per_day['lib'].to_numpy(float), per_day['D_t'].to_numpy(float)
    tot = g.groupby('clono', observed=True)[count_col].sum()
    keep = tot.index[tot.to_numpy(float) / D.sum() >= prefilter]
    sub = g[g['clono'].isin(keep)]
    X = (sub.pivot_table(index='clono', columns='day', values=count_col, aggfunc='sum',
                         fill_value=0, observed=True)
            .reindex(columns=days, fill_value=0).to_numpy(float))
    return days, X, L, D


def _loso_records(X, L, D, phi, N, c=1):
    """One record per (clone, held-out sample) for clones passing the floor in the others."""
    det = X >= c
    n_s = X.shape[1]
    recs = []
    for t in range(n_s):
        o = np.delete(np.arange(n_s), t)
        ok = det[:, o].sum(1) >= MIN_OTHER
        f = (X[ok][:, o] * det[ok][:, o]).sum(1) / (det[ok][:, o] * D[o]).sum(1)  # rel. D_t
        lam = (X[ok][:, o] * det[ok][:, o]).sum(1) / (det[ok][:, o] * L[o]).sum(1)  # rel. L_t
        sel = f >= FLOOR_FREQ
        f, lam = f[sel], lam[sel]
        mu = lam * L[t]
        p0 = stats.nbinom.cdf(c - 1, phi, phi / (phi + mu))         # NB: P(X < c)
        pi = np.exp(-N[t] * lam)
        recs.append(pd.DataFrame({'t': t, 'freq': f,
                                  'missing': ~det[ok][sel, t],
                                  'pred_nb': p0,
                                  'pred_dropout': pi + (1 - pi) * p0}))
    return pd.concat(recs, ignore_index=True)


def _simulate_like(X, L, D, phi, N, rng, c=1):
    """Donor with the same samples, clone frequencies from all samples, NB + dropout."""
    det = X >= c
    lam = (X * det).sum(1) / np.maximum((det * L).sum(1), 1e-300)
    lam = lam[det.sum(1) >= 1]
    mu = lam[:, None] * L[None, :]
    present = rng.random(mu.shape) >= np.exp(-N[None, :] * lam[:, None])
    Xs = rng.negative_binomial(phi, phi / (phi + mu)) * present
    return Xs.astype(float)


def absence_diagnostic(d, phis, N_tables=None, n_sim=N_SIM, seed=0, count_col='count'):
    """phis: {donor: phi} (e.g. the floored truncated-CR column of Section 2.3).
    N_tables: optional {donor: estimate_dropout_N output}; computed if missing."""
    rng = np.random.default_rng(seed)
    by_bin, by_sample = [], []
    for p, g in d.groupby('patient', observed=True):
        phi = float(phis[p])
        if N_tables is not None and p in N_tables:
            Nt = N_tables[p]
        else:
            Nt = nb.estimate_dropout_N(g, phi=phi, c_val=1)
        days, X, L, D = _donor_matrix(g, count_col)
        N = Nt.set_index('day').reindex(days)['N_t'].to_numpy(float)

        real = _loso_records(X, L, D, phi, N)
        sims = pd.concat([_loso_records(_simulate_like(X, L, D, phi, N, rng), L, D, phi, N)
                          .assign(rep=r) for r in range(n_sim)], ignore_index=True)
        for df_ in (real, sims):
            df_['bin'] = pd.cut(df_['freq'], FREQ_BINS, right=False)
            df_['day'] = days[df_['t']]

        def summarise(key):
            r = real.groupby(key, observed=True).agg(
                n=('missing', 'size'), obs_miss=('missing', 'mean'),
                pred_nb=('pred_nb', 'mean'), pred_dropout=('pred_dropout', 'mean'))
            s = sims.groupby(key, observed=True)['missing'].mean().rename('sim_miss')
            out = r.join(s)
            out['excess'] = out['obs_miss'] - out['sim_miss']       # absolute, per clone
            out['ratio'] = out['obs_miss'] / out['sim_miss'].where(out['sim_miss'] > 0)
            return out.reset_index().assign(donor=short(p) if 'short' in globals() else p)

        by_bin.append(summarise('bin'))
        by_sample.append(summarise('day'))
    cols = ['donor', 'n', 'obs_miss', 'sim_miss', 'excess', 'ratio', 'pred_dropout', 'pred_nb']
    return (pd.concat(by_bin)[['donor', 'bin'] + cols[1:]],
            pd.concat(by_sample)[['donor', 'day'] + cols[1:]])


def plot_absence(by_bin, ax=None):
    import matplotlib.pyplot as plt
    donors = by_bin['donor'].unique()
    fig, axes = plt.subplots(1, len(donors), figsize=(2.6 * len(donors), 2.9),
                             sharey=True, squeeze=False)
    for ax, dn in zip(axes[0], donors):
        b = by_bin[by_bin['donor'] == dn]
        x = np.array([np.sqrt(iv.left * min(iv.right, 1e-2)) for iv in b['bin']])
        pos = lambda s: s.where(s > 0)          # zeros cannot be shown on a log axis
        ax.plot(x, pos(b['obs_miss']), 'o-', color='k', label='observed')
        ax.plot(x, pos(b['sim_miss']), 's--', color='#2a78d6', label='expected (simulated model)')
        ax.plot(x, pos(b['pred_nb']), ':', color='#8a8984', label='NB only, plug-in')
        ax.set(xscale='log', yscale='log', title=dn)
    axes[0, 0].set_ylabel('P(missing at held-out sample)')
    fig.supxlabel('clone frequency, estimated from the other samples', fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig


# ---- run on the real donors (needs d_qc, est_floor and short from Section 2.3) ----
# phis = {p: est_floor.loc[short(p), 'truncated + Cox-Reid (used)'] for p in DONORS}
# absence_by_bin, absence_by_sample = absence_diagnostic(d_qc, phis)
# display(absence_by_bin.round(4))
# display(absence_by_sample.round(4))
# plot_absence(absence_by_bin);
