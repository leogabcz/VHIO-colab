"""Dispersion experiments for the donor NB null.

Experiment 1 (ladder): legacy fit vs the new fit, switching on one correction at a time,
    globally and per frequency bin, so each change in phi can be attributed.
Experiment 2 (per donor): truncated Cox-Reid fit per donor, with profile-likelihood CIs,
    and a likelihood-ratio test of one shared phi against one phi per donor.
Diagnostics: per-bin profile curves of the objective, and a simulation with constant phi
    on the real sampling designs, to separate method artefacts from real structure.

Nothing here modifies nb_glm; it only calls it. Usage in the notebook:

    import dispersion_experiments as dx
    legacy_in = healthy_raw.loc[healthy_raw['count'] > 0, ['patient', 'day', 'clono', 'count']]
    lad = dx.run_ladder(nb, d, legacy=nb_legacy, legacy_df=legacy_in)
    lad['global']; dx.ladder_bin_table(lad); dx.ladder_freq_table(lad); dx.plot_ladder_bins(lad)
    pdn = dx.per_donor(nb, d)
    prof = dx.profile_curves(nb, d)
    sim = dx.simulate_like_real(d, dx.lambda_pool(d), phi=2.0)
"""
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

# One correction switched on per step; S4 is the new default fit.
LADDER = [
    ('S1 sparse, no corrections', dict(cox_reid=False, resolve_mean=False, truncate=False)),
    ('S2 + Cox-Reid',             dict(cox_reid=True,  resolve_mean=False, truncate=False)),
    ('S3 + exact NB mean',        dict(cox_reid=True,  resolve_mean=True,  truncate=False)),
    ('S4 + truncation (new)',     dict(cox_reid=True,  resolve_mean=True,  truncate=True)),
]


def _bin_mid(label):
    lo, hi = map(float, re.findall(r'-?\d+\.\d+', label)[:2])
    return (lo + hi) / 2


# -----------------------------------------------------------------------------
# Experiment 1: ladder
# -----------------------------------------------------------------------------

def legacy_fit(legacy, legacy_df, count_col='count', n_bins=6):
    """Legacy NB fit as originally run: sparse detected rows, column-sum lib,
    pooled mean, untruncated, no CR, no frequency floor, clones with >= 2 rows.

    legacy_df must hold only patient, day, clono, count (a 'lib' column breaks the
    legacy merge). Uses fit_nb_dispersion_trended, whose 'global_phi' is the same fit
    as fit_nb_dispersion_mle, so one pass gives both.

    Returns (global_phi, bins DataFrame, n_clones).
    """
    tr = legacy.fit_nb_dispersion_trended(legacy_df, count_col=count_col, n_bins=n_bins)
    e = np.asarray(tr['logfreq_edges'])
    bins = pd.DataFrame({
        'bin_log10_freq': [f"[{e[i]:.2f},{e[i + 1]:.2f})" for i in range(len(e) - 1)],
        'mid_log10_freq': (e[:-1] + e[1:]) / 2,
        'n_clones': np.nan,
        'phi': [np.nan if p is None else p for p in tr['phi_per_bin']],
    })
    n_clones = int((legacy_df.loc[legacy_df[count_col] > 0]
                    .groupby(['patient', 'clono']).size() >= 2).sum())
    return float(tr['global_phi']), bins, n_clones


def run_ladder(nb, df, legacy=None, legacy_df=None, steps=LADDER, n_bins=6,
               legacy_n_bins=None, grid_df=None, **fit_kwargs):
    """Fit phi at each step of the ladder.

    df            : sparse donor data with lib, D_t and c (as used by fit_nb_dispersion_cr).
    legacy        : the legacy nb_glm module (optional).
    legacy_df     : its input, as the legacy fit was run (sparse; patient, day, clono, count).
    n_bins        : frequency bins for the new fits.
    legacy_n_bins : frequency bins for the legacy fit (default: n_bins). Use 10 to
                    reproduce the original legacy binning.
    grid_df       : deprecated alias of legacy_df.
    fit_kwargs    : passed to every new fit (min_detect, min_freq_mult, ...).
                    min_freq_mult=0.0 gives the new fits the same clones as the legacy fit.

    Returns dict with 'global' (one row per step) and 'bins' (long format).
    S1-S4 share clones and bin edges, so their bins are directly comparable; the legacy
    bins use a different frequency scale and edges (compare with ladder_freq_table or
    plot_ladder_bins).
    """
    if legacy_df is None:
        legacy_df = grid_df
    glob, bins = [], []
    if legacy is not None:
        t0 = time.time()
        phi, b, n_cl = legacy_fit(legacy, legacy_df, n_bins=legacy_n_bins or n_bins)
        glob.append({'step': 'S0 legacy', 'phi': phi, 'n_clones': n_cl,
                     'seconds': round(time.time() - t0, 1), 'warnings': ''})
        bins.append(b.assign(step='S0 legacy'))
    for name, flags in steps:
        t0 = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                r = nb.fit_nb_dispersion_cr(df, n_bins=n_bins, **flags, **fit_kwargs)
        msgs = sorted({str(x.message) for x in w})
        glob.append({'step': name, 'phi': r['phi'], 'n_clones': r['n_clones'],
                     'seconds': round(time.time() - t0, 1), 'warnings': '; '.join(msgs)})
        b = r['bins'].copy()
        b['mid_log10_freq'] = b['bin_log10_freq'].map(_bin_mid)
        bins.append(b[['bin_log10_freq', 'mid_log10_freq', 'n_clones', 'phi']].assign(step=name))
    g = pd.DataFrame(glob)
    g['change_vs_prev'] = g['phi'].pct_change()
    return {'global': g, 'bins': pd.concat(bins, ignore_index=True)}


def ladder_bin_table(lad, include_legacy=False):
    """Wide table of per-bin phi. S1-S4 share bins; the legacy rows (if included)
    have their own edges, so they appear on separate rows."""
    b = lad['bins']
    if not include_legacy:
        b = b[~b['step'].str.startswith('S0')]
    return b.pivot_table(index=['mid_log10_freq', 'bin_log10_freq'],
                         columns='step', values='phi').sort_index()


def ladder_freq_table(lad, n_cuts=8):
    """Per-bin phi of every step, including legacy, on a common grid of frequency ranges
    (bins assigned by their mid-point). Use this to compare legacy with S1-S4."""
    b = lad['bins'].copy()
    b['freq_range'] = pd.cut(b['mid_log10_freq'], n_cuts)
    return b.pivot_table(index='freq_range', columns='step', values='phi',
                         observed=True).round(3)


def plot_ladder_bins(lad, ax=None, logy=True):
    """phi against bin mid log10 frequency, one line per step."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    for step, g in lad['bins'].groupby('step', sort=False):
        g = g.sort_values('mid_log10_freq')
        ax.plot(g['mid_log10_freq'], g['phi'], marker='o', label=step)
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('log10 frequency (bin mid-point)')
    ax.set_ylabel('phi (size)')
    ax.legend(fontsize=8)
    return ax


# -----------------------------------------------------------------------------
# Shared: objective inputs, profile CIs
# -----------------------------------------------------------------------------

def _filtered_blocks(nb, df, count_col='count', lib_col='lib', depth_col='D_t', c_col='c',
                     min_detect=2, min_freq_mult=5.0, cutoff_frac=None, truncate=True):
    """Detected rows of retained clones, filtered exactly as fit_nb_dispersion_cr does."""
    cutoff_frac = nb.CUTOFF_FRAC if cutoff_frac is None else cutoff_frac
    x, L, Dd, starts, n_per, c = nb._sparse_blocks(
        df, count_col, lib_col, depth_col, min_detect, c_col if truncate else None)
    sx = np.add.reduceat(x, starts)
    p_hat = sx / np.add.reduceat(L, starts)
    p_depth = sx / np.add.reduceat(Dd, starts)
    keep = p_depth >= min_freq_mult * cutoff_frac
    obs = np.repeat(keep, n_per)
    return dict(x=x[obs], L=L[obs], c=None if c is None else c[obs],
                n_per=n_per[keep], lam0=p_hat[keep], p_depth=p_depth[keep])


def _subset(blk, clone_mask):
    o = np.repeat(clone_mask, blk['n_per'])
    npb = blk['n_per'][clone_mask]
    return dict(x=blk['x'][o], L=blk['L'][o],
                c=None if blk['c'] is None else blk['c'][o],
                n_per=npb, lam0=blk['lam0'][clone_mask],
                starts=np.concatenate([[0], np.cumsum(npb)[:-1]]))


def cr_inputs(nb, df, truncate=True, **filter_kwargs):
    """Arrays the CR objective needs, built exactly as fit_nb_dispersion_cr builds them."""
    keys = ('count_col', 'lib_col', 'depth_col', 'c_col', 'min_detect',
            'min_freq_mult', 'cutoff_frac')
    blk = _filtered_blocks(nb, df, truncate=truncate,
                           **{k: v for k, v in filter_kwargs.items() if k in keys})
    return _subset(blk, np.ones(len(blk['n_per']), bool))


def cr_objective(nb, inp, phi, resolve_mean=True, cox_reid=True):
    """Negative CR-adjusted profile log-likelihood at phi (sum over the input's clones)."""
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        return nb._nb_neg_loglik_cr(np.log(phi), inp['x'], inp['L'], inp['starts'],
                                    inp['n_per'], inp['lam0'], resolve_mean, inp['c'],
                                    cox_reid)


def profile_ci(nb, inp, phi_hat, level=0.95, span=3.0, **obj_kwargs):
    """Profile-likelihood CI for phi: {phi : objective(phi) <= objective(phi_hat) + chi2/2}.

    Searches up to a factor exp(span) either side of phi_hat; returns NaN for an
    endpoint not reached within that range.
    """
    crit = stats.chi2.ppf(level, 1) / 2
    f0 = cr_objective(nb, inp, phi_hat, **obj_kwargs)
    g = lambda lp: cr_objective(nb, inp, np.exp(lp), **obj_kwargs) - f0 - crit
    lp0 = np.log(phi_hat)
    out = []
    for edge in (lp0 - span, lp0 + span):
        try:
            out.append(float(np.exp(brentq(g, *sorted((edge, lp0)), xtol=1e-3))))
        except ValueError:
            out.append(np.nan)
    return tuple(out)


# -----------------------------------------------------------------------------
# Experiment 2: per-donor fits
# -----------------------------------------------------------------------------

def per_donor(nb, df, n_bins=3, level=0.95, donor_col='patient', **fit_kwargs):
    """Truncated CR fit per donor, profile CIs, and a shared-vs-per-donor phi LRT.

    The objective is a sum over clones and every clone belongs to one donor, so the
    pooled objective at any phi is the sum of the donor objectives. That gives
        LR = 2 * sum_d [obj_d(phi_pooled) - obj_d(phi_d)]  ~  chi2(n_donors - 1)
    under a shared phi (approximately, since the objective is CR-adjusted).
    With many clones per donor the LRT detects even small differences, so read the
    spread of phi_d and the CIs as the effect size, and the p-value only as a guide.
    """
    flags = dict(cox_reid=True, resolve_mean=True, truncate=True)
    flags.update({k: v for k, v in fit_kwargs.items() if k in flags})
    other = {k: v for k, v in fit_kwargs.items() if k not in flags}
    obj_kw = dict(resolve_mean=flags['resolve_mean'], cox_reid=flags['cox_reid'])

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        pooled = nb.fit_nb_dispersion_cr(df, n_bins=n_bins, **flags, **other)
    phi_pool = pooled['phi']

    rows, bins = [], []
    for dnr, sub in df.groupby(donor_col):
        t0 = time.time()
        try:
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                r = nb.fit_nb_dispersion_cr(sub, n_bins=n_bins, **flags, **other)
        except ValueError as e:
            rows.append({donor_col: dnr, 'error': str(e)})
            continue
        inp = cr_inputs(nb, sub, truncate=flags['truncate'], **other)
        lo, hi = profile_ci(nb, inp, r['phi'], level=level, **obj_kw)
        rows.append({donor_col: dnr, 'phi': r['phi'], 'ci_lo': lo, 'ci_hi': hi,
                     'at_bound': r['phi_at_bound'], 'n_clones': r['n_clones'],
                     'n_obs': r['n_obs'],
                     'obj_own': cr_objective(nb, inp, r['phi'], **obj_kw),
                     'obj_pooled': cr_objective(nb, inp, phi_pool, **obj_kw),
                     'seconds': round(time.time() - t0, 1)})
        bins.append(r['bins'].assign(**{donor_col: dnr}))

    tab = pd.DataFrame(rows)
    ok = tab['phi'].notna() if 'phi' in tab else pd.Series(False, index=tab.index)
    lr = 2 * (tab.loc[ok, 'obj_pooled'] - tab.loc[ok, 'obj_own']).sum()
    dof = int(ok.sum()) - 1
    inp_all = cr_inputs(nb, df, truncate=flags['truncate'], **other)
    pooled_ci = profile_ci(nb, inp_all, phi_pool, level=level, **obj_kw)
    return {'per_donor': tab,
            'bins': pd.concat(bins, ignore_index=True) if bins else pd.DataFrame(),
            'pooled': {'phi': phi_pool, 'ci': pooled_ci, 'n_clones': pooled['n_clones']},
            'lrt': {'LR': float(lr), 'df': dof,
                    'p': float(stats.chi2.sf(lr, dof)) if dof > 0 else np.nan},
            'phi_spread': {'min': tab.loc[ok, 'phi'].min(), 'max': tab.loc[ok, 'phi'].max(),
                           'cv': tab.loc[ok, 'phi'].std() / tab.loc[ok, 'phi'].mean()}}


# -----------------------------------------------------------------------------
# Diagnostics: per-bin profile curves
# -----------------------------------------------------------------------------

def bin_inputs(nb, df, n_bins=6, **filter_kwargs):
    """Per-bin objective inputs, binned exactly as fit_nb_dispersion_cr bins them
    (quantiles of log10 p_depth over retained clones). Always carries c, so each bin
    can be evaluated truncated or not."""
    keys = ('count_col', 'lib_col', 'depth_col', 'c_col', 'min_detect',
            'min_freq_mult', 'cutoff_frac')
    blk = _filtered_blocks(nb, df, truncate=True,
                           **{k: v for k, v in filter_kwargs.items() if k in keys})
    lp = np.log10(blk['p_depth'])
    edges = np.unique(np.quantile(lp, np.linspace(0, 1, n_bins + 1)))
    b_of = np.clip(np.digitize(lp, edges[1:-1]), 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        inp = _subset(blk, b_of == b)
        inp['label'] = f"[{edges[b]:.2f},{edges[b + 1]:.2f})"
        out.append(inp)
    return out


def bin_summary(bins_in):
    """What each bin looks like: clones, share with exactly 2 detections, mu/c."""
    rows = []
    for bi in bins_in:
        mu = np.repeat(bi['lam0'], bi['n_per']) * bi['L']
        rows.append({'bin': bi['label'], 'n_clones': len(bi['n_per']),
                     'frac_ndet_2': float((bi['n_per'] == 2).mean()),
                     'median_ndet': float(np.median(bi['n_per'])),
                     'median_mu_over_c': float(np.median(mu / bi['c']))})
    return pd.DataFrame(rows)


def profile_curves(nb, df, n_bins=6, which=None, phi_grid=None,
                   resolve_mean=True, cox_reid=True, **filter_kwargs):
    """Objective over a phi grid per bin, untruncated and truncated.

    which    : indices of bins to evaluate (default: all).
    phi_grid : phi values (default: 40 log-spaced points in [0.01, 20]).
    Returns (curves, summary): curves in long format with 'rel' = objective minus its
    minimum within (bin, truncate), plus a 'finite' flag; summary from bin_summary.
    Look for non-finite values (numerical trouble) and for curves that keep falling
    towards the smallest phi (the likelihood itself preferring phi -> 0).
    """
    bins_in = bin_inputs(nb, df, n_bins=n_bins, **filter_kwargs)
    phi_grid = np.exp(np.linspace(np.log(0.01), np.log(20), 40)) if phi_grid is None else phi_grid
    idx = range(len(bins_in)) if which is None else which
    rows = []
    for i in idx:
        bi = bins_in[i]
        for tr in (False, True):
            inp = dict(bi, c=bi['c'] if tr else None)
            for phi in phi_grid:
                f = cr_objective(nb, inp, phi, resolve_mean=resolve_mean, cox_reid=cox_reid)
                rows.append({'bin': bi['label'], 'truncate': tr, 'phi': float(phi),
                             'obj': float(f)})
    cur = pd.DataFrame(rows)
    cur['finite'] = np.isfinite(cur['obj'])
    cur['rel'] = cur['obj'] - cur.where(cur['finite']).groupby(
        [cur['bin'], cur['truncate']])['obj'].transform('min')
    return cur, bin_summary([bins_in[i] for i in idx])


def plot_profile_curves(curves, ax=None, ylim=50):
    """Relative objective against phi, one line per (bin, truncate)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(7, 4.5))[1]
    for (b, tr), g in curves.groupby(['bin', 'truncate']):
        ax.plot(g['phi'], g['rel'], ls='-' if tr else '--',
                label=f"{b} {'trunc' if tr else 'untrunc'}")
    ax.set_xscale('log')
    ax.set_ylim(-1, ylim)
    ax.set_xlabel('phi'); ax.set_ylabel('objective - min')
    ax.legend(fontsize=7)
    return ax


# -----------------------------------------------------------------------------
# Diagnostics: simulation on the real designs
# -----------------------------------------------------------------------------

def lambda_pool(df, count_col='count', lib_col='lib'):
    """Abundance pool for simulation: pooled ratio of every detected clone, no filters,
    so it reaches below the frequency floor. Overstates lambda for often-missed clones,
    which makes simulated data sit slightly further above the threshold than real data."""
    g = df[df[count_col] > 0].groupby(['patient', 'clono'])
    return (g[count_col].sum() / g[lib_col].sum()).to_numpy()


def simulate_like_real(df, lam_pool, phi, seed=0, donor_col='patient'):
    """Neutral NB counts with constant phi on each donor's real design (days, D_t, lib, c),
    with the real number of clones per donor; only detected rows (count >= c) kept."""
    rng = np.random.default_rng(seed)
    samp = df.groupby([donor_col, 'day'], as_index=False)[['D_t', 'lib', 'c']].first()
    out = []
    for pat, s in samp.groupby(donor_col):
        n_cl = df.loc[df[donor_col] == pat, 'clono'].nunique()
        lam = rng.choice(lam_pool, size=n_cl, replace=True)
        mu = lam[:, None] * s['lib'].to_numpy()[None, :]
        x = rng.negative_binomial(phi, phi / (phi + mu))
        ci, ti = np.nonzero(x >= s['c'].to_numpy()[None, :])
        out.append(pd.DataFrame({donor_col: pat, 'clono': ci,
                                 'day': s['day'].to_numpy()[ti], 'count': x[ci, ti],
                                 'D_t': s['D_t'].to_numpy()[ti],
                                 'lib': s['lib'].to_numpy()[ti],
                                 'c': s['c'].to_numpy()[ti]}))
    return pd.concat(out, ignore_index=True)


# -----------------------------------------------------------------------------
# Reporting: figure (.png) and tables (.csv)
# -----------------------------------------------------------------------------

def add_legacy(lad, phi0, bins0, n_clones=None):
    """Insert a separately computed legacy fit (from legacy_fit) into a ladder result
    as step 'S0 legacy', e.g. when S0 was run once and the new steps run separately."""
    g = lad['global']
    g = g[g['step'] != 'S0 legacy']
    s0 = pd.DataFrame([{'step': 'S0 legacy', 'phi': phi0, 'n_clones': n_clones,
                        'seconds': np.nan, 'warnings': ''}])
    g = pd.concat([s0, g.drop(columns='change_vs_prev', errors='ignore')], ignore_index=True)
    g['change_vs_prev'] = g['phi'].pct_change()
    b = lad['bins'][lad['bins']['step'] != 'S0 legacy']
    b = pd.concat([bins0.assign(step='S0 legacy'), b], ignore_index=True)
    return {'global': g, 'bins': b}


def ladder_tables(lad, n_cuts=8):
    """The tables of a ladder result, ready to save or display.

    global       : one row per step (phi, % change vs previous step, clones, time).
    bins_long    : every bin of every step (long format).
    bins_S1S4    : per-bin phi of the new steps, which share bins (wide).
    bins_by_freq : every step, legacy included, on a common grid of frequency ranges.
    """
    g = lad['global'].copy()
    g['change_vs_prev_%'] = (100 * g['change_vs_prev']).round(1)
    g = g[['step', 'phi', 'change_vs_prev_%', 'n_clones', 'seconds']].round({'phi': 3})
    bl = lad['bins'][['step', 'bin_log10_freq', 'mid_log10_freq', 'n_clones', 'phi']].copy()
    out = {'global': g, 'bins_long': bl.round({'phi': 3, 'mid_log10_freq': 3})}
    new = lad['bins'][~lad['bins']['step'].str.startswith('S0')]
    if len(new):
        out['bins_S1S4'] = ladder_bin_table(lad).round(3)
    out['bins_by_freq'] = ladder_freq_table(lad, n_cuts=n_cuts)
    return out


def save_ladder_report(lad, path_prefix='ladder', title=None, n_cuts=8, logy=True, dpi=150):
    """Save the ladder comparison as one .png figure and .csv tables.

    Figure: left, global phi per step (with % change vs previous step); right, per-bin
    phi against bin mid-point log10 frequency, one line per step.
    Files: <prefix>.png, <prefix>_global.csv, <prefix>_bins_long.csv,
           <prefix>_bins_S1S4.csv, <prefix>_bins_by_freq.csv.
    Returns dict with 'fig', 'tables' and 'files'.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tabs = ladder_tables(lad, n_cuts=n_cuts)
    g = lad['global'].reset_index(drop=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8),
                                 gridspec_kw={'width_ratios': [1, 1.4]})
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    steps = list(g['step'])
    a1.bar(range(len(g)), g['phi'], color=[colors[i % len(colors)] for i in range(len(g))])
    for i, r in g.iterrows():
        lab = f"{r['phi']:.2f}"
        if pd.notna(r['change_vs_prev']):
            lab += f"\n({100 * r['change_vs_prev']:+.0f}%)"
        a1.text(i, r['phi'], lab, ha='center', va='bottom', fontsize=8)
    a1.set_xticks(range(len(g)))
    a1.set_xticklabels([s.replace(' (new)', '').replace(', no corrections', '')
                        for s in steps], rotation=30, ha='right', fontsize=8)
    a1.set_ylabel('global phi (size)')
    a1.set_ylim(0, 1.25 * np.nanmax(g['phi']))
    a1.set_title('Global phi per step', fontsize=10)

    for i, step in enumerate(steps):
        b = lad['bins'][lad['bins']['step'] == step].sort_values('mid_log10_freq')
        if len(b):
            a2.plot(b['mid_log10_freq'], b['phi'], marker='o', label=step,
                    color=colors[i % len(colors)],
                    ls='--' if step.startswith('S0') else '-')
    if logy:
        a2.set_yscale('log')
    a2.set_xlabel('log10 frequency (bin mid-point)')
    a2.set_ylabel('phi (size)')
    a2.set_title('Per-bin phi', fontsize=10)
    a2.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    files = {'figure': f'{prefix}.png'}
    fig.savefig(files['figure'], dpi=dpi, bbox_inches='tight')
    for name, t in tabs.items():
        files[name] = f'{prefix}_{name}.csv'
        t.to_csv(files[name], index=name in ('bins_S1S4', 'bins_by_freq'))
    return {'fig': fig, 'tables': tabs, 'files': files}
