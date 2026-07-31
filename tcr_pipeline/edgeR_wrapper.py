"""
edger_wrapper.py

rpy2 bridge to real edgeR, used as a validation baseline for the custom
Python NB GLM / dispersion machinery in nb_glm.py and noise_model.py.
"""
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Optional

edger = importr('edgeR')
base = importr('base')


def _long_to_count_matrix(
    df: pd.DataFrame,
    sample_cols: list,
    count_col: str = 'count',
    clono_col: str = 'clono',
) -> pd.DataFrame:
    """
    Pivot long-format rows into a clonotype x sample count matrix,
    zero-filled, columns ordered/selected by sample_cols.
    """
    wide = df.pivot_table(
        index=clono_col, columns='day', values=count_col,
        fill_value=0, aggfunc='sum',
    )
    return wide.reindex(columns=sample_cols, fill_value=0)


def edger_exact_test(
    df: pd.DataFrame,
    patient: str,
    day_pre,
    day_post,
    count_col: str = 'count',
    clono_col: str = 'clono',
    patient_col: str = 'patient',
    min_mean_count: float = 4.0,
) -> pd.DataFrame:
    """
    Standard edgeR classic pipeline (TMM + qCML + exact test), matching the
    Pavlova et al. methods exactly: filter clonotypes with mean count <4
    across the pair, TMM-normalize, estimate dispersion by quantile-adjusted
    conditional maximum likelihood (estimateCommonDisp/estimateTagwiseDisp
    -- this is real qCML, only valid for a two-group comparison), run the
    exact NB test.

    Use this to validate a single pairwise call against find_non_neutral /
    nb_glm_trajectory_test restricted to the same two timepoints.

    Returns
    -------
    pd.DataFrame, one row per retained clonotype:
        clono, logFC, logCPM, PValue (NOT yet BH-corrected --
        apply your own bh_correct for a fair comparison of the FDR step too)
    """
    sub = df[df[patient_col] == patient]
    mat = _long_to_count_matrix(sub, sample_cols=[day_pre, day_post],
                                 count_col=count_col, clono_col=clono_col)
    mat = mat.loc[mat.mean(axis=1) >= min_mean_count]
    if mat.shape[0] == 0:
        return pd.DataFrame(columns=['clono', 'logFC', 'logCPM', 'PValue'])

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_counts = ro.conversion.py2rpy(mat.astype(int))
    r_counts.rownames = ro.StrVector(mat.index.astype(str))
    r_counts.colnames = ro.StrVector(['pre', 'post'])

    dge = edger.DGEList(counts=r_counts)
    dge = edger.calcNormFactors(dge, method='TMM')
    dge = edger.estimateCommonDisp(dge)
    dge = edger.estimateTagwiseDisp(dge)
    result = edger.exactTest(dge)

    top = edger.topTags(result, n=mat.shape[0], **{'sort.by': 'none'})
    with localconverter(ro.default_converter + pandas2ri.converter):
        top_df = ro.conversion.rpy2py(top.rx2('table'))

    return top_df.reset_index().rename(columns={'index': 'clono'})


def edger_trended_dispersion_single_patient(
    patient_df: pd.DataFrame,
    count_col: str = 'count',
    clono_col: str = 'clono',
    day_col: str = 'day',
) -> pd.DataFrame:
    """
    edgeR trended dispersion using ONE donor's own timepoints as samples.
    Clonotype IDs are only meaningful within a donor (locally anonymized --
    'clono_1' in donor A is unrelated to 'clono_1' in donor B), so a DGEList
    must never mix rows across donors.
    """
    wide = patient_df.pivot_table(
        index=clono_col, columns=day_col, values=count_col,
        fill_value=0, aggfunc='sum',
    )
    if wide.shape[1] < 2:
        raise ValueError("Need at least 2 timepoints to fit a dispersion trend.")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_counts = ro.conversion.py2rpy(wide.astype(int))
    r_counts.rownames = ro.StrVector(wide.index.astype(str))
    r_counts.colnames = ro.StrVector([str(c) for c in wide.columns])

    dge = edger.DGEList(counts=r_counts)
    dge = edger.calcNormFactors(dge, method='TMM')
    design = base.matrix(1, nrow=wide.shape[1], ncol=1)
    dge = edger.estimateDisp(dge, design=design)

    with localconverter(ro.default_converter + pandas2ri.converter):
        avg_logcpm = np.array(dge.rx2('AveLogCPM'))
        trended = np.array(dge.rx2('trended.dispersion'))
        tagwise = np.array(dge.rx2('tagwise.dispersion'))
    common = float(dge.rx2('common.dispersion')[0])

    return pd.DataFrame({
        'clono': wide.index,
        'avg_log_cpm': avg_logcpm,
        'trended_dispersion': trended,
        'tagwise_dispersion': tagwise,
        'common_dispersion': common,
    })


def edger_dispersion_trend_pooled(
    healthy_df: pd.DataFrame,
    count_col: str = 'count',
    clono_col: str = 'clono',
    patient_col: str = 'patient',
    day_col: str = 'day',
    min_timepoints: int = 2,
    lowess_frac: float = 0.3,
) -> dict:
    """
    Fit trended dispersion separately per healthy donor, then pool the
    resulting (avg_log_cpm, dispersion) points across donors and re-smooth.

    This is the dispersion-trend analogue of the pooling risk already
    flagged for the slope null (see fit_stratified_slope_model notes):
    naive concatenation still weights donors by their clonotype count, not
    equally. If donor-count imbalance turns out to matter here too, weight
    each donor's contribution before the pooled loess, or fit the loess
    per-donor and average the resulting curves on a common x-grid instead.
    """
    per_patient_frames = []
    for patient, sub in healthy_df.groupby(patient_col):
        print(f"Computing dispersion of  donor '{patient}'")
        if sub[day_col].nunique() < min_timepoints:
            continue
        pdf = edger_trended_dispersion_single_patient(
            sub, count_col=count_col, clono_col=clono_col, day_col=day_col,
        )
        pdf['patient'] = patient
        per_patient_frames.append(pdf)

    if not per_patient_frames:
        raise ValueError("No healthy donors with enough timepoints to fit a trend.")

    pooled = pd.concat(per_patient_frames, ignore_index=True).sort_values('avg_log_cpm')

    smoothed = lowess(
        np.log(pooled['trended_dispersion'].values),
        pooled['avg_log_cpm'].values,
        frac=lowess_frac, it=3, return_sorted=True,
    )

    return {
        'avg_log_cpm': smoothed[:, 0],
        'dispersion': np.exp(smoothed[:, 1]),
        'common_dispersion': float(pooled['common_dispersion'].mean()),
        'per_patient': pooled,  # diagnostic: check curves are consistent donor-to-donor
    }