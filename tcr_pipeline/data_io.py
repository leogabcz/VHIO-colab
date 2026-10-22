"""
data_io.py

Utilities for converting between the raw nested-dict input format and the
canonical long-format DataFrame used throughout the pipeline.

Canonical schema
----------------
donor      : str   — donor or patient identifier
day        : int   — day of sampling (numeric, extracted from keys like "d0", "d14")
clonotype  : str   — clonotype identifier (anonymized)
count      : int   — raw count

(donor, day, clonotype) is the logical primary key.
"""

import re
import pandas as pd
from typing import Optional
import os
import numpy as np
from pathlib import Path

ROOTS = {
    "donors":   Path("../data/donors_clono/"),
    "patients": Path("../data/patients_clono/"),    # adjust if different
}
# each root contains: counts/<subject>.csv, sequences/<subject>_sequences.csv,
#                     sample_sizes.csv, subjects.txt

_COUNT_DTYPES = {"day": "int16", "clono": "int32", "count": "int32"}

def _sample_sizes(group):
    return pd.read_csv(_root(group) / "sample_sizes.csv",
                       dtype={"patient": str, "day": "int16", "D_t": "int64"})

def _root(group) -> Path:
    if group not in ROOTS:
        raise KeyError(f"unknown group {group!r}; use one of {list(ROOTS)}")
    return ROOTS[group]

def write_subject_list(group):
    """Write subjects.txt (one name per line) from the files in counts/. Only creates/overwrites that file."""
    root = _root(group)
    names = sorted(p.stem for p in (root / "counts").glob("*.csv"))
    if not names:
        raise FileNotFoundError(f"no count files in {root / 'counts'}")
    (root / "subjects.txt").write_text("\n".join(names) + "\n")
    return names


def list_subjects(group: str):
    """Read the names from subjects.txt; checks that every listed subject has a counts file."""
    root = _root(group)
    txt = root / "subjects.txt"
    if not txt.exists():
        raise FileNotFoundError(f"{txt} not found; run write_subject_list({group!r}) first")
    names = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
    missing = [n for n in names if not (root / "counts" / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"listed in {txt.name} but without a counts file: {missing}")
    return names

def load_subject(group: str, subject: str,
                 min_count: int = 1,
                 days=None,
                 with_D_t: bool = False,
                 with_freq: bool = False,
                 R_labels: bool = False,
                 add_patient: bool = True,
                 add_group: bool = False) -> pd.DataFrame:
    """
    One donor or patient, long format: [group], [patient], day, clono, count [, D_t] [, freq].

    min_count    keep rows with count >= min_count (applied after reading)
    days         iterable of days to keep (None = all)
    with_D_t     add the library size of each row's sample
    with_freq    add count / D_t
    R_labels     make `clono` the string 'clono_<id>' (heavier; one subject at a time)
    add_patient  add a `patient` column (the subject name)
    add_group    add a `group` column ('donors' / 'patients')
    """
    root = _root(group)
    path = root / "counts" / f"{subject}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} (available: {list_subjects(group)})")

    d = pd.read_csv(path, dtype=_COUNT_DTYPES)
    if days is not None:
        d = d[d["day"].isin(list(days))]
    if min_count > 1:
        d = d[d["count"] >= min_count]
    d = d.reset_index(drop=True)

    if with_D_t or with_freq:
        s = _sample_sizes(group)
        s = s[s["patient"] == subject].drop(columns="patient")
        d = d.merge(s, on="day", how="left", validate="many_to_one")
        if d["D_t"].isna().any():
            missing = sorted(d.loc[d["D_t"].isna(), "day"].unique().tolist())
            raise ValueError(f"{group}/{subject}: no library size for days {missing}")
        if with_freq:
            d["freq"] = d["count"] / d["D_t"]
        if not with_D_t:
            d = d.drop(columns="D_t")

    if R_labels:
        d["clono"] = "clono_" + d["clono"].astype(str)
    if add_patient:
        d.insert(0, "patient", pd.Categorical([subject] * len(d), categories=[subject]))
    if add_group:
        d.insert(0, "group", pd.Categorical([group] * len(d), categories=[group]))
    return d

def load_group(group,
               subjects=None,
               min_count=1,
               days=None,
               with_D_t=True,
               with_freq=True,
               add_group=False,
               compact=True):
    """
    All subjects of a group ('donors' or 'patients') in one long DataFrame.

    subjects   list of names to load (None = every name in subjects.txt)
    compact    store D_t as int32 and freq as float32 (halves the memory of those two columns)
    Other arguments are passed to load_subject.

    Note: `clono` ids are local to each subject. Always index on (patient, clono) together.
    """
    names = list_subjects(group) if subjects is None else list(subjects)
    parts = []
    for s in names:
        parts.append(load_subject(group, s, min_count=min_count, days=days,
                                  with_D_t=with_D_t, with_freq=with_freq,
                                  add_patient=True, add_group=add_group))
    d = pd.concat(parts, ignore_index=True)

    # concatenating categoricals with different categories turns them into object columns
    d["patient"] = pd.Categorical(d["patient"].astype(str), categories=names)
    if add_group:
        d["group"] = pd.Categorical([group] * len(d), categories=[group])
    if compact:
        if "D_t" in d.columns:
            d["D_t"] = d["D_t"].astype("int32")
        if "freq" in d.columns:
            d["freq"] = d["freq"].astype("float32")
    return d

def load_sequences(group: str, subject: str, ids=None) -> pd.DataFrame:
    """clono -> CDR3_sequence, TRBV, TRBJ for one subject (confidential; load only when needed)."""
    m = pd.read_csv(_root(group) / "sequences" / f"{subject}_sequences.csv", dtype={"clono": "int32"})
    return m if ids is None else m[m["clono"].isin(list(ids))]

def sample_qc(df, count_col='count', min_fails_to_drop=2):
    """Per-sample QC metrics, each relative to the same person's other samples.

    Columns *_rel = value / median of the person's OTHER samples.
    fail_* = individual criteria; n_fails = how many failed;
    drop = n_fails >= min_fails_to_drop; review = exactly one failure.
    """
    det = df[df[count_col] > 0]
    rows = []
    for pat, g in det.groupby('patient'):
        days = sorted(g['day'].unique())
        freq = (g.assign(f=g[count_col] / g['D_t'])
                 .pivot_table(index='clono', columns='day', values='f', fill_value=0))
        for day in days:
            gd = g[g['day'] == day]
            rows.append({'patient': pat, 'day': day, 'D_t': gd['D_t'].iloc[0],
                         'n_clones': len(gd),
                         'reads_per_clone': gd[count_col].sum() / len(gd),
                         'top_share': gd[count_col].max() / gd[count_col].sum()})
    q = pd.DataFrame(rows)
    for col in ['n_clones', 'reads_per_clone', 'top_share']:
        q[col + '_rel'] = [
            v / q.loc[(q['patient'] == p) & (q['day'] != dd), col].median()
            for p, dd, v in zip(q['patient'], q['day'], q[col])]

    q['fail_clones']    = q['n_clones_rel'] < 0.2
    q['fail_top']       = q['top_share_rel'] > 5
    q['fail_rpc']       = q['reads_per_clone_rel'] > 5
    fail_cols = ['fail_clones', 'fail_top', 'fail_rpc']
    q['n_fails'] = q[fail_cols].sum(axis=1)
    q['drop'] = q['n_fails'] >= min_fails_to_drop
    q['review'] = q['n_fails'] == 1
    return q.round(4)


def drop_failed_samples(df, qc):
    """Remove the samples marked drop=True in a sample_qc table."""
    bad = qc.loc[qc['drop'], ['patient', 'day']]
    key = pd.MultiIndex.from_frame(df[['patient', 'day']])
    return df[~key.isin(pd.MultiIndex.from_frame(bad))].copy()