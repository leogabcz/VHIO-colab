import pandas as pd

def count_nan_trajectories(
    df,
    fmt: str = 'long',
    n_timepoints: int = None,
) -> pd.DataFrame:
    """
    For each clonotype, count missing timepoints and flag whether any are missing.

    Parameters
    ----------
    df : pd.DataFrame or dict[str, pd.DataFrame]
        Long-format: single DataFrame with columns patient, day, clono, count.
        Wide-format: dict of {patient: DataFrame} as returned by long_df_to_wide,
                     where rows=clonotypes, columns=days, NaN=missing.
    fmt : str
        'long' or 'wide'.
    n_timepoints : int, optional
        Long fmt only. Total timepoints per patient. If None, inferred per patient
        from df['day'].nunique(). Pass explicitly if the DataFrame may be
        pre-filtered and some timepoints are globally absent.

    Returns
    -------
    pd.DataFrame
        One row per (patient, clono) with columns:
            patient, clono, n_timepoints, n_nan, has_nan
    """
    if fmt == 'wide':
        frames = []
        for patient, wide in df.items():
            day_cols = [c for c in wide.columns]
            result = pd.DataFrame({'patient': patient, 'clono': wide.index})
            result['n_timepoints'] = len(day_cols)
            result['n_nan'] = wide.isna().sum(axis=1).values
            result['has_nan'] = result['n_nan'] > 0
            frames.append(result)
        return pd.concat(frames, ignore_index=True)[['patient', 'clono', 'n_timepoints', 'n_nan', 'has_nan']]

    if fmt == 'long':
        if n_timepoints is None:
            n_timepoints_per_patient = (
                df.groupby('patient')['day']
                .nunique()
                .rename('n_timepoints')
            )
        observed = (
            df.groupby(['patient', 'clono'])['day']
            .nunique()
            .rename('n_observed')
            .reset_index()
        )
        if n_timepoints is None:
            observed = observed.join(n_timepoints_per_patient, on='patient')
        else:
            observed['n_timepoints'] = n_timepoints
        observed['n_nan'] = observed['n_timepoints'] - observed['n_observed']
        observed['has_nan'] = observed['n_nan'] > 0
        return observed[['patient', 'clono', 'n_timepoints', 'n_nan', 'has_nan']]

    raise ValueError(f"fmt must be 'long' or 'wide', got '{fmt}'")

def summarize_nan_trajectories(
    df: pd.DataFrame,
    patient_col: str = "patient",
    clono_col: str = "clono",
    day_col: str = "day",
    count_col: str = "count",
) -> pd.DataFrame:
    """
    Summarize NaN trajectory counts across all patients in a long-format DataFrame.

    For each patient, reports the number of timepoints observed, total clonotype
    count, and the number/percentage of clonotypes missing 1, 2, 3, ... timepoints.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns: patient, day, clono, count.
        Can be the full healthy_df or any multi-patient long-format slice.
        NaNs in count_col are treated as missing observations.
    patient_col, clono_col, day_col, count_col : str
        Column name overrides.

    Returns
    -------
    pd.DataFrame
        One row per (patient, n_missing) combination, with columns:
            patient         : donor/patient identifier
            n_timepoints    : total timepoints for that patient (days observed)
            n_clonotypes    : total clonotypes for that patient
            n_missing       : number of missing timepoints (0, 1, 2, ...)
            n_clono         : number of clonotypes with exactly that many missing
            pct_clono       : percentage of clonotypes (out of n_clonotypes)
    """
    records = []

    for patient, pdf in df.groupby(patient_col):
        days = pdf[day_col].nunique()
        n_clono_total = pdf[clono_col].nunique()

        # Pivot to clono x day to detect structural absences as well as NaN counts
        pivot = pdf.pivot_table(
            index=clono_col,
            columns=day_col,
            values=count_col,
            aggfunc="first",   # one count per (clono, day) expected
        )

        # Each row: count NaN cells (absent or explicitly NaN)
        nan_counts = pivot.isna().sum(axis=1)  # Series: clono -> n_missing

        # Distribution over n_missing values
        dist = nan_counts.value_counts().sort_index()

        for n_missing, n_clono in dist.items():
            records.append({
                patient_col:     patient,
                "n_timepoints":  days,
                "n_clonotypes":  n_clono_total,
                "n_missing":     int(n_missing),
                "n_clono":       int(n_clono),
                "pct_clono":     round(100 * n_clono / n_clono_total, 2),
            })

    return pd.DataFrame(records).rename(columns={patient_col: patient_col})

def impute_trajectories(
    df: pd.DataFrame,
    values: str = 'count',
    max_nan_fraction: float = 0.5,
    internal_gap_limit: int = 2,
    pseudocount: float = None,
) -> pd.DataFrame:
    """
    Impute missing timepoints in a long-format TCR DataFrame.

    Strategy (applied per patient x clonotype):
        1. Exclude clonotypes where n_nan / n_timepoints > max_nan_fraction.
        2. Internal gaps of <= internal_gap_limit consecutive missing timepoints:
           linear interpolation between flanking observed values.
        3. All other missing timepoints (edge gaps, long internal gaps):
           fill with per-sample pseudocount (minimum detected value in that
           patient x day sample for the selected column).
        4. Two boolean columns are added:
             has_interpolated : True if any timepoint was filled by interpolation
             has_pseudocount  : True if any timepoint was filled by pseudocount

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns: patient, day, clono, and at least
        the column specified by `values`.
        Absent rows represent missing timepoints (sparse representation).
    values : str
        Column to impute. Default 'count'. Pass a normalized column (e.g.
        'norm_count', 'clr') to avoid pseudocounts affecting library sizes.
    max_nan_fraction : float
        Clonotypes with proportion of missing timepoints above this threshold
        are excluded entirely. Default 0.5.
    internal_gap_limit : int
        Maximum consecutive missing timepoints filled by interpolation.
        Longer internal gaps and all edge gaps fall back to pseudocount.
        Default 2.
    pseudocount : float, optional
        Value used to fill edge gaps and long internal gaps.
        If None, inferred as the minimum detected value per patient x day
        from the original (pre-exclusion) DataFrame.
        Falls back to per-patient minimum if a sample has no detected values.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with the same columns as input plus:
            has_interpolated : bool, True if any timepoint was interpolated
            has_pseudocount  : bool, True if any timepoint was pseudocount-filled
        All (patient, clono, day) combinations are present; no absent rows remain.
    """
    # preserve original df for pseudocount map — must be computed before
    # exclusion, so days that lose all clonotypes still have a pseudocount
    df_original = df.copy()

    # --- step 1: exclusion ---
    nan_counts = count_nan_trajectories(df, fmt='long')
    nan_counts['nan_fraction'] = nan_counts['n_nan'] / nan_counts['n_timepoints']
    keep = nan_counts.loc[
        nan_counts['nan_fraction'] <= max_nan_fraction, ['patient', 'clono']
    ]
    n_excluded = len(nan_counts) - len(keep)
    if n_excluded > 0:
        print(f"Excluded {n_excluded} clonotypes exceeding max_nan_fraction={max_nan_fraction}")
    df = df.merge(keep, on=['patient', 'clono'], how='inner')

    # --- step 2 & 3: build full grid (per-patient days) ---
    def _patient_grid(g):
        patient = g['patient'].iloc[0]
        days = sorted(g['day'].unique())
        clonos = g['clono'].unique()
        return pd.DataFrame(
            [(patient, c, d) for c in clonos for d in days],
            columns=['patient', 'clono', 'day'],
        )

    full_grid = (
        df.groupby('patient', group_keys=False)
        .apply(_patient_grid)
        .reset_index(drop=True)
    )

    df_full = full_grid.merge(df, on=['patient', 'clono', 'day'], how='left')
    df_full = df_full.sort_values(['patient', 'clono', 'day']).reset_index(drop=True)

    # initialise row-level flags before filling
    df_full['_interp_flag'] = False
    df_full['_pseudo_flag'] = False

    # --- build pseudocount map from original df (pre-exclusion) ---
    if pseudocount is None:
        pseudocount_map = (
            df_original.groupby(['patient', 'day'])[values]
            .min()
            .rename('pseudocount')
            .reset_index()
        )
        df_full = df_full.merge(pseudocount_map, on=['patient', 'day'], how='left')
    else:
        df_full['pseudocount'] = pseudocount

    # --- impute per (patient, clono) ---
    def _impute_group(g: pd.DataFrame) -> pd.DataFrame:
        s = g[values].copy()
        if not s.isna().any():
            return g

        g = g.copy()

        # Identify which NaN positions belong to internal runs short enough
        # to interpolate. Runs longer than internal_gap_limit, and all edge
        # gaps, are left for pseudocount.
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()

        interp_mask = pd.Series(False, index=s.index)
        if first_valid is not None and last_valid is not None:
            fv_loc = s.index.get_loc(first_valid)
            lv_loc = s.index.get_loc(last_valid)

            internal_nan = s.isna().copy()
            internal_nan.iloc[: fv_loc + 1] = False  # exclude leading edge
            internal_nan.iloc[lv_loc:] = False        # exclude trailing edge

            if internal_nan.any():
                run_id = (internal_nan != internal_nan.shift()).cumsum()
                for rid in run_id[internal_nan].unique():
                    run_idx = s.index[run_id == rid]
                    if len(run_idx) <= internal_gap_limit:
                        interp_mask[run_idx] = True

        # Interpolate only the vetted positions (no pandas limit needed)
        s_interp = s.copy()
        if interp_mask.any():
            s_filled = s.interpolate(method='linear', limit_area='inside', limit=None)
            s_interp[interp_mask] = s_filled[interp_mask]

        filled_by_interp = s.isna() & s_interp.notna()
        still_missing = s_interp.isna()

        g[values] = s_interp
        g.loc[still_missing, values] = g.loc[still_missing, 'pseudocount']
        g.loc[filled_by_interp, '_interp_flag'] = True
        g.loc[still_missing, '_pseudo_flag'] = True
        return g

    df_full = (
        df_full
        .groupby(['patient', 'clono'], group_keys=False)
        .apply(_impute_group)
    )

    # safety net: catch any NaNs still remaining after groupby
    still_missing = df_full[values].isna()
    if still_missing.any():
        df_full.loc[still_missing, values] = df_full.loc[still_missing, 'pseudocount']
        df_full.loc[still_missing, '_pseudo_flag'] = True

    # --- promote row-level flags to clonotype level ---
    clono_flags = (
        df_full.groupby(['patient', 'clono'])[['_interp_flag', '_pseudo_flag']]
        .any()
        .rename(columns={
            '_interp_flag': 'has_interpolated',
            '_pseudo_flag': 'has_pseudocount',
        })
        .reset_index()
    )
    df_full = df_full.drop(columns=['_interp_flag', '_pseudo_flag']).merge(
        clono_flags, on=['patient', 'clono'], how='left'
    )

    return df_full.drop(columns='pseudocount').reset_index(drop=True)