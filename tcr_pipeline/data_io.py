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
import rdata
from typing import Optional
import os
import numpy as np

def load_rdata(data_dir: str) -> dict:
    """
    Load the raw rdata file into a nested python dictionary
    
    Parameters
    ----------
    data_dir : str
        Location of the Rdata file
    
    Retrurns
    --------
    converted : dict
        Nested dictionary of the data following the canonical schema
    """
    
    parsed = rdata.parser.parse_file(data_dir)
    converted = rdata.conversion.convert(parsed)
    return converted

def parse_day(day_str: str) -> int:
    """
    Extract the numeric day from a string key like 'd0', 'd14', 'd-9'.

    Parameters
    ----------
    day_str : str
        Timepoint key, expected format 'd<integer>' (e.g. 'd0', 'd14', 'd-9').

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If the string does not match the expected pattern.
    """
    match = re.fullmatch(r"d(-?\d+)", day_str)
    if match is None:
        raise ValueError(
            f"Timepoint key '{day_str}' does not match expected format 'd<integer>'. "
            "Examples of valid keys: 'd0', 'd14', 'd-9'."
        )
    return int(match.group(1))


def nested_dict_to_long(
    data: dict,
    clonotype_col: str = "clono",
    count_col: str = "count",
) -> pd.DataFrame:
    """
    Convert the raw nested-dict input into the canonical long-format DataFrame.

    Parameters
    ----------
    data : dict
        Nested dict with structure:
            { donor: { day_str: pd.DataFrame(clonotype_col, count_col, ...) } }
        Extra columns in the per-timepoint DataFrames are dropped.
    clonotype_col : str
        Name of the clonotype identifier column in the per-timepoint DataFrames.
    count_col : str
        Name of the raw count column.

    Returns
    -------
    pd.DataFrame
        Columns: donor, day (int), clonotype, count.
        Sorted by donor -> day -> clonotype.
    """
    frames = []
    for donor, timepoints in data.items():
        for day_str, df in timepoints.items():
            day = parse_day(day_str)
            frame = df.copy()
            frame = frame.rename(columns={clonotype_col: "clono", count_col: "count"})
            frame["donor"] = donor
            frame["day"] = day
            frames.append(frame)

    if not frames:
        raise ValueError("Input data dict is empty.")

    long = pd.concat(frames, ignore_index=True)
    long["count"] = long["count"].astype(int)

    long = long[["donor", "day", "clono", "count"]]
    long = long.rename(columns={"donor": "patient", "clono": "clono"})

    return long.sort_values(["patient", "day", "clono"]).reset_index(drop=True)


def long_df_to_wide(
    data: pd.DataFrame,
    values: str = "count",
) -> dict:
    """
    Convert the long format data frame a dict of wide-format DataFrames,
    one per donor.

    Wide format: rows = clonotypes, columns = days (as integers), values = counts.
    Clonotypes absent at a given timepoint are filled with `fill_value`.

    Parameters
    ----------
    data : pd.df
        Long-form canonical dataframe 
    values : str
        Values which will be shown in the wide format (e.g. count, norm_counts)

    
    Returns
    -------
    dict
        { donor: pd.DataFrame }
        Each DataFrame has clonotype as the index, day integers as columns,
        sorted in ascending day order.
    """
    
    result = {}
    for donor, donor_df in data.groupby("patient"):
        wide = (
            donor_df
            .pivot(index="clono", columns="day", values=values)
        )
        wide.columns.name = None
        wide = wide[sorted(wide.columns)]
        result[donor] = wide

    return result


def long_to_nested_dict(
    df: pd.DataFrame,
    extra_cols: Optional[list] = None,
) -> dict:
    """
    Convert the canonical long-format DataFrame back to a nested dict.
    Useful if downstream tools expect the original structure.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least: donor, day (int), clonotype, count.
    extra_cols : list[str] | None
        Additional columns to include in the per-timepoint DataFrames.
        If None, all columns beyond [donor, day] are included.

    Returns
    -------
    dict
        { donor: { 'd<day>': pd.DataFrame } }
    """
    base_cols = ["clono", "count"]
    if extra_cols is not None:
        keep_cols = base_cols + [c for c in extra_cols if c not in base_cols]
    else:
        keep_cols = [c for c in df.columns if c not in ("patient", "day")]

    result = {}
    for donor, donor_df in df.groupby("patient"):
        result[donor] = {}
        for day, day_df in donor_df.groupby("day"):
            result[donor][f"d{int(day)}"] = day_df[keep_cols].reset_index(drop=True)

    return result

def save_noise_model(model: dict, path: str) -> None:
    """
    Save a noise model dict to a single JSON file.

    Works with any noise model produced by compute_slope_noise_model,
    regardless of fit_distribution ('normal' or 't') — all scalar
    parameters present per donor (e.g. mu, sigma, df) are saved.

    Parameters
    ----------
    model : dict
        Output of compute_slope_noise_model.
    path : str
        Full output path, e.g. 'results/noise_model_clr.json'.
        Directory is created if it does not exist.

    Returns
    -------
    None

    Examples
    --------
    save_noise_model(model, "results/noise_model_clr.json")
    """
    import json

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    def _params(vals):
        return {
            k: float(v)
            for k, v in vals.items()
            if k not in ("rates", "clonotype_rates")
        }

    def _clonotype_rates(vals):
        return {
            clono: rates.tolist()
            for clono, rates in vals.get("clonotype_rates", {}).items()
        }

    payload = {
        "norm_col":         model["norm_col"],
        "fit_distribution": model.get("fit_distribution", "normal"),
        "pooled":           _params(model["pooled"]),
        "per_donor": {
            donor: {
                **_params(vals),
                "rates": vals["rates"].tolist(),
                "clonotype_rates": _clonotype_rates(vals),
            }
            for donor, vals in model["per_donor"].items()
        },
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved noise model to {path}")


def load_noise_model(path: str) -> dict:
    """
    Load a noise model previously saved with save_noise_model.

    Parameters
    ----------
    path : str
        Path to the JSON file written by save_noise_model.

    Returns
    -------
    dict
        Same structure as the output of compute_slope_noise_model:
            per_donor        : { donor: { mu, sigma, [df], rates, clonotype_rates } }
            pooled           : { mu, sigma, [df], rates }
            norm_col         : str
            fit_distribution : str

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.

    Examples
    --------
    model = load_noise_model("results/noise_model_clr.json")
    model["pooled"]["mu"]
    model["per_donor"]["HD106"]["rates"]
    """
    import json

    if not os.path.exists(path):
        raise FileNotFoundError(f"Noise model file not found: {path}")

    with open(path) as f:
        payload = json.load(f)

    per_donor = {
        donor: {**{
                    k: v
                    for k, v in vals.items()
                    if k not in ("rates", "clonotype_rates")
                },
                "rates": np.array(vals["rates"]),
                "clonotype_rates": {
                    clono: np.array(rates)
                    for clono, rates in vals.get("clonotype_rates", {}).items()
                }}
        for donor, vals in payload["per_donor"].items()
    }

    pooled_rates = np.concatenate([v["rates"] for v in per_donor.values()])

    return {
        "per_donor": per_donor,
        "pooled": {**payload["pooled"], "rates": pooled_rates},
        "norm_col":         payload["norm_col"],
        "fit_distribution": payload.get("fit_distribution", "normal"),
    }
