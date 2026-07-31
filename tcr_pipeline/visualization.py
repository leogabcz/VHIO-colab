"""
visualization.py
 
Plotting functions for TCR trajectory data.
 
Available functions
-------------------
show_traj : plot a single clonotype trajectory from a wide-format DataFrame
"""
 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from tcr_pipeline.noise_model import _pdf
 
 
def show_traj(
    df: pd.DataFrame,
    clonotype_id: str,
    ax=None,
    color: str = "steelblue",
    label: str = None,
    title: str = None,
) -> None:
    """
    Plot a single clonotype trajectory from a wide-format DataFrame.
    NaN timepoints are marked as red crosses at y=0 to distinguish
    "not detected" from "zero".
 
    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame with clonotypes as index and days as columns.
        As returned by nested_dict_to_wide()[patient].
    clonotype_id : str
        Index label of the clonotype to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created and shown.
    color : str
        Line and marker color. Default 'steelblue'.
    label : str, optional
        Legend label. Defaults to the clonotype_id string.
    title : str, optional
        Plot title. Defaults to the clonotype_id string.
 
    Returns
    -------
    None
 
    Raises
    ------
    ValueError
        If clonotype_id is not found in df.index.
 
    Examples
    --------
    wide = nested_dict_to_wide(data)
    show_traj(wide["DONOR_1"], "clone_A")
    """
    if clonotype_id not in df.index:
        raise ValueError(f"Clonotype '{clonotype_id}' not found in dataset.")
 
    days = list(df.columns)
    row = df.loc[clonotype_id]
 
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
 
    # Plot line with NaNs filled as 0 to maintain continuity
    row_filled = row.fillna(0).values.astype(float)
    ax.plot(
        days, row_filled,
        marker="o", linewidth=1.8, markersize=6,
        color=color, label=label or str(clonotype_id),
    )
 
    # Mark NaN positions with a red cross at y=0
    nan_days = [d for d, is_nan in zip(days, row.isna()) if is_nan]
    if nan_days:
        ax.scatter(
            nan_days, [0] * len(nan_days),
            marker="x", color="red", s=80, zorder=5, label="missing (NaN)",
        )
 
    ax.set_xlabel("Day")
    ax.set_ylabel("Concentration")
    ax.set_title(title or str(clonotype_id))
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
 
    if standalone:
        plt.tight_layout()
        plt.show()
        

def plot_noise_model(
    model: dict,
    show_pooled: bool = True,
    show_per_donor: bool = True,
    ax=None,
) -> None:
    """
    Visualize the slope noise model: rate distributions and fitted normals.

    Parameters
    ----------
    model : dict
        Output of compute_slope_noise_model.
    show_pooled : bool
        Whether to show the pooled distribution. Default True.
    show_per_donor : bool
        Whether to overlay per-donor fitted normals. Default True.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure is created and shown.

    Returns
    -------
    None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 4))

    # Plot pooled histogram
    if show_pooled:
        pooled_rates = model["pooled"]["rates"]
        ax.hist(
            pooled_rates, bins=100, density=True,
            color="lightgray", edgecolor="none", alpha=0.8, label="pooled rates",
        )
        x = np.linspace(pooled_rates.min(), pooled_rates.max(), 300)
        y, fit_label = _pdf(x, model["pooled"])
        ax.plot(x, y, color="black", linewidth=1.5, label=f"pooled fit  {fit_label}")

    # Overlay per-donor fitted curves
    if show_per_donor:
        colors = plt.cm.tab10(np.linspace(0, 1, len(model["per_donor"])))
        for (donor, values), color in zip(model["per_donor"].items(), colors):
            rates = values["rates"]
            x = np.linspace(rates.min(), rates.max(), 300)
            y, fit_label = _pdf(x, values)
            ax.plot(x, y, linewidth=1.0, linestyle="--", color=color,
                    label=f"{donor}  {fit_label}", alpha=0.8)

    ax.set_xlabel(f"Rate (log2 fold-change / day)  [{model['norm_col']}]")
    ax.set_ylabel("Density")
    ax.set_title(f"Slope noise model ({model.get('fit_distribution', 'normal')})")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    if standalone:
        plt.tight_layout()
        plt.show()

def plot_trajectories(
    df: pd.DataFrame,
    y_col: str = "count",
    clono_col: str = "clono",
    day_col: str = "day",
    ax: plt.Axes = None,
    title: str = None,
    color_map: dict = None,
    legend: bool = True,
    alpha: float = 0.7,
    linewidth: float = 1.2,
) -> plt.Axes:
    """
    Plot multiple clonotype trajectories overlaid on a single axes.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame for a single patient/donor. Must contain
        columns for day, clono, and the chosen y column.
    y_col : str
        Column to plot on the y-axis (e.g. 'count', 'clr', 'cpm').
    clono_col : str
        Column identifying clonotypes.
    day_col : str
        Column identifying timepoints.
    ax : plt.Axes, optional
        Axes to draw on. A new figure is created if None.
    title : str, optional
        Axes title.
    color_map : dict, optional
        Mapping of clonotype -> color. If None, cycles through the
        current matplotlib color cycle.
    legend : bool
        Whether to draw a legend. Set False when plotting many clonotypes.
    alpha : float
        Line transparency.
    linewidth : float
        Line width.

    Returns
    -------
    plt.Axes
        The axes with trajectories drawn.
    """
    if ax is None:
        _, ax = plt.subplots()

    clonotypes = df[clono_col].unique()
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, clono in enumerate(clonotypes):
        sub = df[df[clono_col] == clono].sort_values(day_col)
        color = color_map[clono] if color_map else prop_cycle[i % len(prop_cycle)]
        ax.plot(
            sub[day_col],
            sub[y_col],
            marker="o",
            markersize=3,
            label=clono,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

    ax.set_xlabel(day_col)
    ax.set_ylabel(y_col)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    return ax