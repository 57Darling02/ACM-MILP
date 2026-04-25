import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


GRAPH_FEATURE_COLUMNS = [
    "n_conss",
    "n_vars",
    "n_cont_vars",
    "ratio_cont_vars",
    "n_nonzeros",
    "coef_dens",
    "var_degree_mean",
    "var_degree_std",
    "var_degree_min",
    "var_degree_max",
    "cons_degree_mean",
    "cons_degree_std",
    "cons_degree_min",
    "cons_degree_max",
    "lhs_mean",
    "lhs_std",
    "lhs_min",
    "lhs_max",
    "rhs_mean",
    "rhs_std",
    "rhs_min",
    "rhs_max",
    "obj_mean",
    "obj_std",
    "obj_min",
    "obj_max",
    "clustering",
    "modularity",
]

HARDNESS_COLUMNS = [
    ("solving_time", "Solving Time"),
    ("num_nodes", "B&B Nodes"),
]

COLOR_CYCLE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "plot"


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "instance" in df.columns:
        return df.set_index("instance")
    return df


def pick_feature_columns(
    df: pd.DataFrame,
    preferred_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    if preferred_columns:
        return [col for col in preferred_columns if col in df.columns]

    ordered = [col for col in GRAPH_FEATURE_COLUMNS if col in df.columns]
    if ordered:
        return ordered

    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def to_numeric_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return values.astype(float)


def build_title(base: str, annotation: Optional[str] = None) -> str:
    if annotation:
        return "{0}\n{1}".format(base, annotation)
    return base


def compute_kde_curve(values: np.ndarray, points: int = 256) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if values.size < 2:
        return None
    std = float(np.std(values))
    if std < 1e-12:
        return None

    left = float(np.min(values))
    right = float(np.max(values))
    padding = max(std * 0.2, 1e-9)
    x = np.linspace(left - padding, right + padding, points)
    kde = gaussian_kde(values)
    y = kde(x)
    return x, y


def normalize_histogram(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    counts = np.clip(counts, 0.0, None)
    total = counts.sum()
    if total <= 0:
        return np.full_like(counts, 1.0 / len(counts), dtype=float)
    counts = counts + 1e-12
    return counts / counts.sum()


def compute_js_divergence(values_a: np.ndarray, values_b: np.ndarray, bins: int = 5) -> Optional[float]:
    if bins <= 0:
        raise ValueError("bins must be positive")

    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]

    if values_a.size == 0 or values_b.size == 0:
        return None

    pooled = np.concatenate([values_a, values_b])
    if pooled.std() < 1e-10:
        return 0.0

    _, edges = np.histogram(pooled, bins=bins, density=True)
    p_counts, _ = np.histogram(values_a, bins=edges)
    q_counts, _ = np.histogram(values_b, bins=edges)

    p = normalize_histogram(p_counts)
    q = normalize_histogram(q_counts)
    m = 0.5 * (p + q)

    js_divergence = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    return float(js_divergence)


def compute_js_divergence_from_series(series_a: pd.Series, series_b: pd.Series, bins: int = 5) -> Optional[float]:
    values_a = to_numeric_series(series_a).to_numpy()
    values_b = to_numeric_series(series_b).to_numpy()
    return compute_js_divergence(values_a, values_b, bins=bins)


def make_distribution_plot(
    series_map: Dict[str, pd.Series],
    title: str,
    x_label: str,
    output_path: str,
    bins: int = 30,
    dpi: int = 200,
    show_hist: bool = True,
    annotation: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for idx, (label, series) in enumerate(series_map.items()):
        values = to_numeric_series(series).to_numpy()
        if values.size == 0:
            continue

        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        if show_hist:
            ax.hist(
                values,
                bins=min(bins, max(5, values.size)),
                density=True,
                alpha=0.18,
                color=color,
                edgecolor="none",
            )

        kde_curve = compute_kde_curve(values)
        if kde_curve is None:
            ax.axvline(float(values[0]), color=color, linewidth=2.0, label=f"{label} (constant)")
        else:
            x, y = kde_curve
            ax.plot(x, y, color=color, linewidth=2.2, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(build_title(title, annotation))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_grid_kde_plots(
    dataframe_map: Dict[str, pd.DataFrame],
    columns: Sequence[str],
    title: str,
    output_path: str,
    bins: int = 30,
    dpi: int = 200,
    max_cols: int = 3,
    annotations: Optional[Dict[str, str]] = None,
) -> None:
    columns = [col for col in columns if any(col in df.columns for df in dataframe_map.values())]
    if not columns:
        return

    n_cols = min(max_cols, max(1, len(columns)))
    n_rows = int(math.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for idx, column in enumerate(columns):
        ax = axes[idx // n_cols, idx % n_cols]
        plotted = False
        for label_idx, (label, df) in enumerate(dataframe_map.items()):
            if column not in df.columns:
                continue
            values = to_numeric_series(df[column]).to_numpy()
            if values.size == 0:
                continue

            color = COLOR_CYCLE[label_idx % len(COLOR_CYCLE)]
            ax.hist(
                values,
                bins=min(bins, max(5, values.size)),
                density=True,
                alpha=0.15,
                color=color,
                edgecolor="none",
            )
            kde_curve = compute_kde_curve(values)
            if kde_curve is None:
                ax.axvline(float(values[0]), color=color, linewidth=1.8, label=f"{label} (constant)")
            else:
                x, y = kde_curve
                ax.plot(x, y, color=color, linewidth=2.0, label=label)
            plotted = True

        ax.set_title(build_title(column, (annotations or {}).get(column)))
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2, linestyle="--")
        if plotted:
            ax.legend(fontsize=8)

    total_axes = n_rows * n_cols
    for idx in range(len(columns), total_axes):
        fig.delaxes(axes[idx // n_cols, idx % n_cols])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_scatter_plot(
    dataframe_map: Dict[str, pd.DataFrame],
    x_col: str,
    y_col: str,
    title: str,
    output_path: str,
    dpi: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False

    for idx, (label, df) in enumerate(dataframe_map.items()):
        if x_col not in df.columns or y_col not in df.columns:
            continue

        x = to_numeric_series(df[x_col])
        y = to_numeric_series(df[y_col])
        xy = pd.concat([x.rename(x_col), y.rename(y_col)], axis=1).dropna()
        if xy.empty:
            continue

        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        ax.scatter(
            xy[x_col],
            xy[y_col],
            s=14,
            alpha=0.45,
            color=color,
            label=label,
            edgecolors="none",
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if ax.get_xlim()[0] >= 0 and ax.get_xlim()[1] > 0:
        ax.set_xscale("symlog", linthresh=1.0)
    if ax.get_ylim()[0] >= 0 and ax.get_ylim()[1] > 0:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_feature_suite(
    dataframe_map: Dict[str, pd.DataFrame],
    output_dir: str,
    columns: Sequence[str],
    title_prefix: str,
    bins: int = 30,
    dpi: int = 200,
    annotations: Optional[Dict[str, str]] = None,
) -> List[str]:
    ensure_dir(output_dir)
    saved_paths = []

    overview_path = os.path.join(output_dir, "kde_overview.png")
    make_grid_kde_plots(
        dataframe_map=dataframe_map,
        columns=columns,
        title=f"{title_prefix} Graph Statistics KDE",
        output_path=overview_path,
        bins=bins,
        dpi=dpi,
        annotations=annotations,
    )
    if os.path.exists(overview_path):
        saved_paths.append(overview_path)

    for column in columns:
        file_name = "kde_{0}.png".format(sanitize_filename(column))
        plot_path = os.path.join(output_dir, file_name)
        series_map = {
            label: df[column]
            for label, df in dataframe_map.items()
            if column in df.columns
        }
        make_distribution_plot(
            series_map=series_map,
            title=f"{title_prefix}: {column}",
            x_label=column,
            output_path=plot_path,
            bins=bins,
            dpi=dpi,
            annotation=(annotations or {}).get(column),
        )
        if os.path.exists(plot_path):
            saved_paths.append(plot_path)

    return saved_paths


def plot_hardness_suite(
    dataframe_map: Dict[str, pd.DataFrame],
    output_dir: str,
    title_prefix: str,
    bins: int = 30,
    dpi: int = 200,
    annotations: Optional[Dict[str, str]] = None,
) -> List[str]:
    ensure_dir(output_dir)
    saved_paths = []

    for column, label in HARDNESS_COLUMNS:
        plot_path = os.path.join(output_dir, "{0}_distribution.png".format(sanitize_filename(column)))
        series_map = {
            frame_label: df[column]
            for frame_label, df in dataframe_map.items()
            if column in df.columns
        }
        make_distribution_plot(
            series_map=series_map,
            title=f"{title_prefix}: {label} Distribution",
            x_label=label,
            output_path=plot_path,
            bins=bins,
            dpi=dpi,
            annotation=(annotations or {}).get(column),
        )
        if os.path.exists(plot_path):
            saved_paths.append(plot_path)

    scatter_path = os.path.join(output_dir, "hardness_scatter.png")
    make_scatter_plot(
        dataframe_map=dataframe_map,
        x_col="num_nodes",
        y_col="solving_time",
        title=f"{title_prefix}: Hardness Scatter",
        output_path=scatter_path,
        dpi=dpi,
    )
    if os.path.exists(scatter_path):
        saved_paths.append(scatter_path)

    return saved_paths


def find_tensorboard_event_files(path_or_dir: str) -> List[str]:
    if os.path.isfile(path_or_dir):
        return [path_or_dir]

    event_files = []
    for root, _, files in os.walk(path_or_dir):
        for file_name in files:
            if "tfevents" in file_name:
                event_files.append(os.path.join(root, file_name))
    event_files.sort()
    return event_files


def load_scalar_history_from_events(
    path_or_dir: str,
    tag: str,
) -> pd.DataFrame:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is not installed. Install `tensorboard` to parse event files."
        ) from exc

    rows = []
    for event_file in find_tensorboard_event_files(path_or_dir):
        accumulator = EventAccumulator(event_file)
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])
        if tag not in scalar_tags:
            continue
        for event in accumulator.Scalars(tag):
            rows.append(
                {
                    "source": event_file,
                    "step": int(event.step),
                    "value": float(event.value),
                    "wall_time": float(event.wall_time),
                }
            )

    if not rows:
        raise RuntimeError("No scalar data found for tag `{0}`.".format(tag))

    history = pd.DataFrame(rows).sort_values(["step", "wall_time"]).drop_duplicates(subset=["step"], keep="last")
    return history.reset_index(drop=True)


def parse_loss_history_from_log(log_path: str) -> pd.DataFrame:
    pattern = re.compile(r"Step\s+(\d+)/(\d+)\.\s+Loss:\s+([0-9eE+\-.]+)")
    rows = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            rows.append(
                {
                    "step": int(match.group(1)),
                    "total_steps": int(match.group(2)),
                    "value": float(match.group(3)),
                }
            )

    if not rows:
        raise RuntimeError("No training loss lines found in `{0}`.".format(log_path))

    return pd.DataFrame(rows).sort_values("step").drop_duplicates(subset=["step"], keep="last").reset_index(drop=True)


def compute_smoothed(values, window: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if window <= 1 or array.size == 0:
        return array

    series = pd.Series(array)
    smoothed = series.rolling(window=window, min_periods=1, center=False).mean()
    return smoothed.to_numpy()


def plot_training_loss(
    history: pd.DataFrame,
    output_path: str,
    title: str,
    smooth_window: int = 50,
    dpi: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(history["step"], history["value"], color="#4c78a8", alpha=0.3, linewidth=1.2, label="Raw loss")

    if smooth_window > 1:
        smoothed = compute_smoothed(history["value"], smooth_window)
        ax.plot(history["step"], smoothed, color="#d62728", linewidth=2.2, label="Smoothed loss")

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_manifest(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

