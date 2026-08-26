#!/usr/bin/env python3
"""
Plot table1_.csv as dual-line charts showing Hard vs Soft Negative impact.

Each subplot shows:
- Solid line: Hard Negative (NRR(hard) or legacy Score column)
- Dashed line: Soft Negative (NRR(soft) or legacy Score column)
- Dash-dot line: BGE-m3 (NRR (BGE-m3) or legacy Score column)
- Plateau annotation: For GPT-4o FeTaQA Hard line (K=20→50)

X: K
Y: NRR
Hue: Models
Facet: Dataset (3 subplots in a row)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.lines import Line2D

MODEL_DISPLAY_NAMES = {
    "gpt-4o": "GPT-4o",
    "gpt-oss-20b": "GPT-OSS-20b",
    "claude-haiku-4-5": "Claude-Haiku-4.5",
    "qwen3-32b": "Qwen-3-32b",
}


def model_display_name(model: str) -> str:
    """Return the paper-facing name while preserving model IDs in the data."""
    return MODEL_DISPLAY_NAMES.get(model, model)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and normalize table1_.csv data."""
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    if not all(col in df.columns for col in ["Dataset", "Model", "K"]):
        raise SystemExit("CSV missing required columns: ['Dataset', 'Model', 'K']")

    hard_column, soft_column, real_column = resolve_table1_metric_columns(df)
    
    # Convert to numeric
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df[hard_column] = pd.to_numeric(df[hard_column], errors="coerce")
    df[soft_column] = pd.to_numeric(df[soft_column], errors="coerce")
    if real_column is not None and real_column in df.columns:
        df[real_column] = pd.to_numeric(df[real_column], errors="coerce")
    
    drop_cols = ["K", hard_column, soft_column, "Dataset", "Model"]
    if real_column is not None and real_column in df.columns:
        drop_cols.append(real_column)
    df = df.dropna(subset=drop_cols)
    return df


def resolve_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column that exists in the dataframe."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def resolve_table1_metric_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """Resolve the current soft/hard/BGE-m3 columns, preferring NRR columns."""
    hard_column = resolve_first_existing_column(
        df,
        ["NRR(hard)", "NRR (hard)", "Score (Hard Avg)"],
    )
    soft_column = resolve_first_existing_column(
        df,
        ["NRR(soft)", "NRR (soft)", "Score (Soft Avg)"],
    )
    real_column = resolve_first_existing_column(
        df,
        [
            "NRR (BGE-m3)",
            "NRR(BGE-m3)",
            "NRR(BGE-M3)",
            "Score (BGE-m3)",
            "Score (Real Avg)",
            "Score (Real)",
            "NRR(real)",
            "NRR(Real)",
        ],
    )

    if hard_column is None or soft_column is None or real_column is None:
        raise SystemExit(
            "CSV missing required metric columns; expected Hard, Soft, and BGE-M3 NRR columns."
        )

    return hard_column, soft_column, real_column


def resolve_sem_column(df: pd.DataFrame, mean_column: str) -> str | None:
    """Return the first matching SEM column for a mean column, if present."""
    candidate_columns = [
        mean_column.replace("Avg", "SEM"),
        mean_column.replace("Avg", "Sem"),
        mean_column.replace("Avg", "sem"),
        mean_column.replace("Avg", "_SEM"),
        mean_column.replace("Avg", "_sem"),
        mean_column.replace("Avg", " (SEM)"),
        mean_column.replace("Avg", " (sem)"),
        mean_column.replace("Score", "NRR"),
        f"SEM({mean_column})",
        f"sem({mean_column})",
    ]
    for candidate in candidate_columns:
        if candidate in df.columns:
            return candidate
    return None


def resolve_real_column(df: pd.DataFrame, mean_column: str) -> str | None:
    """Return the first matching real-retriever column for a mean column, if present."""
    candidate_columns = [
        "NRR (BGE-m3)",
        "NRR(BGE-m3)",
        "NRR(BGE-M3)",
        "Score (Real Avg)",
        "Score (Real)",
        "NRR(real)",
        "NRR(Real)",
    ]
    for candidate in candidate_columns:
        if candidate in df.columns:
            return candidate
    return None


def get_sem_series(
    subset: pd.DataFrame,
    mean_column: str,
    sem_column: str | None,
) -> pd.Series:
    """Return frozen SEM values, failing when they are unavailable."""
    if sem_column and sem_column in subset.columns:
        sem = pd.to_numeric(subset[sem_column], errors="coerce")
        if sem.notna().all():
            return sem
    raise SystemExit(f"SEM requested but no complete frozen SEM column exists for {mean_column}")


def get_real_series(
    subset: pd.DataFrame,
    real_column: str,
) -> pd.Series:
    """Return complete frozen BGE-M3 values."""
    real = pd.to_numeric(subset[real_column], errors="coerce")
    if real.notna().all():
        return real
    raise SystemExit("BGE-M3 plotting requires complete frozen values")


def get_real_sem_series(
    subset: pd.DataFrame,
    real_sem_column: str | None,
) -> pd.Series:
    """Return complete frozen BGE-M3 SEM values."""
    if real_sem_column and real_sem_column in subset.columns:
        sem = pd.to_numeric(subset[real_sem_column], errors="coerce")
        if sem.notna().all():
            return sem
    raise SystemExit("SEM requested but no complete frozen BGE-M3 SEM column exists")


def darken_color(color, factor: float = 0.55):
    """Darken a matplotlib color while keeping the same hue family."""
    rgb = mcolors.to_rgb(color)
    return tuple(max(0.0, min(1.0, channel * factor)) for channel in rgb)


def plot_mean_with_sem_band(
    ax,
    x_values,
    mean_values,
    sem_values,
    *,
    color,
    linestyle,
    linewidth,
    marker,
    markerfacecolor,
    markeredgecolor,
    markersize,
    label,
    line_alpha=1.0,
    band_alpha=0.08,
):
    """Plot a mean line with an optional SEM band behind it."""
    if sem_values is not None:
        ax.fill_between(
            x_values,
            mean_values - sem_values,
            mean_values + sem_values,
            color=color,
            alpha=band_alpha,
            linewidth=0,
            zorder=1,
        )

    ax.plot(
        x_values,
        mean_values,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markerfacecolor=markerfacecolor,
        markeredgecolor=markeredgecolor,
        markersize=markersize,
        alpha=line_alpha,
        zorder=2,
    )


def plot_dual_lines(
    df: pd.DataFrame,
    output_dir: Path,
    show_plateau: bool = False,
    show_sem: bool = False,
) -> None:
    """
    Plot dual-line charts: Hard (solid) vs Soft (dashed) lines per model.
    Include plateau annotation for GPT-4o FeTaQA Hard line (K=20→50) if enabled.
    """
    
    # Map old dataset names to new ones if needed
    dataset_mapping = {
        "E2E-WTQ": "E2E-WTQ",
        "BIRD-SQL": "BIRD-SQL",
        "OTT-QA": "OTT-QA",
        "FeTaQA": "FeTaQA"  # In case it appears
    }
    
    # Get unique datasets in order
    available_datasets = df["Dataset"].unique()
    # Try to match standard order: E2E-WTQ, FeTaQA, OTT-QA or similar
    datasets = [d for d in ["E2E-WTQ", "FeTaQA", "OTT-QA", "BIRD-SQL"] if d in available_datasets]
    if not datasets:
        datasets = sorted(available_datasets)
    
    models = sorted(df["Model"].unique())
    k_values = sorted(pd.to_numeric(df["K"], errors="coerce").dropna().unique())
    k_tick_labels = [str(int(k)) if float(k).is_integer() else str(k) for k in k_values]
    
    # Color map per model for consistency
    color_cycle = plt.get_cmap("tab10")
    model_colors = {m: color_cycle(i % 10) for i, m in enumerate(models)}
    # Distinct marker per model to aid grayscale readability
    marker_list = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '8']
    model_markers = {m: marker_list[i % len(marker_list)] for i, m in enumerate(models)}
    
    hard_column, soft_column, real_column = resolve_table1_metric_columns(df)
    hard_sem_column = resolve_sem_column(df, hard_column)
    soft_sem_column = resolve_sem_column(df, soft_column)
    real_sem_column = resolve_sem_column(df, real_column) if real_column else None

    # Create one figure per dataset (for clarity)
    for dataset in datasets:
        subset = df[df["Dataset"] == dataset].sort_values(["Model", "K"])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model in models:
            model_data = subset[subset["Model"] == model].sort_values("K")
            if model_data.empty:
                continue

            hard_sem = get_sem_series(model_data, hard_column, hard_sem_column) if show_sem else None
            soft_sem = get_sem_series(model_data, soft_column, soft_sem_column) if show_sem else None
            real_values = get_real_series(model_data, real_column)
            real_sem = get_real_sem_series(model_data, real_sem_column) if show_sem else None
            
            # Hard Negative (Solid line)
            plot_mean_with_sem_band(
                ax,
                model_data["K"],
                pd.to_numeric(model_data[hard_column], errors="coerce").fillna(0.0),
                hard_sem,
                color=model_colors[model],
                linestyle="-",
                linewidth=2.2,
                marker=model_markers[model],
                markerfacecolor=model_colors[model],
                markeredgecolor=model_colors[model],
                markersize=5,
                label=f"{model_display_name(model)} (Hard)",
            )
            
            # Soft Negative (Dashed line)
            plot_mean_with_sem_band(
                ax,
                model_data["K"],
                pd.to_numeric(model_data[soft_column], errors="coerce").fillna(0.0),
                soft_sem,
                color=model_colors[model],
                linestyle="--",
                linewidth=2.2,
                marker=model_markers[model],
                markerfacecolor='none',
                markeredgecolor=model_colors[model],
                markersize=5,
                label=f"{model_display_name(model)} (Soft)",
                line_alpha=0.8,
            )

            real_line_color = darken_color(model_colors[model], 0.85)
            plot_mean_with_sem_band(
                ax,
                model_data["K"],
                real_values,
                real_sem,
                color=real_line_color,
                linestyle="-.",
                linewidth=2.8,
                marker='X',
                markerfacecolor=real_line_color,
                markeredgecolor=real_line_color,
                markersize=6,
                label=f"{model_display_name(model)} (BGE-m3)",
                line_alpha=1.0,
                band_alpha=0.03,
            )
        
        if show_plateau and dataset == "FeTaQA":
            gpt4o_data = subset[subset["Model"] == "gpt-4o"].sort_values("K")
            if not gpt4o_data.empty:
                # Find K=20 and K=50
                k_vals = gpt4o_data["K"].values
                if 20 in k_vals and 50 in k_vals:
                    # Add light gray background for plateau region
                    plateau_x_min = 15  # A bit before K=20
                    plateau_x_max = 55  # A bit after K=50
                    y_min, y_max = ax.get_ylim()
                    ax.axvspan(plateau_x_min, plateau_x_max, 
                              alpha=0.1, color="gray", zorder=0)
                    
                    # Add text annotation
                    mid_k = (20 + 50) / 2
                    mid_score = gpt4o_data[gpt4o_data["K"].isin([20, 50])][hard_column].mean()
                    ax.text(mid_k, mid_score + 0.01, "Resilient Plateau",
                           fontsize=11, ha="center", fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        ax.set_title(f"{dataset}", 
            fontsize=44, fontweight="bold")
        ax.set_xlabel("Retrieval depth K (Number of Retrieved Tables)", fontsize=52)
        ax.set_ylabel("NRR", fontsize=52)
        ax.tick_params(axis="both", which="major", labelsize=48)
        ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
        
        # Create better legend for color (models), linestyle (Hard vs Soft/Real), and SEM band
        # Model legend: colored markers (no line)
        model_handles = [Line2D([0], [0], color=model_colors[m], marker='o', linestyle='', markersize=6) for m in models if not subset[subset['Model']==m].empty]
        model_labels = [model_display_name(m) for m in models if not subset[subset['Model']==m].empty]

                # Style legend: solid vs dashed vs dash-dot (use black so works in grayscale)
        style_handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=2.0),
                                 Line2D([0], [0], color='black', linestyle='--', linewidth=2.0),
                                 Line2D([0], [0], color='black', linestyle='-.', linewidth=2.0)]
        style_labels = ['Hard negative', 'Soft negative', 'BGE-M3']

        band_handles = []
        band_labels = []
        if show_sem:
            band_handles = [mpatches.Patch(facecolor='gray', alpha=0.14, edgecolor='none')]
            band_labels = ['95% confidence interval']

        # Combine model/style/band into one figure-level legend at top-center.
        combined_handles = model_handles + style_handles + band_handles
        combined_labels = model_labels + style_labels + band_labels
        fig.legend(
            combined_handles,
            combined_labels,
            loc="upper center",
            ncol=min(len(combined_handles), 8),
            frameon=True,
            fontsize=36,
            bbox_to_anchor=(0.5, 1.12),
            framealpha=0.95,
            markerscale=2.6,
            handlelength=3.6,
            handletextpad=0.6,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_safe = dataset.replace("-", "_").replace(" ", "_").lower()
        png_path = output_dir / f"table1_new_impact_gap_{dataset_safe}.png"
        pdf_path = output_dir / f"table1_new_impact_gap_{dataset_safe}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
        
        plt.close(fig)


def plot_combined_facets(
    df: pd.DataFrame,
    output_dir: Path,
    num_cols: int = 3,
    show_plateau: bool = False,
    show_sem: bool = False,
) -> None:
    """
    Plot all datasets in an NxM faceted layout with dual lines.
    Show the Impact Gap more clearly: Soft - Hard = semantic interference penalty.
    
    Args:
        num_cols: Number of columns for faceted layout (default: 3)
        show_plateau: Whether to show plateau annotations (default: True)
    """
    
    # Get available datasets in a consistent order
    available_datasets = df["Dataset"].unique()
    # Try standard order first
    ordered_datasets = ["E2E-WTQ", "FeTaQA", "OTTQA", "OTT-QA", "BIRD-SQL"]
    datasets = [d for d in ordered_datasets if d in available_datasets]
    if not datasets:
        # If none of the standard names match, use sorted available datasets
        datasets = sorted(available_datasets)
    
    n_datasets = len(datasets)
    k_values = sorted(pd.to_numeric(df["K"], errors="coerce").dropna().unique())
    k_tick_labels = [str(int(k)) for k in k_values]

    
    models = sorted(df["Model"].unique())
    
    hard_column, soft_column, real_column = resolve_table1_metric_columns(df)
    hard_sem_column = resolve_sem_column(df, hard_column)
    soft_sem_column = resolve_sem_column(df, soft_column)
    real_sem_column = resolve_sem_column(df, real_column) if real_column else None

    # Color map per model
    color_cycle = plt.get_cmap("tab10")
    model_colors = {m: color_cycle(i % 10) for i, m in enumerate(models)}
    # Distinct marker per model to aid grayscale readability
    marker_list = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '8']
    model_markers = {m: marker_list[i % len(marker_list)] for i, m in enumerate(models)}
    
    # Calculate grid layout
    n_rows = (n_datasets + num_cols - 1) // num_cols  # Ceiling division
    # Make the overall figure square so faceted subplots appear proportional
    cell_size = 5
    total_w = cell_size * num_cols
    total_h = cell_size * n_rows
    base = max(total_w, total_h)
    # Reduce overall height for paper space efficiency (width kept as base)
    figsize = (base, base * 0.66)
    
    fig, axes = plt.subplots(n_rows, num_cols, figsize=figsize, sharey=False)
    
    # Flatten axes array for easier iteration
    if n_rows == 1 and num_cols == 1:
        axes = [axes]
    elif n_rows == 1 or num_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df["Dataset"] == dataset].sort_values(["Model", "K"])

        for model in models:
            model_data = subset[subset["Model"] == model].sort_values("K")
            if model_data.empty:
                continue

            hard_sem = get_sem_series(model_data, hard_column, hard_sem_column) if show_sem else None
            soft_sem = get_sem_series(model_data, soft_column, soft_sem_column) if show_sem else None
            real_values = get_real_series(model_data, real_column)
            real_sem = get_real_sem_series(model_data, real_sem_column) if show_sem else None

            # Hard Negative (Solid line)
            plot_mean_with_sem_band(
                ax,
                model_data["K"],
                pd.to_numeric(model_data[hard_column], errors="coerce").fillna(0.0),
                hard_sem,
                color=model_colors[model],
                linestyle="-",
                linewidth=2.0,
                marker=model_markers[model],
                markerfacecolor=model_colors[model],
                markeredgecolor=model_colors[model],
                markersize=8,
                label=f"{model_display_name(model)} Hard",
            )
            
            # Soft Negative (Dashed line)
            plot_mean_with_sem_band(
                ax,
                model_data["K"],
                pd.to_numeric(model_data[soft_column], errors="coerce").fillna(0.0),
                soft_sem,
                color=model_colors[model],
                linestyle="--",
                linewidth=2.0,
                marker=model_markers[model],
                markerfacecolor='none',
                markeredgecolor=model_colors[model],
                markersize=8,
                label=f"{model_display_name(model)} Soft",
                line_alpha=0.7,
            )

            plot_mean_with_sem_band(
                ax,
                model_data["K"],
                real_values,
                real_sem,
                color=darken_color(model_colors[model], 0.85),
                linestyle="-.",
                linewidth=2.6,
                marker='X',
                markerfacecolor=darken_color(model_colors[model], 0.85),
                markeredgecolor=darken_color(model_colors[model], 0.85),
                markersize=10,
                label=f"{model_display_name(model)} BGE-m3",
                line_alpha=1.0,
                band_alpha=0.03,
            )
        
        ax.set_title(f"{dataset}", fontsize=34, fontweight="normal")
        ax.set_xlabel("Retrieval depth K", fontsize=28)
        ax.set_xticks(k_values)
        ax.set_xticklabels(k_tick_labels)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
        ax.tick_params(axis="x", which="major", labelsize=20)
        ax.tick_params(axis="y", which="major", labelsize=36)
        ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    
    # Hide extra subplots if we don't have enough datasets
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    axes[0].set_ylabel("NRR", fontsize=40)
    
    # Shared legend at bottom
    # Build model legend handles (colors) and style legend handles (linestyles)
    model_handles = [Line2D([0], [0], color=model_colors[m], marker='o', linestyle='', markersize=10) for m in models]
    model_labels = [model_display_name(model) for model in models]
    style_handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=3.0),
                     Line2D([0], [0], color='black', linestyle='--', linewidth=3.0),
                     Line2D([0], [0], color='black', linestyle='-.', linewidth=3.0)]
    style_labels = ['Hard negative', 'Soft negative', 'BGE-M3']
    band_handles = []
    band_labels = []
    if show_sem:
        band_handles = [mpatches.Patch(facecolor='gray', alpha=0.14, edgecolor='none')]
        band_labels = ['95% confidence interval']

    # Combine model/style/band into one figure-level legend at top-center.
    combined_handles = model_handles + style_handles + band_handles
    combined_labels = model_labels + style_labels + band_labels
    legend = fig.legend(
        combined_handles,
        combined_labels,
        loc="upper center",
        ncol=max(1, (len(combined_handles) + 1) // 2),
        frameon=True,
        fontsize=30,
        bbox_to_anchor=(0.5, 1.12),
        framealpha=0.95,
        markerscale=2.6,
        handlelength=3.6,
        handletextpad=0.6,
    )

    # Draw the figure to compute legend and axes sizes, then scale subplots
    # so their combined width matches the legend width (visual parity).
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_box = legend.get_window_extent(renderer=renderer)
        # fraction of figure width occupied by legend (in display coords)
        legend_frac = legend_box.width / fig.bbox.width

        # fraction of figure width occupied by visible axes combined
        axes_total_frac = sum(ax.get_position().width for ax in axes if ax.get_visible())

        if axes_total_frac > 0 and legend_frac > 0:
            scale = legend_frac / axes_total_frac
            # Constrain scale to reasonable bounds to avoid extreme resizes
            scale = max(1.0, min(scale, 2.0))
            if scale > 1.01:
                new_width = fig.get_size_inches()[0] * scale
                fig.set_size_inches(new_width, fig.get_size_inches()[1], forward=True)
                # Redraw so layout reflects new size
                fig.canvas.draw()
    except Exception:
        # If anything goes wrong (headless renderer issues), fall back gracefully
        pass

    fig.tight_layout(rect=[0.01, 0, 0.99, 0.90])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "table1_new_impact_gap_faceted.png"
    pdf_path = output_dir / "table1_new_impact_gap_faceted.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_two_column_workflow(
    df: pd.DataFrame,
    output_dir: Path,
    show_sem: bool = False,
) -> None:
    """Draw a readable two-column Controlled/Uncontrolled workflow.

    Rows group results by dataset.  The left column contains only controlled
    NRR curves (hard and soft negatives); the right column contains only the
    natural BGE-M3 hit-rate curves.  This preserves the caption's line-style
    semantics while avoiding twelve overlapping curves in a single panel.
    """
    available_datasets = df["Dataset"].unique()
    ordered_datasets = ["E2E-WTQ", "FeTaQA", "OTTQA", "OTT-QA", "BIRD-SQL"]
    datasets = [dataset for dataset in ordered_datasets if dataset in available_datasets]
    if not datasets:
        datasets = sorted(available_datasets)

    models = sorted(df["Model"].unique())
    k_values = sorted(pd.to_numeric(df["K"], errors="coerce").dropna().unique())
    k_tick_labels = [str(int(k)) if float(k).is_integer() else str(k) for k in k_values]

    hard_column, soft_column, real_column = resolve_table1_metric_columns(df)
    hard_sem_column = resolve_sem_column(df, hard_column)
    soft_sem_column = resolve_sem_column(df, soft_column)
    real_sem_column = resolve_sem_column(df, real_column) if real_column else None

    # Fixed Okabe-Ito color-blind-friendly mapping. Keep these assignments
    # independent of alphabetical model order.
    fixed_model_colors = {
        "gpt-4o": "#56B4E9",             # Sky Blue
        "gpt-oss-20b": "#E69F00",        # Orange
        "claude-haiku-4-5": "#009E73",   # Bluish Green
        "qwen3-32b": "#D55E00",          # Vermillion
    }
    fallback_palette = ["#0072B2", "#CC79A7", "#F0E442", "#000000"]
    model_colors = {
        model: fixed_model_colors.get(model, fallback_palette[index % len(fallback_palette)])
        for index, model in enumerate(models)
    }
    markers = ["o", "s", "^", "D", "P", "v"]
    model_markers = {model: markers[index % len(markers)] for index, model in enumerate(models)}

    fig, axes = plt.subplots(
        len(datasets),
        2,
        figsize=(12.2, 3.05 * len(datasets) + 1.15),
        sharex=True,
        # Each chart needs its own range: controlled NRR and BGE-M3 NRR
        # occupy substantially different intervals.
        sharey=False,
        squeeze=False,
    )

    for row, dataset in enumerate(datasets):
        controlled_ax, uncontrolled_ax = axes[row]
        subset = df[df["Dataset"] == dataset].sort_values(["Model", "K"])

        for model in models:
            model_data = subset[subset["Model"] == model].sort_values("K")
            if model_data.empty:
                continue

            hard_values = pd.to_numeric(model_data[hard_column], errors="coerce")
            soft_values = pd.to_numeric(model_data[soft_column], errors="coerce")
            real_values = get_real_series(model_data, real_column)
            hard_sem = get_sem_series(model_data, hard_column, hard_sem_column) if show_sem else None
            soft_sem = get_sem_series(model_data, soft_column, soft_sem_column) if show_sem else None
            real_sem = get_real_sem_series(model_data, real_sem_column) if show_sem else None

            common = dict(
                color=model_colors[model],
                marker=model_markers[model],
                markeredgecolor=model_colors[model],
                markersize=5.2,
                label=model_display_name(model),
                band_alpha=0.06,
            )
            plot_mean_with_sem_band(
                controlled_ax,
                model_data["K"],
                hard_values,
                hard_sem,
                linestyle="-",
                linewidth=2.15,
                markerfacecolor=model_colors[model],
                **common,
            )
            plot_mean_with_sem_band(
                controlled_ax,
                model_data["K"],
                soft_values,
                soft_sem,
                linestyle="--",
                linewidth=1.9,
                markerfacecolor="white",
                line_alpha=0.9,
                **common,
            )
            plot_mean_with_sem_band(
                uncontrolled_ax,
                model_data["K"],
                real_values,
                real_sem,
                linestyle="-.",
                linewidth=2.25,
                markerfacecolor=model_colors[model],
                **common,
            )

        controlled_ax.set_ylabel(
            f"{dataset}\nNRR",
            fontsize=14,
            fontweight="normal",
            labelpad=8,
        )
        uncontrolled_ax.set_ylabel("NRR", fontsize=14, labelpad=8)
        for ax in (controlled_ax, uncontrolled_ax):
            ax.axhline(1.0, color="#777777", linewidth=0.8, alpha=0.55, zorder=0)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(k_values)
            ax.set_xticklabels(k_tick_labels)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
            ax.tick_params(labelsize=11.5)

    axes[0, 0].set_title(
        "Controlled environment\n(Gold table guaranteed)",
        fontsize=13,
        fontweight="bold",
        color="#333333",
        pad=10,
    )
    axes[0, 1].set_title(
        "Uncontrolled environment\n(Natural BGE-M3 retrieval)",
        fontsize=13,
        fontweight="bold",
        color="#333333",
        pad=10,
    )
    for ax in axes[-1]:
        ax.set_xlabel("Retrieval depth $K$", fontsize=14)

    model_handles = [
        Line2D(
            [0], [0],
            color=model_colors[model],
            marker=model_markers[model],
            linewidth=2,
            markersize=6,
        )
        for model in models
    ]
    condition_handles = [
        Line2D([0], [0], color="#222222", linestyle="-", linewidth=2.2),
        Line2D([0], [0], color="#222222", linestyle="--", linewidth=2.0),
        Line2D([0], [0], color="#222222", linestyle="-.", linewidth=2.2),
    ]
    handles = model_handles + condition_handles
    labels = list(models) + ["Hard negative", "Soft negative", "BGE-M3"]
    if show_sem:
        handles.append(mpatches.Patch(facecolor="#777777", alpha=0.12, edgecolor="none"))
        labels.append("Mean ± SEM")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=4,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#C8C8C8",
        borderpad=0.65,
        fontsize=12,
        handlelength=2.8,
        columnspacing=1.35,
    )
    fig.subplots_adjust(top=0.855, bottom=0.09, left=0.075, right=0.985, hspace=0.28, wspace=0.19)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "table1_new_impact_gap_faceted.png"
    pdf_path = output_dir / "table1_new_impact_gap_faceted.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot table1_.csv as dual-line charts (Hard vs Soft Negatives) with Impact Gap"
    )
    parser.add_argument(
        "--csv",
        default="plots/csv/table1_.csv",
        help="Path to table1_.csv (default: plots/csv/table1_.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default="plots",
        help="Output directory for the figures (default: plots)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Create combined faceted plot instead of separate plots per dataset",
    )
    parser.add_argument(
        "--num-cols",
        type=int,
        default=3,
        help="Number of columns for faceted layout when --combined is used (default: 3)",
    )
    parser.add_argument(
        "--no-plateau",
        action="store_true",
        help="Disable plateau annotation on FeTaQA charts",
    )
    parser.add_argument(
        "--show-sem",
        action="store_true",
        help="Show a Mean ± SEM band around each line",
    )
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    
    df = load_data(csv_path)
    if df.empty:
        raise SystemExit("No valid rows found in CSV")
    
    out_dir = Path(args.out_dir)
    
    if args.combined:
        plot_two_column_workflow(
            df,
            out_dir,
            show_sem=args.show_sem,
        )
    else:
        plot_dual_lines(
            df,
            out_dir,
            show_plateau=not args.no_plateau,
            show_sem=args.show_sem,
        )


if __name__ == "__main__":
    main()
