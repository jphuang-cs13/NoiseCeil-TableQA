#!/usr/bin/env python3
"""
100% Stacked Bar Chart: The Dichotomy of Distraction

Compares error type distributions between hard and soft negatives.
Shows that hard negatives actively mislead models (high Distractor Extraction),
while soft negatives produce relatively more refusal and reasoning failures.

X-axis (two levels):
  - Outer: negative type (Hard Negatives, Soft Negatives)
  - Inner: dataset (E2E-WTQ, OTT-QA, FeTaQA)

Y-axis: Error type percentage (0-100% stacked)

Color scheme:
  - Red: Distractor Extraction (dangerous deception)
  - Blue: Premature Refusal
  - Orange: Reasoning Hallucination
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ERROR_TYPES = ["Distractor Extraction", "Premature Refusal", "Reasoning Hallucination"]


def load_and_aggregate_data(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV and aggregate across all models and question types.
    
    Returns DataFrame with columns:
    - negative (hard/soft)
    - dataset (e2ewtq/feta/ottqa)
    - error_type (the three rule_order_v2 labels)
    - count (summed across all models and question types)
    """
    df = pd.read_csv(csv_path)
    required = {"dataset", "model", "negative", "question_type", "error_type", "count"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if set(df["error_type"].unique()) != set(ERROR_TYPES):
        raise ValueError("CSV must contain exactly the rule_order_v2 error taxonomy")
    
    # Aggregate: sum counts by negative, dataset, and error_type
    # (This sums across all models and question_types)
    agg_df = df.groupby(["negative", "dataset", "error_type"])["count"].sum().reset_index()
    
    return agg_df


def prepare_plot_data(agg_df: pd.DataFrame):
    """
    Prepare data for 100% stacked bar chart.
    
    Returns:
    - negatives: list of negative types
    - datasets: list of dataset names  
    - all_error_types: list of error types (for legend)
    """
    # Define order of negative types and datasets
    negatives = ["hard", "soft"]
    datasets = ["e2ewtq", "ottqa", "feta"]
    
    # Get all error types
    all_error_types = ERROR_TYPES
    
    return negatives, datasets, all_error_types


def create_100_stacked_bar_chart(
    agg_df: pd.DataFrame,
) -> plt.Figure:
    """Create 100% stacked bar chart."""
    
    # Color scheme - carefully chosen to distinguish error types
    color_map = {
        "Distractor Extraction": "#E74C3C",  # Red - dangerous deception
        "Premature Refusal": "#3498DB",
        "Reasoning Hallucination": "#F39C12",
    }
    
    # Setup: Hard and Soft negatives, each with 3 datasets
    negatives = ["hard", "soft"]
    datasets_order = ["e2ewtq", "ottqa", "feta"]
    dataset_labels = ["E2E-WTQ", "OTT-QA", "FeTaQA"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Prepare bar positions
    # Reduce bar width to free space for larger text
    bar_width = 0.3
    # Reduce gap between hard/soft groups slightly
    group_gap = 0.2

    # Hard negatives: positions 0, 1, 2
    # Soft negatives start with a small separation gap.
    x_hard = np.arange(3)
    x_soft = np.arange(3) + 3 + group_gap
    separator_x = (x_hard[-1] + x_soft[0]) / 2.0
    hard_center = float(np.mean(x_hard))
    soft_center = float(np.mean(x_soft))
    
    # Collect data for each bar
    all_bars_data = []
    all_bars_labels = []
    
    # Hard negatives
    for i, ds in enumerate(datasets_order):
        bar_labels = []
        bar_values = []
        
        subset = agg_df[(agg_df["negative"] == "hard") & (agg_df["dataset"] == ds)]
        total = subset["count"].sum()
        
        for error_type in ERROR_TYPES:
            error_count = subset[subset["error_type"] == error_type]["count"].sum()
            pct = (error_count / total * 100) if total > 0 else 0
            bar_values.append(pct)
            bar_labels.append(error_type)
        
        all_bars_data.append((x_hard[i], bar_values, bar_labels))
    
    # Soft negatives
    for i, ds in enumerate(datasets_order):
        bar_labels = []
        bar_values = []
        
        subset = agg_df[(agg_df["negative"] == "soft") & (agg_df["dataset"] == ds)]
        total = subset["count"].sum()
        
        for error_type in ERROR_TYPES:
            error_count = subset[subset["error_type"] == error_type]["count"].sum()
            pct = (error_count / total * 100) if total > 0 else 0
            bar_values.append(pct)
            bar_labels.append(error_type)
        
        all_bars_data.append((x_soft[i], bar_values, bar_labels))
    
    # Plot bars with stacking
    error_types = ERROR_TYPES
    
    bottom = np.zeros(6)  # 6 bars total
    x_positions = np.concatenate([x_hard, x_soft])
    
    for error_type in error_types:
        heights = []
        
        # Hard negatives
        for ds in datasets_order:
            subset = agg_df[(agg_df["negative"] == "hard") & (agg_df["dataset"] == ds)]
            total = subset["count"].sum()
            error_count = subset[subset["error_type"] == error_type]["count"].sum()
            pct = (error_count / total * 100) if total > 0 else 0
            heights.append(pct)
        
        # Soft negatives
        for ds in datasets_order:
            subset = agg_df[(agg_df["negative"] == "soft") & (agg_df["dataset"] == ds)]
            total = subset["count"].sum()
            error_count = subset[subset["error_type"] == error_type]["count"].sum()
            pct = (error_count / total * 100) if total > 0 else 0
            heights.append(pct)
        
        ax.bar(
            x_positions,
            heights,
            bar_width,
            bottom=bottom,
            label=error_type,
            color=color_map.get(error_type, "#CCCCCC"),
            edgecolor="white",
            linewidth=1,
        )
        bottom += np.array(heights)
    
    # Formatting
    ax.set_ylabel("Error Type Distribution (%)", fontsize=22, fontweight="bold")
    ax.set_ylim(0, 100)
    
    # X-axis labels
    all_labels = dataset_labels + dataset_labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_labels, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    
    # Add group labels for hard and soft
    ax.text(hard_center, -15, "Hard Negatives", ha="center", fontsize=22, fontweight="bold")
    ax.text(soft_center, -15, "Soft Negatives", ha="center", fontsize=22, fontweight="bold")
    
    # Add separation line
    ax.axvline(separator_x, color="black", linestyle="--", linewidth=2, alpha=0.4)
    
    # Grid
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_yticks(np.arange(0, 101, 10))
    
    # Legend: horizontal layout below the title.
    handles, labels = ax.get_legend_handles_labels()
    mapped_labels = labels
    # Place legend above the axes (outside the plotting area) to avoid overlap
    fig.legend(
        handles,
        mapped_labels,
        loc="upper center",
        # lower the legend slightly to reduce whitespace between legend and chart
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(mapped_labels), 4),
        fontsize=19,
        frameon=True,
        fancybox=True,
        shadow=False,
    )

    # Move plotting area down moderately to make space for the legend above
    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    
    return fig


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    filename_base: str,
) -> None:
    """Save figure in both PNG and PDF formats."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PNG
    png_path = output_dir / f"{filename_base}.png"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    print(f"Saved PNG: {png_path}")
    
    # PDF
    pdf_path = output_dir / f"{filename_base}.pdf"
    fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight")
    print(f"Saved PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 100% stacked bar chart comparing hard vs soft negative error types"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="artifacts/error_analysis/Error Analysis - Error Type_v2.csv",
        help="Path to Error Analysis - Error Type CSV file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reproduced_figures"),
        help="Output directory (default: plots)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="error_anatomy_a_dichotomy_rule_order_v2",
        help="Output filename base without extension (default: error_anatomy_a_dichotomy)",
    )
    
    args = parser.parse_args()
    
    # Load data
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading data from: {csv_path}")
    agg_df = load_and_aggregate_data(csv_path)

    print("\nExact plotted segments (count and within-bar percentage):")
    for negative in ["hard", "soft"]:
        for dataset in ["e2ewtq", "ottqa", "feta"]:
            subset = agg_df[(agg_df["negative"] == negative) & (agg_df["dataset"] == dataset)]
            total = int(subset["count"].sum())
            print(f"  {negative} / {dataset} (n={total:,})")
            for error_type in ERROR_TYPES:
                count = int(subset.loc[subset["error_type"] == error_type, "count"].sum())
                print(f"    {error_type}: {count:,} ({count / total * 100:.6f}%)")
    
    print(f"\nAggregated {len(agg_df)} error type categories across all models")
    
    # Prepare plot data
    negatives, datasets, error_types = prepare_plot_data(agg_df)
    
    print(f"\nError types found:")
    for et in error_types:
        print(f"  - {et}")
    
    # Create figure
    print(f"\nGenerating 100% stacked bar chart...")
    fig = create_100_stacked_bar_chart(agg_df)
    
    # Save
    save_figure(fig, args.out_dir, args.filename)
    
    print(f"\n✓ 100% stacked bar chart generation complete!")


if __name__ == "__main__":
    main()
