#!/usr/bin/env python3
"""
100% Stacked Bar Chart: Model Behavioral Profiles

Compares models on their most vulnerable task (Lookup) under hard negatives.
Shows absolute error counts to reveal the behavioral hierarchy:
- Open-source models (Qwen): Extremely vulnerable to deception
- Commercial models (GPT-4o, Claude): Better defenses, prefer refusal over deception

X-axis: Models (Qwen3-32b → GPT-4o → Claude-Haiku-4.5)
Y-axis: Within-reader error type percentage
Color: Error type (same palette as dichotomy figure)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ERROR_TYPES = ["Distractor Extraction", "Premature Refusal", "Reasoning Hallucination"]


def load_and_filter_data(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV and filter for:
    - negative == 'hard' (hard negatives only)
    - question_type == 'Lookup' (most vulnerable task)
    - Aggregate across all datasets
    """
    df = pd.read_csv(csv_path)
    required = {"dataset", "model", "negative", "question_type", "error_type", "count"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if set(df["error_type"].unique()) != set(ERROR_TYPES):
        raise ValueError("CSV must contain exactly the rule_order_v2 error taxonomy")
    
    # Filter strictly
    filtered = df[
        (df["negative"] == "hard") & 
        (df["question_type"] == "Lookup")
    ]
    
    # Aggregate by model and error_type
    agg_df = filtered.groupby(["model", "error_type"])["count"].sum().reset_index()
    
    return agg_df


def create_stacked_bar_chart(agg_df: pd.DataFrame) -> plt.Figure:
    """Create absolute count stacked bar chart with model hierarchy."""
    
    # Color scheme (same as dichotomy figure)
    color_map = {
        "Distractor Extraction": "#E74C3C",  # Red - deception
        "Premature Refusal": "#3498DB",
        "Reasoning Hallucination": "#F39C12",
    }
    
    # Model order aligned with paper definitions.
    model_order = ["qwen3-32b", "gpt-oss-20b", "claude-haiku-4-5", "gpt-4o"]
    model_labels = [
        "Qwen-3-32b",
        "GPT-OSS-20b",
        "Claude-Haiku-4.5",
        "GPT-4o",
    ]
    
    # Get all error types
    error_types = ERROR_TYPES
    
    # Prepare data for plotting
    x_pos = np.arange(len(model_order))
    # Reduce bar width to free space for larger text
    bar_width = 0.25
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Collect data for each model
    model_data = {}
    for model in model_order:
        model_data[model] = {}
        model_subset = agg_df[agg_df["model"] == model]
        
        for error_type in error_types:
            count = model_subset[model_subset["error_type"] == error_type]["count"].sum()
            model_data[model][error_type] = count
    
    # Plot stacked bars
    bottom = np.zeros(len(model_order))
    
    for error_type in error_types:
        heights = []
        for model in model_order:
            total = sum(model_data[model].values())
            count = model_data[model].get(error_type, 0)
            heights.append(count / total * 100 if total else 0)
        
        ax.bar(
            x_pos,
            heights,
            bar_width,
            bottom=bottom,
            label=error_type,
            color=color_map.get(error_type, "#CCCCCC"),
            edgecolor="white",
            linewidth=1.5,
        )
        bottom += np.array(heights)
    
    # Formatting
    ax.set_ylabel("Error Type Distribution (%)",
                  fontsize=19, fontweight="bold")
    # X-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    
    # Grid
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
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
        description="Generate a 100% stacked bar chart for model behavioral profiles"
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
        default="error_anatomy_b_model_profile_rule_order_v2",
        help="Output filename base without extension (default: error_anatomy_b_model_profile)",
    )
    
    args = parser.parse_args()
    
    # Load data
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading data from: {csv_path}")
    agg_df = load_and_filter_data(csv_path)
    
    print(f"\nFiltered data: Hard Negatives + Lookup Tasks")
    print(f"Aggregated {len(agg_df)} model-error_type combinations\n")
    
    # Show summary
    for model in ["qwen3-32b", "gpt-oss-20b", "claude-haiku-4-5", "gpt-4o"]:
        model_total = agg_df[agg_df["model"] == model]["count"].sum()
        distraction = agg_df[(agg_df["model"] == model) & (agg_df["error_type"] == "Distractor Extraction")]["count"].sum()
        print(f"{model} (n={int(model_total):,}):")
        for error_type in ERROR_TYPES:
            count = int(agg_df[(agg_df["model"] == model) & (agg_df["error_type"] == error_type)]["count"].sum())
            pct = count / model_total * 100 if model_total else 0
            print(f"  {error_type}: {count:,} ({pct:.6f}%)")
    
    # Create figure
    print("Generating 100% stacked bar chart...")
    fig = create_stacked_bar_chart(agg_df)
    
    # Save
    save_figure(fig, args.out_dir, args.filename)
    
    print(f"\n✓ Model behavioral profile chart generation complete!")


if __name__ == "__main__":
    main()
