#!/usr/bin/env python3
"""
Cross-Dataset Quadrant Distribution

1x3 faceted stacked bar chart that replaces the Sankey view for Figure X(a).
It compares 4 models across 3 datasets under hard negatives at target_k == 50,
using absolute counts for the four transition quadrants:
- Absolute Robustness (both_success_count)
- Noise-Induced Failure (perfect_to_fail_count)
- Inherent Deficit (both_fail_count)
- Serendipitous (perfect_from_fail_count)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


MODEL_ORDER = ["qwen3-32b", "gpt-oss-20b", "claude-haiku-4-5", "gpt-4o"]
MODEL_LABELS = {
    "qwen3-32b": "Qwen-3-32b",
    "gpt-oss-20b": "GPT-OSS-20b",
    "claude-haiku-4-5": "Claude-Haiku-4.5",
    "gpt-4o": "GPT-4o",
}
DATASET_ORDER = ["e2ewtq", "ottqa", "feta"]
DATASET_LABELS = {
    "e2ewtq": "E2E-WTQ",
    "ottqa": "OTT-QA",
    "feta": "FeTaQA",
}

COLOR_MAP = {
    "Absolute Robustness": "#4C78A8",  # blue
    "Noise-Induced Failure": "#D62728",  # red
    "Inherent Deficit": "#D9D9D9",  # light gray
    "Serendipitous": "#8BC34A",  # light green
}

STACK_ORDER = [
    ("Absolute Robustness", "both_success_count"),
    ("Noise-Induced Failure", "perfect_to_fail_count"),
    ("Inherent Deficit", "both_fail_count"),
    ("Serendipitous", "perfect_from_fail_count"),
]


def load_and_filter_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    filtered = df[(df["target_k"] == 50) & (df["negative"] == "hard")].copy()
    return filtered


def build_panel_data(filtered: pd.DataFrame, dataset: str) -> dict:
    panel = filtered[filtered["dataset"] == dataset]
    data = {}
    for model in MODEL_ORDER:
        model_rows = panel[panel["model"] == model]
        data[model] = {
            metric_name: int(model_rows[column].sum())
            for metric_name, column in STACK_ORDER
        }
    return data


def format_axis(ax: plt.Axes, max_total: int, percent: bool = False) -> None:
    """Format y-axis either as counts or percentages.

    If percent is True, formats ticks as percentages (0-100%).
    """
    if percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=20)
    else:
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        ax.set_ylim(0, max_total * 1.14 if max_total > 0 else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)


def add_total_labels(ax: plt.Axes, x_positions: np.ndarray, totals: list[int]) -> None:
    for x_pos, total in zip(x_positions, totals):
        ax.text(
            x_pos,
            total + max(8, total * 0.015),
            f"{total:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )


def create_figure(filtered: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0), sharey=False)

    x_positions = np.arange(len(MODEL_ORDER))
    bar_width = 0.64

    for ax, dataset in zip(axes, DATASET_ORDER):
        panel_data = build_panel_data(filtered, dataset)
        totals = [sum(panel_data[model].values()) for model in MODEL_ORDER]
        bottom = np.zeros(len(MODEL_ORDER))

        # Normalize counts to percentages per model so each stacked bar sums to 100%
        for metric_name, _column in STACK_ORDER:
            heights = []
            for i, model in enumerate(MODEL_ORDER):
                total = totals[i]
                if total > 0:
                    val = panel_data[model][metric_name] / total * 100.0
                else:
                    val = 0.0
                heights.append(val)

            ax.bar(
                x_positions,
                heights,
                width=bar_width,
                bottom=bottom,
                color=COLOR_MAP[metric_name],
                edgecolor="white",
                linewidth=1.2,
                label=metric_name,
            )
            bottom += np.array(heights)

        ax.set_title(DATASET_LABELS[dataset], fontsize=18, fontweight="bold", pad=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [MODEL_LABELS[model] for model in MODEL_ORDER],
            fontsize=12,
            rotation=20,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
        # Show percentages on y-axis
        format_axis(ax, int(max(totals)), percent=True)
        ax.tick_params(axis='x', labelsize=12)

        if ax is axes[0]:
            ax.set_ylabel("Percentage (%)", fontsize=22, fontweight="bold")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP[metric_name])
        for metric_name, _column in STACK_ORDER
    ]
    labels = [metric_name for metric_name, _column in STACK_ORDER]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 1.00),
        fontsize=20,
    )

    plt.tight_layout(rect=[0.02, 0.06, 1, 0.90])
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, filename_base: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{filename_base}.png"
    pdf_path = output_dir / f"{filename_base}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


def print_summary(filtered: pd.DataFrame) -> None:
    print("=== Filtered Summary: target_k == 50 and negative == 'hard' ===")
    for dataset in DATASET_ORDER:
        print(f"\n{DATASET_LABELS[dataset]}:")
        panel = filtered[filtered["dataset"] == dataset]
        for model in MODEL_ORDER:
            model_rows = panel[panel["model"] == model]
            total = int(model_rows["total"].sum())
            robust = int(model_rows["both_success_count"].sum())
            fail = int(model_rows["perfect_to_fail_count"].sum())
            deficit = int(model_rows["both_fail_count"].sum())
            serendip = int(model_rows["perfect_from_fail_count"].sum())
            print(
                f"  {MODEL_LABELS[model]:16s} total={total:4d}  robust={robust:4d}  "
                f"failure={fail:4d}  deficit={deficit:4d}  serendip={serendip:4d}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 1x3 faceted stacked bar chart for cross-dataset quadrant distribution"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="plots/csv/Error Analysis - Distribution.csv",
        help="Path to Error Analysis - Distribution CSV file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory (default: plots)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="error_quadrant_breakdown",
        help="Output filename base without extension (default: error_quadrant_breakdown)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading data from: {csv_path}")
    filtered = load_and_filter_data(csv_path)
    print_summary(filtered)

    fig = create_figure(filtered)
    save_figure(fig, args.out_dir, args.filename)
    print("\n✓ Cross-dataset quadrant breakdown complete!")


if __name__ == "__main__":
    main()
