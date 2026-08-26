#!/usr/bin/env python3
"""
Plot table3_.csv as an absolute variance bar chart.

Design goals:
- Avoid the traditional Start-Middle-End line chart entirely.
- Show a separate panel for each dataset so within-dataset differences are clear.
- Use retrieval depth K on the x-axis in every panel.
- Compare Soft Negative vs Hard Negative with grouped bars.
- Use a consistent, color-blind-friendly model palette.

The CSV contains multiple K values per dataset/model. This script keeps those
K values separate instead of averaging them.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

MODEL_DISPLAY_NAMES = {
    "gpt-4o": "GPT-4o",
    "gpt-oss-20b": "GPT-OSS-20b",
    "claude-haiku-4-5": "Claude-Haiku-4.5",
    "qwen3-32b": "Qwen-3-32b",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "PingFang TC",
            "Hiragino Sans GB",
            "AppleGothic",
            "Arial Unicode MS",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
    }
)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and normalize table3_.csv data."""
    df = pd.read_csv(csv_path)

    required_cols = [
        "Dataset",
        "Model",
        "K",
        "Absolute Variance(soft)",
        "Absolute Variance(hard)",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise SystemExit(f"CSV missing required columns: {missing_cols}")

    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["Absolute Variance(soft)"] = pd.to_numeric(df["Absolute Variance(soft)"], errors="coerce")
    df["Absolute Variance(hard)"] = pd.to_numeric(df["Absolute Variance(hard)"], errors="coerce")

    df = df.dropna(subset=["Dataset", "Model", "K", "Absolute Variance(soft)", "Absolute Variance(hard)"])
    return df


def summarize_variance_by_k(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize absolute variance for each Dataset × Model × K combination."""
    summary = (
        df.groupby(["Dataset", "Model", "K"], as_index=False)
        .agg(
            {
                "Absolute Variance(soft)": "mean",
                "Absolute Variance(hard)": "mean",
            }
        )
        .rename(
            columns={
                "Absolute Variance(soft)": "soft_variance",
                "Absolute Variance(hard)": "hard_variance",
            }
        )
    )
    return summary


def plot_dataset_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    dataset: str,
    k_order: list[int],
    model_order: list[str],
    model_colors: dict[str, str],
) -> None:
    """Plot one dataset panel, grouped by K, model, and negative type."""
    dataset_subset = summary[summary["Dataset"] == dataset]
    ks = [k for k in k_order if k in dataset_subset["K"].unique()]
    models = [model for model in model_order if model in dataset_subset["Model"].unique()]

    bar_width = 0.085
    pair_width = bar_width * 2
    model_gap = 0.025
    group_width = len(models) * pair_width + (len(models) - 1) * model_gap
    condition_specs = [
        ("soft_variance", -bar_width / 2, ""),
        ("hard_variance", bar_width / 2, "////"),
    ]
    k_centers = list(range(len(ks)))

    for k_index, k_value in enumerate(ks):
        k_rows = dataset_subset[dataset_subset["K"] == k_value]
        group_start = k_centers[k_index] - group_width / 2
        for model_index, model in enumerate(models):
            row = k_rows[k_rows["Model"] == model]
            if row.empty:
                continue

            row = row.iloc[0]
            model_color = model_colors.get(model, "#666666")
            model_center = (
                group_start
                + model_index * (pair_width + model_gap)
                + pair_width / 2
            )

            for column_name, offset, hatch in condition_specs:
                value = float(row[column_name])
                ax.bar(
                    model_center + offset,
                    value,
                    width=bar_width,
                    color=model_color,
                    edgecolor="#333333",
                    linewidth=0.55,
                    hatch=hatch,
                    zorder=3,
                )

    panel_max = max(
        dataset_subset["hard_variance"].max(),
        dataset_subset["soft_variance"].max(),
    )
    ax.set_xticks(k_centers)
    ax.set_xticklabels([str(int(k)) for k in ks])
    ax.set_xlabel("Retrieval depth (K)", fontsize=16, labelpad=6)
    ax.set_ylim(0, panel_max * 1.16)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(dataset, fontsize=18, fontweight="bold", pad=9)
    ax.tick_params(axis="both", labelsize=13)


def plot_absolute_variance_bars(
    summary: pd.DataFrame,
    output_dir: Path,
    show_spike_annotation: bool = False,
) -> None:
    """Plot absolute variance bars with one enlarged panel per dataset."""
    k_order = [30, 40, 50]
    ks = [k for k in k_order if k in summary["K"].unique()]
    if not ks:
        ks = sorted(summary["K"].unique())

    dataset_order = ["E2E-WTQ", "OTTQA", "OTT-QA", "FeTaQA"]
    model_order = ["gpt-4o", "gpt-oss-20b", "claude-haiku-4-5", "qwen3-32b"]

    model_colors = {
        "gpt-4o": "#56B4E9",
        "gpt-oss-20b": "#E69F00",
        "claude-haiku-4-5": "#009E73",
        "qwen3-32b": "#D55E00",
    }

    model_handles = [
        mpatches.Patch(facecolor=model_colors.get(model, "#666666"), edgecolor=model_colors.get(model, "#666666"), label=MODEL_DISPLAY_NAMES.get(model, model))
        for model in model_order
    ]
    condition_handles = [
        mpatches.Patch(facecolor="white", edgecolor="#333333", label="Soft negative"),
        mpatches.Patch(facecolor="white", edgecolor="#333333", hatch="////", label="Hard negative"),
    ]

    combined_handles = model_handles + condition_handles
    combined_labels = [MODEL_DISPLAY_NAMES.get(model, model) for model in model_order] + ["Soft negative", "Hard negative"]

    datasets = [d for d in dataset_order if d in summary["Dataset"].unique()]
    if not datasets:
        datasets = sorted(summary["Dataset"].unique())

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.2 * len(datasets), 5.4),
        sharey=False,
    )
    if len(datasets) == 1:
        axes = [axes]

    for axis, dataset in zip(axes, datasets):
        plot_dataset_panel(
            axis,
            summary,
            dataset,
            ks,
            model_order,
            model_colors,
        )

    axes[0].set_ylabel("Absolute Variance", fontsize=17, labelpad=6)

    legend = fig.legend(
        handles=combined_handles,
        labels=combined_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(combined_handles),
        frameon=True,
        fancybox=False,
        edgecolor="#555555",
        framealpha=1.0,
        fontsize=13,
        borderpad=0.55,
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    fig.tight_layout(rect=[0.01, 0, 0.99, 0.90], w_pad=2.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "table3_new_absolute_variance_bar.png"
    pdf_path = output_dir / "table3_new_absolute_variance_bar.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot table3_.csv as an absolute variance bar chart"
    )
    parser.add_argument(
        "--csv",
        default="plots/csv/table3_.csv",
        help="Path to table3_.csv (default: plots/csv/table3_.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default="plots",
        help="Output directory for the figures (default: plots)",
    )
    parser.add_argument(
        "--show-spike-annotation",
        action="store_true",
        help="Show the FeTaQA hard spike callout annotation",
    )
    parser.add_argument(
        "--k-values",
        default=None,
        help="Comma-separated K values to plot (default: all K values in the CSV)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = load_data(csv_path)
    if df.empty:
        raise SystemExit("No valid rows found in CSV")

    summary = summarize_variance_by_k(df)
    if summary.empty:
        raise SystemExit("No summarized rows available for plotting")

    if args.k_values is not None:
        requested_ks = []
        for raw_value in args.k_values.split(","):
            raw_value = raw_value.strip()
            if not raw_value:
                continue
            requested_ks.append(int(raw_value))
    else:
        requested_ks = None

    if requested_ks is not None:
        summary = summary[summary["K"].isin(requested_ks)]
        if summary.empty:
            raise SystemExit(f"No rows found for K values: {requested_ks}")

    output_dir = Path(args.out_dir)
    plot_absolute_variance_bars(
        summary,
        output_dir,
        show_spike_annotation=args.show_spike_annotation,
    )


if __name__ == "__main__":
    main()
