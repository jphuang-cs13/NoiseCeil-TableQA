#!/usr/bin/env python3
"""
Hard Negative Tax plot for table4_.csv

Chart: "Hard Negative Tax" (grouped shaded line chart)
- X: K (retrieval depth)
- Y: Cost per Success (CpS, $)
- Columns are datasets; rows group proprietary and open-weight model tiers.
- For each model: plot Soft, Hard, and BGE-M3 (real retriever) CpS trends.
- Fill the area between Soft and Hard CpS with a semi-transparent patch to show
  the hard-negative tax.

Expected CSV columns (case-sensitive-ish):
- Required: `Dataset`, `Model`, `K`, `CpS(Soft)`, `CpS(Hard)`,
  and `CpS(BGE-m3)`.
"""

import argparse
from pathlib import Path
import math
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


def find_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_table4_columns(df: pd.DataFrame) -> tuple[str, str, str | None]:
    """Resolve the CpS columns, preferring the refreshed table4 schema."""
    soft_col = find_column(
        df,
        ["CpS(Soft)", "Cost per Success(soft)", "CpS(soft)", "CpS(Soft mean)"]
    )
    hard_col = find_column(
        df,
        ["CpS(Hard)", "Cost per Success(hard)", "CpS(hard)", "CpS(Hard mean)"]
    )
    real_col = find_column(
        df,
        ["CpS(BGE-m3)", "CpS(BGE-M3)", "CpS(Real)", "CpS(real)", "Cost per Success(real)"]
    )

    if soft_col is None or hard_col is None or real_col is None:
        raise SystemExit(
            "CSV missing required CpS columns; expected CpS(Soft), CpS(Hard), "
            "and CpS(BGE-m3) or legacy variants."
        )

    return soft_col, hard_col, real_col


def load_table4(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Detect the schema used by the current CSV.
    k_col = find_column(df, ["K", "k", "depth"])
    dataset_col = find_column(df, ["Dataset", "dataset"])
    model_col = find_column(df, ["Model", "model"])
    soft_col, hard_col, real_col = resolve_table4_columns(df)

    missing = [
        name
        for name, val in [
            ("Dataset", dataset_col),
            ("K", k_col),
            ("Model", model_col),
            ("soft cost", soft_col),
            ("hard cost", hard_col),
        ]
        if val is None
    ]
    if missing:
        raise SystemExit(f"CSV is missing required columns (tried variants). Missing: {missing}")

    selected_cols = [dataset_col, model_col, k_col, soft_col, hard_col]
    selected_names = ["Dataset", "Model", "K", "soft_cps", "hard_cps"]
    selected_cols.append(real_col)
    selected_names.append("real_cps")

    df = df[selected_cols].copy()
    df.columns = selected_names

    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["soft_cps"] = pd.to_numeric(df["soft_cps"], errors="coerce")
    df["hard_cps"] = pd.to_numeric(df["hard_cps"], errors="coerce")
    df["real_cps"] = pd.to_numeric(df["real_cps"], errors="coerce")

    df = df.dropna(
        subset=["Dataset", "Model", "K", "soft_cps", "hard_cps", "real_cps"]
    )
    return df


def plot_hard_negative_tax(df: pd.DataFrame, out_dir: Path) -> None:
    ks = sorted(df["K"].unique())
    dataset_order = ["E2E-WTQ", "OTTQA", "FeTaQA"]
    datasets = [d for d in dataset_order if d in df["Dataset"].unique()]
    if not datasets:
        datasets = sorted(df["Dataset"].unique())

    model_order = [m for m in ["gpt-4o", "claude-haiku-4-5", "gpt-oss-20b", "qwen3-32b"] if m in df["Model"].unique()]
    if not model_order:
        model_order = sorted(df["Model"].unique())

    colors = {
        "gpt-4o": "#56B4E9",
        "claude-haiku-4-5": "#009E73",
        "gpt-oss-20b": "#E69F00",
        "qwen3-32b": "#D55E00",
    }

    ncols = len(datasets)
    nrows = 1
    cell_size = 5
    total_w = cell_size * ncols
    total_h = cell_size * nrows
    base = max(total_w, total_h)
    fig, axes = plt.subplots(1, ncols, figsize=(base * 1.18, base * 0.80), sharex=True)
    if ncols == 1:
        axes = [axes]

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        dataset_df = df[df["Dataset"] == dataset]

        for model in model_order:
            mdl = dataset_df[dataset_df["Model"] == model]
            if mdl.empty:
                continue

            # Reindex by K to ensure ordered x; explicitly aggregate numeric columns only.
            agg_cols = ["soft_cps", "hard_cps", "real_cps"]
            grouped = mdl.groupby("K", as_index=True)[agg_cols].mean()
            y_soft = [grouped["soft_cps"].get(k, math.nan) for k in ks]
            y_hard = [grouped["hard_cps"].get(k, math.nan) for k in ks]
            y_real = [grouped["real_cps"].get(k, math.nan) for k in ks]

            color = colors.get(model, "#666666")

            ax.plot(
                ks,
                y_soft,
                label=f"{MODEL_DISPLAY_NAMES.get(model, model)} soft",
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=5,
                zorder=4,
            )

            ax.plot(
                ks,
                y_hard,
                label=f"{MODEL_DISPLAY_NAMES.get(model, model)} hard",
                color=color,
                linewidth=1.8,
                linestyle=":",
                marker="s",
                markersize=4,
                alpha=0.9,
                zorder=3,
            )

            y1 = [s if not math.isnan(s) else None for s in y_soft]
            y2 = [h if not math.isnan(h) else None for h in y_hard]

            ax.fill_between(
                ks,
                y1,
                y2,
                where=[(not math.isnan(a)) and (not math.isnan(b)) for a, b in zip(y1, y2)],
                interpolate=True,
                color=color,
                alpha=0.22,
                linewidth=0,
                zorder=2,
            )

            # Real retriever CpS: dashed line, usually above hard CpS.
            ax.plot(
                ks,
                y_real,
                color=color,
                linewidth=1.8,
                linestyle="--",
                marker="x",
                markersize=4,
                alpha=0.95,
                zorder=5,
            )

            # Highlight additional tax from hard -> real retriever when real is worse.
            ax.fill_between(
                ks,
                y_hard,
                y_real,
                where=[
                    (not math.isnan(h)) and (not math.isnan(r)) and (r >= h)
                    for h, r in zip(y_hard, y_real)
                ],
                interpolate=True,
                color=color,
                alpha=0.08,
                linewidth=0,
                zorder=1,
            )

        ax.set_title(dataset, fontsize=44, pad=8, fontweight="bold")
        ax.set_xlabel("Retrieval depth K", fontsize=42)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(int(k)) for k in ks], fontsize=48)
        ax.tick_params(axis="y", labelsize=48)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        if i == 0:
            ax.set_ylabel("Cost per Success (CpS, $)", fontsize=52)

    # Shared legend: model colors + line-style meaning + shaded tax area.
    legend_line_handles = [
        plt.Line2D([0], [0], color=colors.get(model, "#666666"), linewidth=2.0, marker="o", markersize=5, label=MODEL_DISPLAY_NAMES.get(model, model))
        for model in model_order
    ]
    soft_handle = plt.Line2D([0], [0], color="#444444", linewidth=2.2, marker="o", markersize=5, label="Soft CpS")
    hard_handle = plt.Line2D([0], [0], color="#444444", linewidth=1.8, linestyle=":", marker="s", markersize=4, label="Hard CpS")
    tax_patch = mpatches.Patch(facecolor="#999999", alpha=0.22, label="Hard Negative Tax (CpS difference)")
    real_style_handle = plt.Line2D(
        [0], [0],
        color="#444444",
        linewidth=1.8,
        linestyle="--",
        marker="x",
        markersize=5,
        label="BGE-M3 (Real retriever) CpS",
    )
    legend_items = legend_line_handles + [soft_handle, hard_handle, tax_patch, real_style_handle]
    legend = fig.legend(
        handles=legend_items,
        loc="upper center",
        ncol=max(1, math.ceil(len(legend_items) / 2)),
        frameon=True,
        fontsize=36,
        bbox_to_anchor=(0.5, 1.12),
        markerscale=2.6,
        handlelength=3.6,
        handletextpad=0.6,
    )

    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_box = legend.get_window_extent(renderer=renderer)
        legend_frac = legend_box.width / fig.bbox.width
        axes_total_frac = sum(ax.get_position().width for ax in axes if ax.get_visible())

        if axes_total_frac > 0 and legend_frac > 0:
            scale = legend_frac / axes_total_frac
            scale = max(1.0, min(scale, 2.0))
            if scale > 1.01:
                new_width = fig.get_size_inches()[0] * scale
                fig.set_size_inches(new_width, fig.get_size_inches()[1], forward=True)
                fig.canvas.draw()
    except Exception:
        pass

    fig.tight_layout(rect=[0.01, 0, 0.99, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "table4_new_hard_negative_tax.png"
    pdf = out_dir / "table4_new_hard_negative_tax.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def plot_hard_negative_tax_grouped(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot Figure 4 as dataset columns and model-tier rows."""
    ks = sorted(df["K"].unique())
    preferred_datasets = ["E2E-WTQ", "OTTQA", "FeTaQA"]
    datasets = [d for d in preferred_datasets if d in df["Dataset"].unique()]
    if not datasets:
        datasets = sorted(df["Dataset"].unique())

    # Okabe-Ito high-contrast, colorblind-safe palette requested for the paper.
    colors = {
        "gpt-4o": "#56B4E9",            # sky blue
        "claude-haiku-4-5": "#009E73", # bluish green
        "gpt-oss-20b": "#E69F00",      # orange
        "qwen3-32b": "#D55E00",        # vermillion
    }
    display_names = MODEL_DISPLAY_NAMES
    requested_tiers = {
        "Proprietary models": ["gpt-4o", "claude-haiku-4-5"],
        "Open-weight models": ["gpt-oss-20b", "qwen3-32b"],
    }
    present_models = set(df["Model"].unique())
    tiers = {
        tier: [model for model in models if model in present_models]
        for tier, models in requested_tiers.items()
        if any(model in present_models for model in models)
    }

    fig, axes = plt.subplots(
        len(tiers), len(datasets), figsize=(15.5, 9.0),
        sharex=True, squeeze=False,
    )

    for row, (tier, models) in enumerate(tiers.items()):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            dataset_df = df[df["Dataset"] == dataset]

            for model in models:
                mdl = dataset_df[dataset_df["Model"] == model]
                if mdl.empty:
                    continue

                value_cols = ["soft_cps", "hard_cps", "real_cps"]
                grouped = mdl.groupby("K", as_index=True)[value_cols].mean()
                y_soft = [grouped["soft_cps"].get(k, math.nan) for k in ks]
                y_hard = [grouped["hard_cps"].get(k, math.nan) for k in ks]
                y_real = [grouped["real_cps"].get(k, math.nan) for k in ks]

                color = colors.get(model, "#666666")
                ax.plot(
                    ks, y_soft, color=color, linewidth=2.2,
                    marker="o", markersize=4.5, zorder=4,
                )
                ax.plot(
                    ks, y_hard, color=color, linewidth=2.0,
                    linestyle=":", marker="s", markersize=4.2, zorder=3,
                )
                ax.fill_between(
                    ks, y_soft, y_hard,
                    where=[
                        not math.isnan(soft) and not math.isnan(hard)
                        for soft, hard in zip(y_soft, y_hard)
                    ],
                    interpolate=True, color=color, alpha=0.16,
                    linewidth=0, zorder=2,
                )
                ax.plot(
                    ks, y_real, color=color, linewidth=1.8,
                    linestyle="--", marker="x", markersize=5, zorder=5,
                )

            if row == 0:
                ax.set_title(dataset, fontsize=15, pad=8, fontweight="bold")
            if row == len(tiers) - 1:
                ax.set_xlabel("Retrieval depth $K$", fontsize=15)
            if col == 0:
                ax.set_ylabel(
                    f"{tier}\nCpS ($)", fontsize=15, fontweight="bold"
                )
            ax.set_xticks(ks)
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
            ax.set_axisbelow(True)
            ax.margins(x=0.03, y=0.08)

    all_models = [model for models in tiers.values() for model in models]
    model_handles = [
        plt.Line2D(
            [0], [0], color=colors[model], linewidth=3,
            label=display_names.get(model, model),
        )
        for model in all_models
    ]
    condition_handles = [
        plt.Line2D(
            [0], [0], color="#333333", linewidth=2.2,
            marker="o", markersize=5, label="Soft",
        ),
        plt.Line2D(
            [0], [0], color="#333333", linewidth=2.0,
            linestyle=":", marker="s", markersize=5, label="Hard",
        ),
        plt.Line2D(
            [0], [0], color="#333333", linewidth=1.8,
            linestyle="--", marker="x", markersize=5,
            label="BGE-M3",
        ),
        mpatches.Patch(
            facecolor="#777777", alpha=0.16,
            label="Hard-negative tax (Hard - Soft CpS gap)",
        ),
    ]
    fig.legend(
        handles=model_handles + condition_handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="#B0B0B0",
        fontsize=13,
        bbox_to_anchor=(0.5, 1.005),
        borderpad=0.8,
        columnspacing=1.8,
        handlelength=2.8,
        handletextpad=0.7,
        markerscale=1.25,
    )
    fig.tight_layout(rect=[0.025, 0.025, 0.995, 0.88], h_pad=1.2, w_pad=1.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "table4_new_hard_negative_tax.png"
    pdf = out_dir / "table4_new_hard_negative_tax.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main():
    parser = argparse.ArgumentParser(description="Plot table4 hard-negative tax chart")
    parser.add_argument("--csv", default="plots/csv/table4_.csv", help="Path to table4_.csv")
    parser.add_argument("--out-dir", default="plots", help="Output directory for figures")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    df = load_table4(csv_path)
    if df.empty:
        raise SystemExit("No valid data found in CSV")

    plot_hard_negative_tax_grouped(df, out_dir)


if __name__ == "__main__":
    main()
