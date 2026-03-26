"""Multi-omic summary visualization.

Generates publication-quality figures summarizing resilience findings
across all omic layers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import log


def plot_cohort_overview(cohort_df: pd.DataFrame, output_path: Path) -> None:
    """Plot cohort demographics and pathology distributions."""
    df = cohort_df[cohort_df["cohort_group"] != "excluded"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Group sizes
    counts = df["cohort_group"].value_counts()
    colors = {"resilient": "#2ecc71", "AD": "#e74c3c", "control": "#3498db"}
    axes[0, 0].bar(counts.index, counts.values, color=[colors.get(g, "grey") for g in counts.index])
    axes[0, 0].set_title("Group Sizes")
    axes[0, 0].set_ylabel("N subjects")

    # Age distribution
    for group in ["resilient", "AD", "control"]:
        vals = df.loc[df["cohort_group"] == group, "age_death"].dropna()
        if len(vals) > 0:
            axes[0, 1].hist(vals, alpha=0.5, label=group, color=colors.get(group), bins=20)
    axes[0, 1].set_title("Age at Death")
    axes[0, 1].legend()

    # Braak distribution
    for group in ["resilient", "AD", "control"]:
        vals = df.loc[df["cohort_group"] == group, "braaksc"].dropna()
        if len(vals) > 0:
            axes[0, 2].hist(vals, alpha=0.5, label=group, color=colors.get(group), bins=7)
    axes[0, 2].set_title("Braak Stage")

    # CERAD distribution
    for group in ["resilient", "AD", "control"]:
        vals = df.loc[df["cohort_group"] == group, "ceradsc"].dropna()
        if len(vals) > 0:
            axes[1, 0].hist(vals, alpha=0.5, label=group, color=colors.get(group), bins=5)
    axes[1, 0].set_title("CERAD Score")

    # Cognitive score (MMSE)
    if "cts_mmse30_lv" in df.columns:
        for group in ["resilient", "AD", "control"]:
            vals = df.loc[df["cohort_group"] == group, "cts_mmse30_lv"].dropna()
            if len(vals) > 0:
                axes[1, 1].hist(vals, alpha=0.5, label=group, color=colors.get(group), bins=20)
        axes[1, 1].set_title("MMSE Score (Last Visit)")

    # Resilience index
    if "cogng_path_slope" in df.columns:
        for group in ["resilient", "AD", "control"]:
            vals = df.loc[df["cohort_group"] == group, "cogng_path_slope"].dropna()
            if len(vals) > 0:
                axes[1, 2].hist(vals, alpha=0.5, label=group, color=colors.get(group), bins=20)
        axes[1, 2].set_title("Resilience Index (cogng_path_slope)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved cohort overview: {output_path}")


def plot_multi_omic_summary(
    convergent_signature: pd.DataFrame,
    output_path: Path,
    top_n: int = 30,
) -> None:
    """Plot multi-omic convergence heatmap.

    Shows which genes appear across which omic layers, sorted by
    number of supporting layers and Fisher meta-analysis p-value.
    """
    if convergent_signature.empty:
        log.warning("No convergent signature to plot")
        return

    top = convergent_signature.head(top_n)

    # Build layer presence matrix
    all_layers = set()
    for layers_str in top["layers"]:
        all_layers.update(layers_str.split(";"))
    all_layers = sorted(all_layers)

    presence = pd.DataFrame(0, index=top["gene"], columns=all_layers)
    for _, row in top.iterrows():
        for layer in row["layers"].split(";"):
            presence.loc[row["gene"], layer] = 1

    fig, ax = plt.subplots(figsize=(max(8, len(all_layers) * 1.2), max(6, top_n * 0.4)))
    sns.heatmap(
        presence,
        cmap="YlOrRd",
        cbar_kws={"label": "Present in layer"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Multi-Omic Resilience Signature (Top {top_n} Genes)")
    ax.set_xlabel("Omic Layer")
    ax.set_ylabel("Gene")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved multi-omic summary: {output_path}")


def plot_de_comparison(
    de_results: dict[str, pd.DataFrame],
    output_path: Path,
    top_n: int = 20,
    padj_thresh: float = 0.05,
) -> None:
    """Compare DE results across contrasts with a dot plot."""
    fig, axes = plt.subplots(1, len(de_results), figsize=(8 * len(de_results), 10))
    if len(de_results) == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, de_results.items()):
        sig = df[df["padj"] < padj_thresh].head(top_n)
        if sig.empty:
            ax.set_title(f"{name}\n(no significant genes)")
            continue

        ax.barh(
            range(len(sig)),
            sig["log2FoldChange"],
            color=["firebrick" if x > 0 else "steelblue" for x in sig["log2FoldChange"]],
        )
        ax.set_yticks(range(len(sig)))
        ax.set_yticklabels(sig["gene"], fontsize=8)
        ax.set_xlabel("log2 Fold Change")
        ax.set_title(f"{name}\n(top {len(sig)} genes)")
        ax.axvline(0, color="grey", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved DE comparison: {output_path}")
