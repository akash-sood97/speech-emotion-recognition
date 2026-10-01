"""
Redraw Modelling figures F09, F11, F13 from saved tables.

Uses only files inside ``Group-10/``. From that folder:

    python -m redraw_modelling_figures

Or, with shared/ on PYTHONPATH:

    python shared/redraw_modelling_figures.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import audio_utils as U

_MODELLING = U.submission_root() / "Modelling"
FIGS = _MODELLING / "outputs" / "figures"
TABLES = _MODELLING / "outputs" / "tables"
DEV_SRC = (
    "Source: own analysis of the supplied audio dataset, "
    "n=1,080 development clips (speaker-grouped 6-fold CV)"
)
CANDIDATE_TIERS = {"M1", "M2", "M3", "M4"}


def plot_f09(comp: pd.DataFrame, baseline_f1: float) -> None:
    """Speaker-grouped ladder with leaky random markers."""
    c = comp[comp["tier"].isin(CANDIDATE_TIERS)].sort_values(
        "grouped_macro_f1_mean"
    )
    sel = "wavlm"
    colors = [
        U.GOOD if m == sel else U.MUTED for m in c["model"]
    ]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(c) + 1.8))
    yy = np.arange(len(c))
    ax.barh(
        yy, c["grouped_macro_f1_mean"],
        xerr=c["grouped_macro_f1_sd"], capsize=3,
        color=colors, error_kw={"lw": 0.8},
    )
    ax.scatter(
        c["random_macro_f1_mean"], yy, marker="|", s=250,
        color=U.INK, zorder=3, label="random split (leaky) mean",
    )
    ax.axvline(
        baseline_f1, color=U.GREY, ls="--", lw=1,
        label=f"majority baseline {baseline_f1:.3f}",
    )
    ax.set_yticks(yy)
    ax.set_yticklabels(
        [f"{m}  ({t})" for m, t in zip(c["model"], c["tier"])]
    )
    for i, (v, s) in enumerate(
        zip(c["grouped_macro_f1_mean"], c["grouped_macro_f1_sd"])
    ):
        ax.text(
            0.01, i, f"{v:.3f} ± {s:.3f}", va="center", fontsize=8,
            color="white", fontweight="600",
        )
    ax.set_xlim(0, 1)
    row = c.reset_index(drop=True)
    i_sel = int(row.index[row["model"] == sel][0])
    ax.annotate(
        f"selected {sel}",
        xy=(row.loc[i_sel, "grouped_macro_f1_mean"], i_sel),
        xytext=(0.82, i_sel + 0.35),
        fontsize=9, color=U.ACCENT,
        arrowprops=dict(arrowstyle="->", color=U.ACCENT, lw=1),
    )
    U.finish_axes(
        ax,
        "WavLM leads the eligible ladder on speaker-grouped macro-F1",
        "macro-F1", "model (tier)",
        source=DEV_SRC, legend=True,
    )
    U.save_figure(fig, "F09_model_comparison.png", out_dir=FIGS)
    plt.close(fig)


def plot_f13(comp: pd.DataFrame) -> None:
    """Grouped vs random split gap."""
    c = comp[
        comp["tier"].isin(CANDIDATE_TIERS | {"E4"})
        & comp["random_macro_f1_mean"].notna()
    ].sort_values("grouped_macro_f1_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(c) + 2), 4.2))
    x = np.arange(len(c))
    w = 0.38
    ax.bar(
        x - w / 2, c["random_macro_f1_mean"], w,
        color=U.ACCENT, label="random split (leaky)",
    )
    ax.bar(
        x + w / 2, c["grouped_macro_f1_mean"], w,
        color=U.GOOD, label="speaker-grouped split",
    )
    for i, (r, g) in enumerate(
        zip(c["random_macro_f1_mean"], c["grouped_macro_f1_mean"])
    ):
        ax.text(i, max(r, g) + 0.02, f"+{r - g:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(c["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    U.finish_axes(
        ax,
        "Random splits overstate every tier — the leakage gap",
        "model", "macro-F1 (6-fold mean)",
        source=DEV_SRC, legend=True,
    )
    U.save_figure(fig, "F13_split_gap.png", out_dir=FIGS)
    plt.close(fig)


def plot_f11() -> None:
    """Permutation importance for tuned logreg (from saved CSVs)."""
    s = pd.read_csv(TABLES / "F11_group_importance.csv")
    ft = pd.read_csv(TABLES / "F11_feature_importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    top_g = s["mean"].idxmax() if "mean" in s.columns else s.index[-1]
    colors = [
        U.GOOD if i == top_g else U.MUTED for i in s.index
    ]
    ax.barh(
        range(len(s)), s["mean"], xerr=s["std"], color=colors, capsize=3,
    )
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(
        [f"{g} ({n} cols)" for g, n in zip(s["group"], s["n_cols"])]
    )
    U.finish_axes(
        ax,
        "Permute a whole group: how much macro-F1 is lost?",
        "drop in macro-F1 (mean ± sd)", "feature group",
        source="",
    )

    ax = axes[1]
    top = ft.head(15).iloc[::-1]
    gcol = {
        "mfcc": U.MUTED, "spectral": "#E8A33D", "energy": U.ACCENT,
        "pitch": "#8E5FA8", "voice_quality": "#6B8E3D",
    }
    ax.barh(
        range(len(top)), top["drop"],
        color=[gcol.get(x, U.MUTED) for x in top["group"]],
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=8)
    U.finish_axes(
        ax,
        "Top 15 single features (colour = group)",
        "drop in macro-F1 (mean)", "feature",
        source=DEV_SRC,
    )
    fig.suptitle(
        "Energy and spectral cues carry more emotion than MFCCs alone",
        fontsize=12, fontweight="600", x=0.01, ha="left",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    U.save_figure(fig, "F11_feature_importance.png", out_dir=FIGS)
    plt.close(fig)


def main() -> None:
    """Redraw F09, F11 and F13 into Modelling/outputs/figures/."""
    comp = pd.read_csv(TABLES / "model_comparison.csv")
    base = pd.read_csv(TABLES / "baselines_rows.csv")
    base_f1 = float(
        base.loc[base["model"] == "majority", "macro_f1"].iloc[0]
    )
    plot_f09(comp, base_f1)
    plot_f11()
    plot_f13(comp)
    print("redrawn F09, F11, F13 →", FIGS)


if __name__ == "__main__":
    main()
