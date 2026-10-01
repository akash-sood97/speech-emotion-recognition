"""Builds outputs/summary.json and figures/00_decision.png from the staged evaluation tables.

Run from the repo root:  python3 analysis/make_case_study.py
Every number quoted in README.md comes from outputs/summary.json.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import viz  # noqa: E402

ROOT = HERE.parent
T = ROOT / "Evaluation" / "outputs" / "tables"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# --- read sources ---------------------------------------------------------------------
hm = pd.read_csv(T / "heldout_metrics.csv").set_index("model")
spk = pd.read_csv(T / "heldout_macro_f1_by_speaker.csv", index_col=0)
s08 = pd.read_csv(T / "summary_08.csv").set_index("H")
fit = json.loads((T / "exam_fit_heldout.json").read_text())
op = json.loads((T / "flag_operating_point.json").read_text())

# Held-out flag values are only stored in the evidence text of H39 in summary_08.csv, e.g.
# "flag rate 39.2%, capture 70.3%, precision 95.7% at threshold 0.827"; parse them from there.
ev = s08.loc["H39", "evidence"]
m = re.search(r"flag rate ([\d.]+)%, capture ([\d.]+)%, precision ([\d.]+)% at threshold ([\d.]+)", ev)
if not m:
    raise ValueError(f"cannot parse held-out flag values from: {ev}")
flag_rate_h, capture_h, precision_h, thr_h = (float(x) for x in m.groups())
assert abs(thr_h - op["threshold"]) < 1e-9, "H39 threshold differs from flag_operating_point.json"

speakers_ahead = int((spk["wavlm"] > spk["logreg"]).sum())
S = {
    "heldout_macro_f1": round(float(hm.loc["wavlm", "macro_f1"]), 4),
    "ci_low": round(float(hm.loc["wavlm", "ci_low"]), 4),
    "ci_high": round(float(hm.loc["wavlm", "ci_high"]), 4),
    "logreg_macro_f1": round(float(hm.loc["logreg", "macro_f1"]), 4),
    "majority_macro_f1": round(float(hm.loc["majority", "macro_f1"]), 4),
    "speakers_ahead": speakers_ahead,
    "heldout_speakers": int(len(spk)),
    "heldout_clips": int(fit["n_scored"]),
    "dev_clips": int(fit["n_train"]),
    "flag_threshold": op["threshold"],
    "flag_rate_dev": op["flag_rate"],
    "flag_rate_heldout": round(flag_rate_h / 100, 3),
    "capture_heldout": round(capture_h / 100, 3),
    "precision_heldout": round(precision_h / 100, 3),
}
# The gain is computed from unrounded scores upstream, so the difference of the rounded values
# (0.1821) would disagree; take the stated value from H38's evidence text in summary_08.csv.
S["gain_vs_logreg"] = float(re.search(r"gain \+([\d.]+) macro-F1", s08.loc["H38", "evidence"]).group(1))
(OUT / "summary.json").write_text(json.dumps(S, indent=2) + "\n")

# --- decision card --------------------------------------------------------------------
viz.setup()


def draw(ax):
    labels = ["Always guess one emotion", "Classical model (512 features)", "WavLM encoder"]
    vals = [S["majority_macro_f1"], S["logreg_macro_f1"], S["heldout_macro_f1"]]
    cols = [viz.C["neutral"], viz.C["orange"], viz.C["blue"]]
    ax.barh(range(3), vals, color=cols, height=0.4)
    ax.set_ylim(2.6, -0.6)
    ax.set_yticks([])
    for i, lab in enumerate(labels):
        ax.text(0, i - 0.3, lab, va="bottom", ha="left", fontsize=12, color=viz.C["muted"])
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=13, fontweight="bold", color=viz.C["ink"])
    ax.set_xlim(0, 0.95)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Macro-F1 on 6 unseen speakers", fontsize=13, loc="left")


viz.decision_card(
    FIG / "00_decision.png",
    f"Flag {S['flag_rate_heldout']*100:.0f}% of calls and reviewers hear {S['capture_heldout']*100:.0f}% of the "
    f"negative ones first — {S['precision_heldout']*100:.0f}% of flags are right.",
    [(f"{S['heldout_macro_f1']:.2f}", f"macro-F1 on {S['heldout_speakers']} speakers never heard in training"),
     (f"+{S['gain_vs_logreg']:.2f}", f"over the best classical model, ahead on {S['speakers_ahead']} of {S['heldout_speakers']} speakers"),
     (f"{S['capture_heldout']*100:.0f}% at {S['flag_rate_heldout']*100:.0f}%", "negative calls caught at the flag rate")],
    draw,
    note="Acted speech, 24 speakers, course-supplied audio. Held-out test scored once; model chosen beforehand.",
)
print(json.dumps(S, indent=2))
