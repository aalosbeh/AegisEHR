"""Generate every figure in the revised paper from the result files.

Usage: python scripts/08_figures.py   (writes figures/*.png at 300 dpi)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

GREEN = ["#1B5E20", "#2E7D32", "#43A047", "#66BB6A", "#A5D6A7", "#C8E6C9"]
GREY = ["#424242", "#757575", "#9E9E9E", "#BDBDBD"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linestyle": "--", "savefig.dpi": 300, "legend.frameon": False})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc, ec="#1B5E20", fs=9, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, a, b, style="-|>", ls="-", color="#1B5E20", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 11.4)
    ax.axis("off")
    box(ax, 0.2, 8.6, 3.4, 1.7, "Clinical note\n(admission, progress,\ndischarge)", "#E8F5E9", fs=8.5)
    box(ax, 0.2, 5.6, 3.4, 1.8, "Optional external\ndraft (LLM FHIR\nbundle)", "#F1F8E9", ec="#757575", fs=8.5)
    box(ax, 4.6, 8.4, 12.0, 2.1, "Orchestrator (central control)\nrouting in dependency order, repair merging,\n"
        "calibrated joint confidence, decision gate", "#2E7D32", fs=9.5, bold=True, tc="white")
    agents = [("Schema Agent\n\nFHIR R4 mapping,\nvalidation,\nstructural repair", 4.6),
              ("Reasoning Agent\n\nassertion,\nsource grounding,\nreconciliation", 7.75),
              ("Temporal Agent\n\nsource-grounded\ndates; 11 axioms\n(7 used by agent)", 10.9),
              ("Pharmacovigilance\nAgent\n\noperational KB,\n4 rule types", 14.05)]
    for t, x in agents:
        box(ax, x, 4.5, 2.55, 2.9, t, "#C8E6C9", fs=8)
    for i in range(3):
        arrow(ax, (agents[i][1] + 2.55, 5.95), (agents[i + 1][1], 5.95))
    for _, x in agents:
        arrow(ax, (x + 1.27, 8.4), (x + 1.27, 7.4), style="<|-|>")
    box(ax, 4.6, 1.8, 12.0, 1.7, "Governance Agent\nrolling PSI (magnitude) and signed SMD (direction) over agent signals;\n"
        "an alarm downgrades AUTO to REVIEW", "#A5D6A7", fs=8.5)
    for _, x in agents:
        arrow(ax, (x + 1.27, 4.5), (x + 1.27, 3.5), ls="--", color="#558B2F")
    box(ax, 17.8, 9.3, 3.9, 1.1, "AUTO\nbundle emitted", "#E8F5E9", fs=8.5)
    box(ax, 17.8, 7.8, 3.9, 1.1, "REVIEW\nbundle + warnings", "#FFF8E1", ec="#F9A825", fs=8.5)
    box(ax, 17.8, 6.3, 3.9, 1.1, "ABSTAIN\nno bundle", "#FFEBEE", ec="#C62828", fs=8.5)
    for y in (9.85, 8.35, 6.85):
        arrow(ax, (16.6, 9.45), (17.8, y))
    box(ax, 17.8, 3.9, 3.9, 1.7, "Typed agent reports\n(MCP messages) and\nprovenance log", "#F5F5F5", ec="#757575", fs=8)
    box(ax, 17.8, 1.8, 3.9, 1.5, "Per-agent Platt\ncalibration and\nnoisy-AND aggregation", "#F5F5F5", ec="#757575", fs=8)
    arrow(ax, (3.6, 9.45), (4.6, 9.45))
    arrow(ax, (3.6, 6.5), (4.6, 6.2), ls="--", color="#757575")
    arrow(ax, (16.6, 2.6), (17.8, 2.55), ls="--", color="#757575")
    arrow(ax, (16.6, 5.95), (17.8, 4.75), ls="--", color="#757575")
    ax.text(10.6, 0.7, "Order S \u2192 R \u2192 T \u2192 P: structure before semantics, semantics before time, "
            "time before drug-safety checks", ha="center", fontsize=9, style="italic", color="#33691E")
    save(fig, "fig1_architecture")


def fig_risk_coverage():
    A = json.loads((RES / "figdata" / "partA_risk_coverage.json").read_text())
    B = json.loads((RES / "figdata" / "partB_curves.json").read_text()) if (RES / "figdata" / "partB_curves.json").exists() else {}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    sty = {"Aegis-EHR": (GREEN[0], "-"), "Single-pass": (GREY[1], "--"), "Rule-based": (GREY[2], ":")}
    for k, v in A.items():
        c, ls = sty[k]
        axs[0].plot(v["coverage"], v["risk"], color=c, ls=ls, lw=1.8, label=f"{k} (AURC {v['aurc']:.3f})")
    axs[0].set_title("(a) Part A, full test set (n = 12,000)", fontsize=10)
    axs[0].set_xlabel("Coverage (share of notes auto-emitted)")
    axs[0].set_ylabel("Risk (share unacceptable among emitted)")
    axs[0].set_ylim(0, 1)
    axs[0].legend(fontsize=8, loc="center right")
    names = {"qwen2.5-7b_direct": ("Qwen2.5-7B direct", GREY[0], "--"),
             "qwen2.5-7b_validator": ("Qwen2.5-7B + validator", GREY[1], "-."),
             "qwen2.5-7b_reflect": ("Qwen2.5-7B + reflection", GREY[2], ":"),
             "qwen2.5-7b_aegis": ("Aegis-EHR on Qwen2.5-7B drafts", GREEN[1], "-"),
             "det_aegis": ("Aegis-EHR (deterministic)", GREEN[0], "-")}
    for k, (lab, c, ls) in names.items():
        if k in B:
            v = B[k]
            axs[1].plot(v["coverage"], v["risk"], color=c, ls=ls, lw=1.8, label=f"{lab} ({v['aurc']:.3f})")
    axs[1].set_title("(b) Part B, LLM subset (n = 150)", fontsize=10)
    axs[1].set_xlabel("Coverage")
    axs[1].set_ylim(0, 1)
    axs[1].legend(fontsize=7.5, loc="upper left")
    save(fig, "fig2_risk_coverage")


def fig_reliability():
    rel = json.loads((RES / "figdata" / "partA_reliability.json").read_text())
    B = json.loads((RES / "figdata" / "partB_curves.json").read_text()) if (RES / "figdata" / "partB_curves.json").exists() else {}
    T = json.loads((RES / "tables" / "partA.json").read_text())["calibration"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax in axs:
        ax.plot([0, 1], [0, 1], color="#BDBDBD", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted confidence")
    axs[0].set_ylabel("Observed acceptability")
    for key, lab, c, mk in (("aegis_noisy_and", "Aegis-EHR, noisy-AND", GREEN[0], "o"),
                            ("aegis_noisy_or", "Aegis-EHR, noisy-OR", GREEN[3], "s"),
                            ("aegis_uncalibrated", "Aegis-EHR, uncalibrated", GREY[2], "x"),
                            ("single", "Single-pass (Platt)", GREY[0], "^")):
        b = rel[key]
        if b:
            x, y, n = zip(*b)
            axs[0].plot(x, y, marker=mk, color=c, lw=1.2, ms=4, label=f"{lab} (ECE {T[key]['ece']:.3f})")
    axs[0].legend(fontsize=7.5, loc="upper left")
    axs[0].set_title("(a) Part A test set, 15 bins", fontsize=10)
    for key, lab, c, mk in (("qwen2.5-7b_direct", "Qwen2.5-7B direct (Platt)", GREY[0], "^"),
                            ("qwen2.5-7b_aegis", "Aegis-EHR on Qwen drafts", GREEN[1], "o")):
        if key in B and B[key]["bins"]:
            x, y, n = zip(*B[key]["bins"])
            axs[1].plot(x, y, marker=mk, color=c, lw=1.2, ms=4, label=lab)
    axs[1].legend(fontsize=7.5, loc="upper left")
    axs[1].set_title("(b) Part B, cross-fitted calibration", fontsize=10)
    save(fig, "fig3_reliability")


def fig_medbleu():
    v = json.loads((RES / "tables" / "partA.json").read_text())["medbleu_validity"]
    share = [s if s is not None else np.nan for s in v["unacceptable_share_by_bleu_decile"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(10)
    ax.bar(x, share, color=GREEN[2], edgecolor=GREEN[0])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a / 10:.1f}–{(a + 1) / 10:.1f}" for a in range(10)], rotation=45, fontsize=8)
    ax.set_xlabel("MedBLEU interval")
    ax.set_ylabel("Share with a critical error")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Spearman ρ(MedBLEU, SWES) = {v['spearman_swes']:.2f}; AUROC for acceptability = {v['auroc_acceptable']:.2f}",
                 fontsize=9)
    save(fig, "fig4_medbleu")


def fig_drift():
    ex = json.loads((RES / "figdata" / "drift_examples.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=False)
    for ax, key, title in ((axs[0], "lexical_abrupt", "(a) Lexical shift, abrupt (magnitude 1.0)"),
                           (axs[1], "lexical_gradual", "(b) Lexical shift, gradual 30-day ramp")):
        e = ex[key]
        d = np.arange(len(e["peak_psi"]))
        ax.plot(d, np.maximum(e["peak_psi"], 1e-3), color=GREEN[0], lw=1.6, label="max PSI over signals")
        ax.set_yscale("log")
        ax.set_ylim(5e-3, 20)
        ax.axhline(0.25, color="#C62828", ls="--", lw=1, label="alert threshold 0.25")
        ax.axvline(e["onset"], color=GREY[1], ls=":", lw=1.2, label="drift onset")
        if e["alarm"] is not None:
            ax.axvline(e["alarm"], color="#C62828", lw=1, alpha=0.6, label="PSI alarm")
        ax.set_xlabel("Day")
        ax.set_ylabel("PSI")
        ax2 = ax.twinx()
        ax2.plot(d, e["roll_acc"], color=GREY[0], lw=1.2, ls="-.", label="7-day acceptability")
        ax2.set_ylabel("Acceptability")
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(False)
        if e["degradation"] is not None:
            ax.axvline(e["degradation"], color="#F9A825", lw=1.2, label="material degradation")
        ax.set_title(title, fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if ax is axs[0]:
            fig.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=4)
    save(fig, "fig5_drift")


def fig_latency():
    nl = json.loads((RES / "tables" / "noise_pairwise_latency.json").read_text())
    A = json.loads((RES / "tables" / "partA.json").read_text())
    steps = nl["latency_steps_ms"]
    lat = nl["latency_single_process"]
    parse_ms = steps.get("parse", 0)
    parts = [("Note parsing (full mode)", parse_ms), ("Schema", steps.get("S", 0)), ("Reasoning", steps.get("R", 0)),
             ("Temporal", steps.get("T", 0)), ("Pharmacovigilance", steps.get("P", 0))]
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    left = 0
    for (lab, v), c in zip(parts, [GREY[3]] + GREEN[:4]):
        axs[0].barh(["Aegis-EHR\n(deterministic)"], [v], left=left, color=c, edgecolor="white", label=f"{lab} {v:.2f} ms")
        left += v
    axs[0].set_xlabel("Mean milliseconds per note (single CPU core)")
    axs[0].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
    axs[0].set_title("(a) Aegis-EHR latency by agent", fontsize=10)
    B = json.loads((RES / "tables" / "partB.json").read_text()) if (RES / "tables" / "partB.json").exists() else None
    names, vals = ["Rule-based", "Single-pass", "Aegis-EHR"], [lat["rule"]["mean_ms"] / 1000, lat["single"]["mean_ms"] / 1000,
                                                             lat["aegis"]["mean_ms"] / 1000]
    if B:
        for tag, lab in (("qwen2.5-7b_direct", "Qwen direct"), ("qwen2.5-7b_validator", "Qwen + validator"),
                         ("qwen2.5-7b_reflect", "Qwen + reflection"), ("llama3.2-3b_direct", "Llama direct"),
                         ("gemma3-12b_direct", "Gemma direct")):
            if tag in B["fhir"]:
                names.append(lab)
                vals.append(B["fhir"][tag].get("median_wall_s", B["fhir"][tag]["mean_wall_s"]))
    axs[1].barh(names, vals, color=[GREEN[1]] * 3 + [GREY[1]] * (len(names) - 3))
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Median seconds per note (log scale)")
    axs[1].set_title("(b) Wall-clock time per note", fontsize=10)
    save(fig, "fig7_latency")


def fig_noise():
    nl = json.loads((RES / "tables" / "noise_pairwise_latency.json").read_text())["noise"]
    p = sorted(float(k) for k in nl)
    g = lambda s, k: [nl[str(x) if str(x) in nl else f"{x}"][s][k] or 0 for x in p]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.0))
    for ax, k, title in ((axs[0], "acceptable", "(a) Acceptability, all notes"), (axs[1], "coverage", "(b) Auto-emitted share"),
                         (axs[2], "silent_error_rate", "(c) Silent error rate")):
        ax.plot(p, g("aegis", k), marker="o", color=GREEN[0], label="Aegis-EHR")
        ax.plot(p, g("single", k), marker="^", color=GREY[1], ls="--", label="Single-pass")
        ax.set_xlabel("Token corruption rate")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.02)
    axs[0].legend(fontsize=8)
    save(fig, "fig6_noise")


if __name__ == "__main__":
    which = sys.argv[1:] or ["architecture", "risk_coverage", "reliability", "medbleu", "drift", "latency", "noise"]
    for w in which:
        try:
            globals()[f"fig_{w}"]()
        except FileNotFoundError as e:
            print("skip", w, e)
