"""Generate every figure used in the Aegis-EHR paper.

Writes PNG (300 dpi) + PDF to figures/ and PNG screenshots/.
All numerical content is the paper's [SIMULATED] data.

Author: Anas AlSobeh, Utah Valley University, 2026
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"
SHOT_DIR = PROJECT_ROOT / "screenshots"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SHOT_DIR.mkdir(parents=True, exist_ok=True)

# Green palette
PALETTE = {
    "primary":   "#2E8B57",
    "secondary": "#4CAF50",
    "light":     "#A5D6A7",
    "dark":      "#1B5E20",
}
BAR_COLORS = ["#1B5E20", "#2E8B57", "#4CAF50", "#7CB342", "#A5D6A7", "#C8E6C9"]


def _save(fig, name: str):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    shot = SHOT_DIR / f"{name}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(shot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png.name}, {pdf.name}, {shot.name}")


# ---------------------------------------------------------------------------
def fig_main_comparison():
    systems = ["GPT-4o", "Claude 3.5", "Reflexion", "TOAI-MR",
               "Med-PaLM 2", "Aegis-EHR"]
    metrics = ["MedBLEU", "FCS", "SPI", "TCR", "PAR"]
    # [SIMULATED] paper values
    data = np.array([
        [0.86, 0.81, 0.83, 0.72, 0.58],   # GPT-4o
        [0.85, 0.86, 0.84, 0.74, 0.60],   # Claude
        [0.86, 0.89, 0.82, 0.70, 0.58],   # Reflexion
        [0.88, 0.85, 0.89, 0.75, 0.60],   # TOAI-MR
        [0.84, 0.82, 0.83, 0.72, 0.58],   # Med-PaLM 2
        [0.924, 0.981, 0.953, 0.903, 0.854],  # Aegis-EHR
    ])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics))
    w = 0.13
    for i, sysname in enumerate(systems):
        offset = (i - (len(systems) - 1) / 2) * w
        ax.bar(x + offset, data[i], width=w, label=sysname,
               color=BAR_COLORS[i % len(BAR_COLORS)],
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score [SIMULATED]")
    ax.set_title("Fig. 2 -- Main comparison across five reconciliation metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncols=3, loc="lower right", framealpha=0.9, fontsize=8)
    _save(fig, "fig_main_comparison")


def fig_threshold_sensitivity():
    tau = np.linspace(0.40, 0.95, 50)
    par = 0.42 + 0.43 * (1.0 / (1.0 + np.exp(-12 * (tau - 0.72))))
    abst = 1.0 - 1.0 / (1.0 + np.exp(-12 * (tau - 0.72))) * 0.5 + 0.02 * tau
    abst = np.clip(abst, 0.0, 1.0) * 0.6
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(tau, par, color=PALETTE["dark"], lw=2.4, label="PAR")
    ax1.set_xlabel(r"Emit threshold $\tau_{\mathrm{emit}}$ [SIMULATED]")
    ax1.set_ylabel("PAR", color=PALETTE["dark"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["dark"])
    ax1.axvline(0.70, color="gray", ls=":", lw=1)
    ax1.text(0.705, 0.55, r"chosen $\tau=0.70$", color="gray")
    ax2 = ax1.twinx()
    ax2.plot(tau, abst, color=PALETTE["secondary"], lw=2.0, ls="--",
             label="Abstention rate")
    ax2.set_ylabel("Abstention rate", color=PALETTE["secondary"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["secondary"])
    fig.suptitle("Fig. 3 -- Threshold sensitivity (PAR vs abstention)")
    _save(fig, "fig_threshold_sensitivity")


def fig_reliability():
    # 10-bin reliability diagram, Aegis-EHR vs GPT-4o
    bins = np.linspace(0.05, 0.95, 10)
    perfect = bins
    aegis = bins + np.array([0.02, 0.01, -0.01, 0.0, 0.01, -0.005,
                             0.005, -0.01, 0.0, -0.005])
    gpt = bins + np.array([0.18, 0.15, 0.12, 0.08, 0.05, 0.02,
                           -0.02, -0.05, -0.08, -0.10])
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls=":", label="Perfect")
    ax.plot(bins, aegis, "o-", color=PALETTE["dark"], lw=2,
            label="Aegis-EHR (ECE=0.041)")
    ax.plot(bins, gpt, "s--", color="#888888", lw=1.8,
            label="GPT-4o (ECE=0.187)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability [SIMULATED]")
    ax.set_ylabel("Empirical frequency [SIMULATED]")
    ax.set_title("Fig. 4 -- Reliability diagram (calibration)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    _save(fig, "fig_reliability")


def fig_drift_psi():
    days = np.arange(1, 121)
    rng = np.random.default_rng(7)
    base = 0.05 + 0.001 * days
    spike = np.where((days > 60) & (days < 75),
                     0.18 * np.exp(-0.2 * (days - 67) ** 2 / 5), 0)
    psi = base + spike + rng.normal(0, 0.012, size=days.size)
    psi = np.clip(psi, 0, None)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(days, psi, color=PALETTE["primary"], lw=1.8)
    ax.axhline(0.25, color="#b71c1c", ls="--", lw=1.5,
               label="Alert threshold PSI = 0.25")
    ax.fill_between(days, 0, psi,
                    where=psi > 0.25, color="#ef9a9a", alpha=0.35,
                    label="Drift alert zone")
    ax.set_xlabel("Day [SIMULATED]")
    ax.set_ylabel("Rolling PSI [SIMULATED]")
    ax.set_title("Fig. 5 -- Governance Agent: PSI over a 120-day window")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "fig_drift_psi")


def fig_latency():
    systems = ["GPT-4o", "Claude 3.5", "Reflexion", "TOAI-MR", "Aegis-EHR"]
    schema = np.array([0.42, 0.40, 0.55, 0.48, 0.31])
    reason = np.array([0.31, 0.32, 0.40, 0.36, 0.28])
    temporal = np.array([0.08, 0.09, 0.12, 0.10, 0.14])
    pharma = np.array([0.05, 0.05, 0.07, 0.06, 0.09])
    gov = np.array([0.0, 0.0, 0.02, 0.02, 0.06])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(systems))
    ax.bar(x, schema, color=PALETTE["dark"], label="Schema")
    ax.bar(x, reason, bottom=schema, color=PALETTE["primary"], label="Reasoning")
    bottom = schema + reason
    ax.bar(x, temporal, bottom=bottom, color=PALETTE["secondary"], label="Temporal")
    bottom = bottom + temporal
    ax.bar(x, pharma, bottom=bottom, color=PALETTE["light"], label="Pharma")
    bottom = bottom + pharma
    ax.bar(x, gov, bottom=bottom, color="#C8E6C9", label="Governance")
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("Latency per note (s) [SIMULATED]")
    ax.set_title("Fig. 6 -- Stacked latency by stage")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_latency")


def fig_cost_pareto():
    # cost per 1k notes (USD), PAR
    systems = [("GPT-4o", 4.20, 0.58),
               ("Claude 3.5", 4.60, 0.60),
               ("Med-PaLM 2", 3.80, 0.58),
               ("Reflexion", 5.20, 0.58),
               ("TOAI-MR", 2.90, 0.60),
               ("Aegis-EHR", 1.85, 0.854)]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for i, (name, cost, par) in enumerate(systems):
        color = PALETTE["dark"] if name == "Aegis-EHR" else "#777777"
        ax.scatter(cost, par, s=160 if name == "Aegis-EHR" else 90,
                   color=color, edgecolor="white", zorder=3)
        ax.annotate(name, (cost, par), xytext=(6, 6),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("Cost per 1k notes (USD) [SIMULATED]")
    ax.set_ylabel("PAR [SIMULATED]")
    ax.set_title("Fig. 7 -- Cost vs PAR Pareto frontier")
    ax.grid(alpha=0.3)
    _save(fig, "fig_cost_pareto")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.5)
    ax.axis("off")

    # Input node
    input_box = mpatches.FancyBboxPatch((0.2, 2.2), 1.6, 1.1,
                                        boxstyle="round,pad=0.05",
                                        facecolor=PALETTE["light"],
                                        edgecolor=PALETTE["dark"], lw=1.5)
    ax.add_patch(input_box)
    ax.text(1.0, 2.75, "Clinical\nNote", ha="center", va="center",
            fontsize=11, fontweight="bold")

    # Five agent nodes
    agents = [
        ("Schema\nAgent\n$a_S$",          3.4, 4.1),
        ("Reasoning\nAgent\n$a_R$",        3.4, 2.75),
        ("Temporal\nAgent\n$a_T$",         3.4, 1.4),
        ("Pharma\nAgent\n$a_P$",           5.6, 2.05),
        ("Governance\nAgent\n$a_G$",       5.6, 3.45),
    ]
    for label, x, y in agents:
        box = mpatches.FancyBboxPatch((x - 0.7, y - 0.55), 1.4, 1.05,
                                      boxstyle="round,pad=0.04",
                                      facecolor=PALETTE["secondary"],
                                      edgecolor=PALETTE["dark"], lw=1.4)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
        ax.annotate("", xy=(x - 0.7, y), xytext=(1.85, 2.75),
                    arrowprops=dict(arrowstyle="->",
                                    color=PALETTE["dark"], lw=1.0))

    # Fusion node (noisy-OR)
    fusion = mpatches.FancyBboxPatch((7.4, 2.2), 1.8, 1.1,
                                     boxstyle="round,pad=0.05",
                                     facecolor=PALETTE["primary"],
                                     edgecolor=PALETTE["dark"], lw=1.6)
    ax.add_patch(fusion)
    ax.text(8.3, 2.75, "Noisy-OR\nFusion\n$\\sigma_{joint}$",
            ha="center", va="center", color="white", fontsize=10,
            fontweight="bold")
    for _, x, y in agents:
        ax.annotate("", xy=(7.4, 2.75), xytext=(x + 0.7, y),
                    arrowprops=dict(arrowstyle="->",
                                    color="#555555", lw=0.9, alpha=0.7))

    # Output FHIR bundle
    out_box = mpatches.FancyBboxPatch((9.6, 2.2), 1.3, 1.1,
                                      boxstyle="round,pad=0.05",
                                      facecolor=PALETTE["dark"],
                                      edgecolor=PALETTE["dark"], lw=1.4)
    ax.add_patch(out_box)
    ax.text(10.25, 2.75, "FHIR\nR4\nBundle", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")
    ax.annotate("", xy=(9.6, 2.75), xytext=(9.2, 2.75),
                arrowprops=dict(arrowstyle="->",
                                color=PALETTE["dark"], lw=1.6))

    # Repair-loop arrow R<=3
    ax.annotate("", xy=(1.0, 3.3), xytext=(8.3, 3.85),
                arrowprops=dict(arrowstyle="->", color="#b71c1c",
                                lw=1.2, ls="--",
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(5.0, 4.55, "repair loop  R$\\leq$3", color="#b71c1c",
            fontsize=9, ha="center", style="italic")

    ax.text(5.5, 0.4,
            "Aegis-EHR mesh: five specialised agents -> noisy-OR fusion "
            "-> abstention gate ($\\tau$=0.70) -> FHIR R4",
            ha="center", fontsize=10, color=PALETTE["dark"])
    fig.suptitle("Fig. 1 -- Aegis-EHR decentralized agentic mesh",
                 fontsize=12, fontweight="bold")
    _save(fig, "fig_architecture")


# ---------------------------------------------------------------------------
def main():
    print(f"Writing figures to {FIG_DIR} and {SHOT_DIR}")
    fig_architecture()
    fig_main_comparison()
    fig_threshold_sensitivity()
    fig_reliability()
    fig_drift_psi()
    fig_latency()
    fig_cost_pareto()
    print("All figures generated.")


if __name__ == "__main__":
    main()
