"""Figure 1 — C2G-Bench System Architecture Diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

BLUE   = "#2166ac"
ORANGE = "#d6604d"
GREEN  = "#4dac26"
GRAY   = "#878787"
PURPLE = "#762a83"
LBLUE  = "#d1e5f0"
LORAN  = "#fddbc7"
LGREEN = "#e6f5cb"
LGRAY  = "#f5f5f5"
WHITE  = "#ffffff"

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13); ax.set_ylim(0, 7)
ax.axis("off")

def box(ax, x, y, w, h, label, sublabel="", fc=LBLUE, ec=BLUE, lw=1.8,
        fs=9, sfs=7.5, bold=False):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2 + (0.13 if sublabel else 0),
            label, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=ec, zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha="center", va="center", fontsize=sfs,
                color=GRAY, style="italic", zorder=4)

def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.5, label="", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, linestyle=ls, mutation_scale=14),
                zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.05, my+0.12, label, fontsize=7, color=color, zorder=6)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(6.5, 6.75, "C2G-Bench: Hierarchical AI Orchestration for Grid-Interactive Data Centers",
        ha="center", va="center", fontsize=11, fontweight="bold", color="#222222")

# ── GRID (left) ───────────────────────────────────────────────────────────────
box(ax, 0.15, 4.3, 1.6, 1.2, "Power Grid", "PJM / NYISO", fc="#fff3cd", ec="#856404", bold=True)
box(ax, 0.15, 2.8, 1.6, 1.0, "RegD Signal", r"$r_t \in [-1,1]$", fc="#fff3cd", ec="#856404")
box(ax, 0.15, 1.5, 1.6, 1.0, "LMP Prices", "$/MWh, 5-min", fc="#fff3cd", ec="#856404")

arrow(ax, 1.75, 4.9, 2.65, 5.6, color="#856404", label="market state")
arrow(ax, 1.75, 3.3, 2.65, 3.5, color="#856404", label="RegD signal")
arrow(ax, 1.75, 2.0, 2.65, 2.0, color="#856404", label="LMP")

# ── UPPER AGENT ───────────────────────────────────────────────────────────────
box(ax, 2.65, 5.1, 2.5, 1.2,
    "Market Orchestrator", "15-min ⟶ C2GMacroEnv",
    fc="#d0e6f7", ec=BLUE, bold=True, lw=2.2)
ax.text(3.90, 5.08, "obs: 14-D  |  action: 2-D", ha="center", fontsize=7, color=GRAY)

# ── UPPER→LOWER arrow ─────────────────────────────────────────────────────────
arrow(ax, 3.90, 5.1, 3.90, 4.25, color=BLUE, lw=2,
      label="commit MW, BESS target")

# ── LOWER AGENT ───────────────────────────────────────────────────────────────
box(ax, 2.65, 3.1, 2.5, 1.1,
    "Hardware Controller", "5-min ⟶ C2GFastEnv",
    fc="#cce5ff", ec=BLUE, bold=True, lw=2.2)
ax.text(3.90, 3.08, "obs: 12-D  |  action: 3-D", ha="center", fontsize=7, color=GRAY)

# ── Reward arrows (dashed) ────────────────────────────────────────────────────
ax.annotate("", xy=(3.90, 5.1), xytext=(3.90, 4.55),
            arrowprops=dict(arrowstyle="<|-", color=GREEN, lw=1.5,
                            linestyle="dashed", mutation_scale=12), zorder=5)
ax.text(3.35, 4.84, "macro reward", fontsize=6.5, color=GREEN, style="italic")

ax.annotate("", xy=(3.90, 3.1), xytext=(3.90, 2.65),
            arrowprops=dict(arrowstyle="<|-", color=GREEN, lw=1.5,
                            linestyle="dashed", mutation_scale=12), zorder=5)
ax.text(3.35, 2.85, "step reward", fontsize=6.5, color=GREEN, style="italic")

# ── THREE LEVERS ──────────────────────────────────────────────────────────────
lev_y = 2.0
lever_data = [
    (5.55, "IT Throttle\n(DVFS)", "#f1c40f", "#7d6608"),
    (7.05, "HVAC Effort\n(Cooling)", "#2ecc71", "#1a6e3c"),
    (8.55, "BESS Dispatch\n(±50 MW)", "#e74c3c", "#7b241c"),
]
for lx, lbl, lfc, lec in lever_data:
    box(ax, lx, lev_y, 1.2, 0.85, lbl, fc=lfc+"55", ec=lec, fs=8, bold=True)
    arrow(ax, 5.15, 3.4, lx+0.6, lev_y+0.85, color=BLUE, lw=1.4)

# ── SIMULATORS ────────────────────────────────────────────────────────────────
sim_y = 0.25
sims = [
    (0.3,  "Workload\n(Alibaba traces)", LGREEN, GREEN),
    (2.1,  "Thermal Twin\n(ODE, dual-zone)", LORAN,  ORANGE),
    (3.9,  "Electrical\n(UPS/PDU/PUE)",    LGRAY,  GRAY),
    (5.7,  "BESS\n(150 MWh NMC)",          "#f8d7da","#842029"),
    (7.5,  "Macro-Grid\n(AR(1) RegD+LMP)", "#e2d9f3",PURPLE),
]
for sx, sl, sfc, sec in sims:
    box(ax, sx, sim_y, 1.65, 0.85, sl, fc=sfc, ec=sec, fs=7.8)

# Arrows from levers to sims
arrow(ax, 6.15, lev_y, 1.13, 1.1,  color=GREEN,  lw=1.2, ls="--")
arrow(ax, 6.15, lev_y, 2.93, 1.1,  color=ORANGE, lw=1.2, ls="--")
arrow(ax, 6.15, lev_y, 4.73, 1.1,  color=GRAY,   lw=1.2, ls="--")
arrow(ax, 9.15, lev_y, 6.53, 1.1,  color="#842029", lw=1.2, ls="--")
arrow(ax, 3.90, 3.1,   8.33, 1.1,  color=PURPLE, lw=1.2, ls="--")

# State return arrows (sims → lower env)
for sx in [1.13, 2.93, 4.73, 6.53]:
    ax.annotate("", xy=(3.90, 3.1), xytext=(sx, 1.1),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.8,
                                linestyle="dotted", mutation_scale=10), zorder=2)

# ── LEGEND ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor="#d0e6f7", edgecolor=BLUE, label="RL Agents (Gymnasium)"),
    mpatches.Patch(facecolor="#fff3cd", edgecolor="#856404", label="Power Grid / Market"),
    mpatches.Patch(facecolor=LGREEN,   edgecolor=GREEN,    label="Physics Simulators"),
    plt.Line2D([0],[0], color=GREEN, lw=1.5, linestyle="--", label="Reward signal"),
    plt.Line2D([0],[0], color=GRAY,  lw=1.2, linestyle="dotted", label="State observation"),
]
ax.legend(handles=legend_items, loc="upper right", fontsize=7.5,
          framealpha=0.9, edgecolor=GRAY, bbox_to_anchor=(1.0, 0.98))

fig.tight_layout(pad=0.3)
fig.savefig("paper/figures/fig1_architecture.pdf", dpi=200, bbox_inches="tight")
fig.savefig("paper/figures/fig1_architecture.png", dpi=200, bbox_inches="tight")
print("fig1 saved")
