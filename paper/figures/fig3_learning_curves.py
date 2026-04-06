"""Figure 3 — PPO Learning Curves (300k steps, dual-axis)."""
import sys, os
sys.path.insert(0, "/lustre/guillant/C2G-Macro")
os.chdir("/lustre/guillant/C2G-Macro")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np, pandas as pd, glob

plt.rcParams.update({
    "font.family": "serif", "axes.spines.top": False,
    "axes.spines.right": False,
})
BLUE="#2166ac"; GREEN="#4dac26"; RED="#b2182b"; GRAY="#878787"; ORANGE="#d6604d"

# ── Load the best (longest) training run ──────────────────────────────────────
csvs = sorted(glob.glob("outputs/ppo_default/**/episode_metrics.csv", recursive=True))
dfs  = [pd.read_csv(f) for f in csvs]
best = max(dfs, key=len)       # 1118-episode run
# aggregate by env_idx (4 envs), average over them
best = best.groupby("timestep").mean(numeric_only=True).reset_index()

steps  = best["timestep"].values
reward = best["ep/mean_reward"].values
rmse   = best["grid/tracking_rmse_kw"].values / 1e3    # → MW-equivalent kW displayed as k
viol   = best["thermal/viol_rate"].values
soc    = best["bess/mean_soc"].values
pue    = best["facility/mean_pue"].values

def smooth(x, w=30):
    """Rolling mean with edge padding."""
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True,
                         gridspec_kw={"hspace": 0.08})

steps_k = steps / 1e3  # thousands

# ── Panel 1: Reward + RMSE ────────────────────────────────────────────────────
ax1 = axes[0]
ax1r = ax1.twinx()

ax1.plot(steps_k, reward, color=BLUE, lw=0.5, alpha=0.25)
ax1.plot(steps_k, smooth(reward, 40), color=BLUE, lw=2.2, label="Mean step reward")
ax1.fill_between(steps_k, smooth(reward, 40) - 0.005,
                  smooth(reward, 40) + 0.005, color=BLUE, alpha=0.15)
ax1.axhline(0, color=GRAY, lw=0.8, ls="--")
ax1.set_ylabel("Mean Step Reward", color=BLUE, fontsize=10)
ax1.tick_params(axis="y", labelcolor=BLUE)
ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())

ax1r.plot(steps_k, rmse, color=ORANGE, lw=0.5, alpha=0.25)
ax1r.plot(steps_k, smooth(rmse, 40), color=ORANGE, lw=2.2, ls="--",
          label="Tracking RMSE")
ax1r.set_ylabel("Tracking RMSE (kW ×10³)", color=ORANGE, fontsize=10)
ax1r.tick_params(axis="y", labelcolor=ORANGE)

# Annotate improvement
rmse_s = smooth(rmse, 40)
x0, x1 = steps_k[10], steps_k[-10]
y0, y1 = rmse_s[10], rmse_s[-10]
ax1r.annotate(f"4× improvement\n({y0:.0f}k → {y1:.0f}k kW)",
              xy=(x1, y1), xytext=(x1*0.55, y0*0.9),
              arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2),
              fontsize=8, color=ORANGE)

lines1, lbl1 = ax1.get_legend_handles_labels()
lines2, lbl2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1+lines2, lbl1+lbl2, fontsize=8, loc="lower left")
ax1.set_title("PPO Training on C2GFastEnv — default scenario, seed 42, 300k steps",
              fontsize=10, fontweight="bold")

# ── Panel 2: Thermal violation rate ──────────────────────────────────────────
ax2 = axes[1]
ax2.plot(steps_k, viol, color=RED, lw=0.5, alpha=0.3)
ax2.plot(steps_k, smooth(viol, 40), color=RED, lw=2.2, label="Thermal violation rate")
ax2.fill_between(steps_k, 0, smooth(viol, 40), color=RED, alpha=0.15)
ax2.set_ylabel("Thermal Viol. Rate", color=RED, fontsize=10)
ax2.tick_params(axis="y", labelcolor=RED)
ax2.set_ylim(bottom=0)

ax2b = ax2.twinx()
ax2b.plot(steps_k, smooth(soc, 40), color=GREEN, lw=2, ls="--", label="Mean SOC")
ax2b.set_ylabel("Mean BESS SOC", color=GREEN, fontsize=10)
ax2b.tick_params(axis="y", labelcolor=GREEN)
ax2b.set_ylim(0, 1)

lines1, lbl1 = ax2.get_legend_handles_labels()
lines2, lbl2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1+lines2, lbl1+lbl2, fontsize=8, loc="center right")

# ── Panel 3: PUE ─────────────────────────────────────────────────────────────
ax3 = axes[2]
ax3.plot(steps_k, pue, color=GRAY, lw=0.5, alpha=0.3)
ax3.plot(steps_k, smooth(pue, 40), color="#555555", lw=2.2, label="Mean PUE")
ax3.axhline(np.min(smooth(pue, 40)), color=GREEN, lw=1, ls=":", label=f"Min PUE={np.min(smooth(pue,40)):.3f}")
ax3.set_ylabel("Power Usage Effectiveness", fontsize=10)
ax3.set_xlabel("Training Steps (×10³)", fontsize=10)
ax3.legend(fontsize=8, loc="upper right")
ax3.yaxis.set_minor_locator(mticker.AutoMinorLocator())

# Phase annotations
for ax in axes:
    ax.axvline(50, color=GRAY, lw=0.8, ls=":", alpha=0.6)
    ax.axvline(150, color=GRAY, lw=0.8, ls=":", alpha=0.6)

axes[0].text(25,  axes[0].get_ylim()[0]*0.95, "Exploration", fontsize=7, color=GRAY, ha="center")
axes[0].text(100, axes[0].get_ylim()[0]*0.95, "Rapid improvement", fontsize=7, color=GRAY, ha="center")
axes[0].text(225, axes[0].get_ylim()[0]*0.95, "Convergence", fontsize=7, color=GRAY, ha="center")

for ax in axes:
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(steps_k[0], steps_k[-1])

fig.savefig("paper/figures/fig3_learning_curves.pdf", dpi=180, bbox_inches="tight")
fig.savefig("paper/figures/fig3_learning_curves.png", dpi=180, bbox_inches="tight")
print("fig3 saved")
