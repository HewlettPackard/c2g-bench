"""Figure 2 — Simulator Fidelity: 4-panel validation figure."""
import sys, os
sys.path.insert(0, "/lustre/guillant/C2G-Macro")
os.chdir("/lustre/guillant/C2G-Macro")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
})
BL="#2166ac"; OR="#d6604d"; GR="#4dac26"; RD="#b2182b"; GY="#878787"

from c2g_env.simulators.workload import WorkloadOrchestrator, WorkloadState
from c2g_env.simulators.thermal  import ThermalTwin
from c2g_env.simulators.bess     import _SimpleBESSModel
from c2g_env.simulators.macro_grid import MacroGridSignal

fig = plt.figure(figsize=(13, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
panel_labels = ["(a)", "(b)", "(c)", "(d)"]

# ─── Panel A: WorkloadOrchestrator 24-h trace ───────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ws = WorkloadOrchestrator(trace_dir="data/processed/workload_traces", seed=0)
ws.reset(seed=0)
T = 288
base_kw   = np.zeros(T)
flex_kw   = np.zeros(T)
spikes    = np.zeros(T, dtype=bool)
for t in range(T):
    s = ws.step(throttle_batch=1.0)
    base_kw[t]  = s.p_base_kw
    flex_kw[t]  = s.p_flex_kw
    spikes[t]   = s.is_spike_active
hours = np.arange(T) * 5 / 60
ax_a.stackplot(hours, base_kw / 1e3, flex_kw / 1e3,
               colors=[BL, OR], alpha=0.75,
               labels=["P_base (rigid)",  "P_flex (batch)"])
for i in range(T-1):
    if spikes[i]:
        ax_a.axvspan(hours[i], hours[i]+5/60, color="yellow", alpha=0.35, lw=0)
import matplotlib.patches as mpatches
spike_patch = mpatches.Patch(color="yellow", alpha=0.6, label="GenAI spike")
ax_a.legend(handles=[*ax_a.get_legend_handles_labels()[0], spike_patch],
            fontsize=7, loc="upper right")
ax_a.set_xlabel("Hour of Day"); ax_a.set_ylabel("IT Power (GW×10⁻¹)")
ax_a.set_title(f"{panel_labels[0]}  Workload Orchestrator — 24-hour trace",
               fontsize=9, fontweight="bold")
ax_a.set_xticks(range(0,25,4))

# ─── Panel B: ThermalTwin step response ─────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
tt = ThermalTwin()
tt.reset()
N = 180  # 15 hours
tA = np.zeros(N); tB = np.zeros(N)
p_A_vals = [50.0]*40 + [100.0]*60 + [50.0]*(N-100)
hvac_vals = [0.5]*40 + [0.9]*60  + [0.5]*(N-100)
for i in range(N):
    p_A = p_A_vals[i]; hvac = hvac_vals[i]
    (tA[i], tB[i]), _ = tt.step(p_it_A_mw=p_A, p_it_B_mw=30.0, hvac_effort=hvac)
h2 = np.arange(N) * 5 / 60
ax_b.plot(h2, tA, color=RD, lw=2.0, label="Zone A (liquid-cooled)")
ax_b.plot(h2, tB, color=BL, lw=2.0, label="Zone B (air-cooled)")
ax_b.axhline(35.0, color="firebrick", lw=1.2, ls=":", label="T_safe = 35°C")
ax_b.axhline(30.0, color=OR,          lw=1.0, ls="--", label="T_warn = 30°C")
ax_b.axvspan(40*5/60, 100*5/60, color=GR, alpha=0.12, label="Load surge")
ax_b.legend(fontsize=7, loc="upper right")
ax_b.set_xlabel("Hour"); ax_b.set_ylabel("Temperature (°C)")
ax_b.set_title(f"{panel_labels[1]}  ThermalTwin — step response",
               fontsize=9, fontweight="bold")

# ─── Panel C: BESS — η vs C-rate + SOC depletion ────────────────────────────
ax_c  = fig.add_subplot(gs[1, 0])
ax_c2 = ax_c.twinx()

bess = _SimpleBESSModel()
bess.reset()

# η curve
ETA_PEAK = bess.ETA_PEAK; K_CRATE = bess.K_CRATE; K_SOC = bess.K_SOC
crates  = np.linspace(0, 0.5, 200)
eta_arr = np.maximum(0.70, ETA_PEAK - K_CRATE * crates**2 - K_SOC*(0.5-0.5)**2)
rte_arr = eta_arr**2 * 100
ax_c.plot(crates, rte_arr, color=BL, lw=2.2, label="RT efficiency η² (%)")
ax_c.set_xlabel("C-rate (hr⁻¹)"); ax_c.set_ylabel("Round-trip η (%)", color=BL)
ax_c.tick_params(axis="y", labelcolor=BL)
ax_c.set_ylim(60, 100)

# SOC depletion at 1C discharge
bess2 = _SimpleBESSModel(); bess2.reset()
soc_arr = []; t_bess = []
for _ in range(72):
    res = bess2.step(power_mw=bess2.P_MAX_MW * 0.8)
    soc_arr.append(res["soc_fraction"] * 100)
    t_bess.append(_ * 5 / 60)
ax_c2.plot(t_bess, soc_arr, color=OR, lw=2.0, ls="--", label="SoC (0.8C discharge)")
ax_c2.set_ylabel("State of Charge (%)", color=OR)
ax_c2.tick_params(axis="y", labelcolor=OR)
ax_c2.axhline(20, color=GY, lw=0.8, ls=":")

h1, l1 = ax_c.get_legend_handles_labels()
h2, l2 = ax_c2.get_legend_handles_labels()
ax_c.legend(h1+h2, l1+l2, fontsize=7, loc="upper right")
ax_c.set_title(f"{panel_labels[2]}  BESS — efficiency & SoC depletion",
               fontsize=9, fontweight="bold")

# ─── Panel D: MacroGridSignal — RegD + LMP ──────────────────────────────────
ax_d  = fig.add_subplot(gs[1, 1])
ax_d2 = ax_d.twinx()

grid = MacroGridSignal(energy_dir="data/processed/energy", committed_mw=20.0, seed=0)
grid.reset(seed=0)
T = 288
delta_p = np.zeros(T); lmp_arr = np.zeros(T)
for t in range(T):
    res = grid.step()
    delta_p[t] = res["delta_p_kw"]
    lmp_arr[t] = res["lmp_usd_mwh"]

hg = np.arange(T) * 5 / 60
ax_d.fill_between(hg, 0, delta_p / 1e3,
                  where=delta_p >= 0, color=GR, alpha=0.55, label="Consume (charge)")
ax_d.fill_between(hg, 0, delta_p / 1e3,
                  where=delta_p < 0,  color=RD, alpha=0.55, label="Shed (discharge)")
ax_d.set_ylabel("RegD Dispatch (MW)", fontsize=9)
ax_d.axhline(0, color=GY, lw=0.7)

ax_d2.plot(hg, lmp_arr, color=OR, lw=1.5, label="LMP ($/MWh)")
ax_d2.set_ylabel("LMP ($/MWh)", color=OR)
ax_d2.tick_params(axis="y", labelcolor=OR)

h1, l1 = ax_d.get_legend_handles_labels()
h2, l2 = ax_d2.get_legend_handles_labels()
ax_d.legend(h1+h2, l1+l2, fontsize=7, loc="upper left")
ax_d.set_xlabel("Hour of Day"); ax_d.set_xticks(range(0,25,4))
ax_d.set_title(f"{panel_labels[3]}  MacroGridSignal — RegD & LMP (24h)",
               fontsize=9, fontweight="bold")

for ax in [ax_a, ax_b, ax_c, ax_d]:
    ax.grid(True, alpha=0.18, linestyle="--")

fig.suptitle("C2G-Bench Simulator Suite — Fidelity Validation",
             fontsize=12, fontweight="bold", y=1.01)

fig.savefig("paper/figures/fig2_simulators.pdf", dpi=180, bbox_inches="tight")
fig.savefig("paper/figures/fig2_simulators.png", dpi=180, bbox_inches="tight")
print("fig2 saved")
