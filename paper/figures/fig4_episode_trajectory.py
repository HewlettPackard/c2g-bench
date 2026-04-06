"""Figure 4 — Episode trajectory: Rule-Based MPC vs PPO agent."""
import sys, os, glob
sys.path.insert(0, "/lustre/guillant/C2G-Macro")
os.chdir("/lustre/guillant/C2G-Macro")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import gymnasium as gym

plt.rcParams.update({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
BLUE="#2166ac"; RED="#b2182b"; GREEN="#4dac26"; GRAY="#878787"; ORANGE="#d6604d"

from c2g_env.env_low_level import C2GFastEnv
from baselines.rule_based_mpc import RuleBasedController

# ── Try to load PPO model (optional) ─────────────────────────────────────────
ppo_model = None
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model_dirs = sorted(glob.glob("outputs/ppo_default/**/final_model.zip", recursive=True))
    vn_paths   = sorted(glob.glob("outputs/ppo_default/**/vec_normalize.pkl", recursive=True))
    if model_dirs and vn_paths:
        def make_env():
            e = C2GFastEnv()
            return e
        eval_vec = DummyVecEnv([make_env])
        eval_vec = VecNormalize.load(vn_paths[-1], eval_vec)
        eval_vec.training   = False
        eval_vec.norm_reward= False
        ppo_model = PPO.load(model_dirs[-1], env=eval_vec, device="cpu")
        print(f"PPO loaded: {model_dirs[-1]}")
except Exception as e:
    print(f"PPO load failed ({e}), plotting rule-based only")

# ── Run one episode for a given policy ───────────────────────────────────────
def rollout_rule_based():
    env  = C2GFastEnv()
    ctrl = RuleBasedController()
    obs, _ = env.reset(seed=0)
    rows = []
    for _ in range(288):
        action, _ = ctrl.predict(obs.reshape(1, -1))
        obs, rew, term, trunc, info = env.step(action[0])
        rows.append(info)
        if term or trunc:
            break
    env.close()
    return rows

def rollout_ppo(model, vn):
    env = C2GFastEnv()
    obs_raw, _ = env.reset(seed=0)
    vn.reset()
    rows = []
    raw_obs = obs_raw
    for _ in range(288):
        obs_t = vn.normalize_obs(raw_obs.reshape(1, -1))
        action, _ = model.predict(obs_t, deterministic=True)
        obs_raw, rew, term, trunc, info = env.step(action[0])
        rows.append(info)
        if term or trunc:
            break
    env.close()
    return rows

print("Running rule-based rollout...")
rb_rows = rollout_rule_based()

ppo_rows = None
if ppo_model is not None:
    print("Running PPO rollout...")
    ppo_rows = rollout_ppo(ppo_model, eval_vec)

def extract(rows, key, default=0.0):
    return np.array([r.get(key, default) for r in rows], dtype=float)

# ── Figure ────────────────────────────────────────────────────────────────────
n_rows = 6
fig, axes = plt.subplots(n_rows, 1, figsize=(11, 11), sharex=True,
                          gridspec_kw={"hspace": 0.08})
t_rb  = np.arange(len(rb_rows)) * 5 / 60   # hours

def plot_both(ax, key, default=0.0, scale=1.0, rb_label="Rule-Based MPC", ppo_label="PPO"):
    y_rb = extract(rb_rows, key, default) * scale
    ax.plot(t_rb, y_rb, color=BLUE, lw=1.8, label=rb_label)
    if ppo_rows is not None:
        t_ppo = np.arange(len(ppo_rows)) * 5 / 60
        y_ppo = extract(ppo_rows, key, default) * scale
        ax.plot(t_ppo, y_ppo, color=RED, lw=1.8, ls="--", label=ppo_label)

# Row 0: RegD signal vs actual BESS dispatch (kW)
ax = axes[0]
regd_rb = extract(rb_rows, "regd_signal_kw")
ax.fill_between(t_rb, 0, regd_rb, where=regd_rb>=0, alpha=0.25, color=GREEN, label="RegD (charge)")
ax.fill_between(t_rb, 0, regd_rb, where=regd_rb<0,  alpha=0.25, color=RED,   label="RegD (discharge)")
ax.plot(t_rb, regd_rb, color=GRAY, lw=1.0, ls=":")
bess_rb = extract(rb_rows, "bess_actual_kw")
ax.plot(t_rb, bess_rb, color=BLUE, lw=2.0, label="BESS dispatch (Rule-Based)")
if ppo_rows:
    t_ppo = np.arange(len(ppo_rows)) * 5 / 60
    ax.plot(t_ppo, extract(ppo_rows, "bess_actual_kw"), color=RED, lw=2, ls="--", label="BESS dispatch (PPO)")
ax.set_ylabel("Power (kW)", fontsize=9)
ax.set_title("24-Hour Episode: Rule-Based MPC vs PPO Agent (C2GFastEnv, default scenario)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=7, loc="upper right", ncol=3)
ax.axhline(0, color=GRAY, lw=0.7)

# Row 1: Tracking error |ΔP|
ax = axes[1]
err_rb = np.abs(extract(rb_rows, "tracking_err_kw"))
ax.plot(t_rb, err_rb, color=BLUE, lw=1.8, label="Rule-Based MPC")
if ppo_rows:
    t_ppo = np.arange(len(ppo_rows)) * 5 / 60
    ax.plot(t_ppo, np.abs(extract(ppo_rows, "tracking_err_kw")), color=RED, lw=1.8, ls="--", label="PPO")
ax.set_ylabel("|Tracking Error| (kW)", fontsize=9)
ax.set_yscale("symlog", linthresh=1e3)
ax.legend(fontsize=8, loc="upper right")

# Row 2: Temperatures zone A & B
ax = axes[2]
tA_rb = extract(rb_rows, "temp_A_C", 20.0)
tB_rb = extract(rb_rows, "temp_B_C", 20.0)
ax.plot(t_rb, tA_rb, color=BLUE,   lw=1.8, label="Zone A (Rule-Based)")
ax.plot(t_rb, tB_rb, color=BLUE,   lw=1.8, ls="--", alpha=0.6, label="Zone B (Rule-Based)")
if ppo_rows:
    t_ppo = np.arange(len(ppo_rows)) * 5 / 60
    ax.plot(t_ppo, extract(ppo_rows, "temp_A_C", 20.0), color=RED, lw=1.8, label="Zone A (PPO)")
    ax.plot(t_ppo, extract(ppo_rows, "temp_B_C", 20.0), color=RED, lw=1.8, ls="--", alpha=0.6, label="Zone B (PPO)")
t_warn = 30.0; t_safe = 27.0
ax.axhline(t_warn, color="firebrick",   lw=1.2, ls=":", label=f"T_warn={t_warn}°C")
ax.axhline(t_safe, color="darkorange", lw=1.2, ls=":", label=f"T_safe={t_safe}°C")
ax.set_ylabel("Temperature (°C)", fontsize=9)
ax.legend(fontsize=7, loc="upper right", ncol=2)

# Row 3: BESS State of Charge
ax = axes[3]
ax.plot(t_rb, extract(rb_rows, "soc") * 100, color=BLUE, lw=1.8, label="Rule-Based MPC")
if ppo_rows:
    t_ppo = np.arange(len(ppo_rows)) * 5 / 60
    ax.plot(t_ppo, extract(ppo_rows, "soc") * 100, color=RED, lw=1.8, ls="--", label="PPO")
ax.axhline(20, color=GRAY, lw=0.8, ls=":", label="SOC min (20%)")
ax.axhline(95, color=GRAY, lw=0.8, ls=":", label="SOC max (95%)")
ax.set_ylabel("BESS SoC (%)", fontsize=9)
ax.set_ylim(0, 105)
ax.legend(fontsize=8, loc="upper right")

# Row 4: LMP (price signal)
ax = axes[4]
lmp_rb = extract(rb_rows, "lmp_usd_per_mwh")
ax.fill_between(t_rb, 0, lmp_rb, alpha=0.25, color=ORANGE)
ax.plot(t_rb, lmp_rb, color=ORANGE, lw=1.5, label="LMP ($/MWh)")
ax.set_ylabel("LMP ($/MWh)", fontsize=9, color=ORANGE)
ax.tick_params(axis="y", labelcolor=ORANGE)
ax.legend(fontsize=8, loc="upper right")

# Row 5: PUE
ax = axes[5]
ax.plot(t_rb, extract(rb_rows, "pue", 1.5), color=BLUE, lw=1.8, label="Rule-Based MPC")
if ppo_rows:
    t_ppo = np.arange(len(ppo_rows)) * 5 / 60
    ax.plot(t_ppo, extract(ppo_rows, "pue", 1.5), color=RED, lw=1.8, ls="--", label="PPO")
ax.axhline(1.0, color=GRAY, lw=0.7, ls="--")
ax.set_ylabel("PUE", fontsize=9)
ax.set_xlabel("Hour of Day", fontsize=10)
ax.set_xticks(range(0, 25, 3))
ax.legend(fontsize=8, loc="upper right")

for a in axes:
    a.grid(True, alpha=0.2, linestyle="--")
    a.set_xlim(0, max(t_rb[-1], 1))

fig.savefig("paper/figures/fig4_episode_trajectory.pdf", dpi=180, bbox_inches="tight")
fig.savefig("paper/figures/fig4_episode_trajectory.png", dpi=180, bbox_inches="tight")
print("fig4 saved")
