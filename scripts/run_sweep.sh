#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# scripts/run_sweep.sh  —  Full C2G-Bench Training & Evaluation Sweep
# ══════════════════════════════════════════════════════════════════════════════
# Runs: 4 scenarios × {PPO, SAC, RuleBased, Random} × 3 seeds = 48 evaluations
#   - PPO: 300k steps    (~8 min/run  on H100)
#   - SAC: 200k steps    (~6 min/run)
#   - Rule-based: no training, evaluation only
#   - Random: no training, evaluation only
#
# Parallelism: up to $MAX_PARALLEL background jobs.
# Results: results/sweep_results.csv  (one row per run)
#
# Usage:
#   bash scripts/run_sweep.sh              # full sweep
#   bash scripts/run_sweep.sh --dry-run    # print commands only
#   MAX_PARALLEL=16 bash scripts/run_sweep.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PYTHON="${ROOT}/.venv/bin/python3"
RESULTS_DIR="${ROOT}/results"
RESULTS_CSV="${RESULTS_DIR}/sweep_results.csv"
MAX_PARALLEL=${MAX_PARALLEL:-4}
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SCENARIOS=(default scenario_a scenario_b scenario_c)
SEEDS=(1 2 3)

mkdir -p "$RESULTS_DIR"

# ── Write CSV header ─────────────────────────────────────────────────────────
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "algo,scenario,seed,total_reward,mean_reward,mean_pue,max_temp_A,max_temp_B,viol_rate,mean_soc,tracking_rmse_kw,mean_lmp,survived,ep_length,wall_seconds" \
        > "$RESULTS_CSV"
fi

# ── Evaluate a trained/rule-based/random agent ────────────────────────────────
# Writes one CSV row per (algo, scenario, seed)
evaluate() {
    local algo=$1 scenario=$2 seed=$3 model_path=${4:-}
    echo "[eval] ${algo} / ${scenario} / seed=${seed}"

    $PYTHON - "$algo" "$scenario" "$seed" "$model_path" << 'PYEOF'
import sys, time, csv, numpy as np
from pathlib import Path

algo       = sys.argv[1]
scenario   = sys.argv[2]
seed       = int(sys.argv[3])
model_path = sys.argv[4] if sys.argv[4] else None

from c2g_env import C2GFastEnv

env = C2GFastEnv(scenario=scenario)

# Load agent
if algo == "random":
    class RandomAgent:
        def predict(self, obs, deterministic=False):
            return env.action_space.sample(), None
    agent = RandomAgent()
elif algo == "rule_based":
    from baselines.rule_based_mpc import RuleBasedController
    agent = RuleBasedController()
else:
    # RL agent (PPO/SAC) — load from zip
    if algo == "ppo":
        from stable_baselines3 import PPO as AlgoCls
    else:
        from stable_baselines3 import SAC as AlgoCls
    agent = AlgoCls.load(model_path, env=env)

# Run 3 eval episodes and average
N_EVAL = 3
all_metrics = []
for ep in range(N_EVAL):
    obs, _ = env.reset(seed=seed * 1000 + ep)
    ep_reward = 0.0
    temps_A, temps_B, pues, socs = [], [], [], []
    tracking_errs, lmps = [], []
    steps = 0
    survived = True
    t0 = time.time()
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        temps_A.append(info.get("temp_A", 0.0))
        temps_B.append(info.get("temp_B", 0.0))
        pues.append(info.get("pue", 1.0))
        socs.append(info.get("bess_soc", 0.5))
        tracking_errs.append(info.get("tracking_err_kw", 0.0) ** 2)
        lmps.append(info.get("lmp", 0.0))
        steps += 1
        if terminated or truncated:
            survived = not terminated
            break
    wall = time.time() - t0

    all_metrics.append({
        "total_reward":    ep_reward,
        "mean_reward":     ep_reward / max(steps, 1),
        "mean_pue":        np.mean(pues) if pues else 1.0,
        "max_temp_A":      np.max(temps_A) if temps_A else 0.0,
        "max_temp_B":      np.max(temps_B) if temps_B else 0.0,
        "viol_rate":       np.mean([t > 33.0 for t in temps_A]) if temps_A else 0.0,
        "mean_soc":        np.mean(socs) if socs else 0.5,
        "tracking_rmse_kw":np.sqrt(np.mean(tracking_errs)) if tracking_errs else 0.0,
        "mean_lmp":        np.mean(lmps) if lmps else 0.0,
        "survived":        int(survived),
        "ep_length":       steps,
        "wall_seconds":    wall,
    })

# Average across eval episodes
avg = {}
for k in all_metrics[0]:
    vals = [m[k] for m in all_metrics]
    avg[k] = np.mean(vals)

csv_path = Path("results/sweep_results.csv")
# Upsert: remove any prior row for this (algo, scenario, seed) before appending
import pandas as _pd, io as _io
_header = "algo,scenario,seed,total_reward,mean_reward,mean_pue,max_temp_A,max_temp_B,viol_rate,mean_soc,tracking_rmse_kw,mean_lmp,survived,ep_length,wall_seconds"
if csv_path.exists():
    _df = _pd.read_csv(csv_path)
    _df = _df[~((_df.algo == algo) & (_df.scenario == scenario) & (_df.seed == int(seed)))]
else:
    _df = _pd.read_csv(_io.StringIO(_header + "\n"))
with open(csv_path, "w", newline="") as _f:
    _df.to_csv(_f, index=False)
with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        algo, scenario, seed,
        f"{avg['total_reward']:.1f}",
        f"{avg['mean_reward']:.4f}",
        f"{avg['mean_pue']:.4f}",
        f"{avg['max_temp_A']:.2f}",
        f"{avg['max_temp_B']:.2f}",
        f"{avg['viol_rate']:.4f}",
        f"{avg['mean_soc']:.4f}",
        f"{avg['tracking_rmse_kw']:.1f}",
        f"{avg['mean_lmp']:.2f}",
        f"{avg['survived']:.0f}",
        f"{avg['ep_length']:.0f}",
        f"{avg['wall_seconds']:.1f}",
    ])
print(f"  reward={avg['total_reward']:.1f}  pue={avg['mean_pue']:.3f}  "
      f"maxT={avg['max_temp_A']:.1f}/{avg['max_temp_B']:.1f}  "
      f"survived={avg['survived']:.0f}")
PYEOF
}

# ── Job counter for parallelism ──────────────────────────────────────────────
NJOBS=0
wait_for_slots() {
    while (( NJOBS >= MAX_PARALLEL )); do
        wait -n 2>/dev/null || true
        NJOBS=$((NJOBS - 1))
    done
}

echo "═══════════════════════════════════════════════════════════════"
echo "  C2G-Bench Full Training & Evaluation Sweep"
echo "  Scenarios: ${SCENARIOS[*]}"
echo "  Seeds:     ${SEEDS[*]}"
echo "  Algos:     PPO, SAC, RuleBased, Random"
echo "  Parallel:  ${MAX_PARALLEL} jobs"
echo "  Output:    ${RESULTS_CSV}"
echo "═══════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Rule-based & Random (no training needed, fast)
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 1: Evaluating Rule-Based & Random baselines …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for algo in rule_based random; do
            if $DRY_RUN; then
                echo "  [dry-run] evaluate $algo $scenario $seed"
            else
                wait_for_slots
                evaluate "$algo" "$scenario" "$seed" "" &
                NJOBS=$((NJOBS + 1))
            fi
        done
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: PPO training + evaluation
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 2: PPO training (300k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train PPO $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] PPO / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/ppo_${scenario}/seed_${seed}"
            # Run Hydra training
            $PYTHON baselines/train_ppo.py \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            # Find the latest run's final model
            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                MODEL="${LATEST%.zip}"
                evaluate "ppo" "$scenario" "$seed" "$MODEL"
            else
                echo "  [WARN] PPO ${scenario}/seed_${seed}: no final_model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: SAC training + evaluation
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 3: SAC training (200k steps × 12 runs) …"
for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if $DRY_RUN; then
            echo "  [dry-run] train SAC $scenario seed=$seed"
            continue
        fi
        wait_for_slots
        (
            echo "[train] SAC / ${scenario} / seed=${seed}"
            OUT_DIR="outputs/sac_${scenario}/seed_${seed}"
            $PYTHON baselines/train_sac.py \
                algo=sac \
                scenario="${scenario}" \
                experiment.seed="${seed}" \
                hydra.run.dir="${OUT_DIR}/\${now:%Y-%m-%d_%H-%M-%S}" \
                2>&1 | tail -5

            LATEST=$(ls -td "${OUT_DIR}"/*/final_model.zip 2>/dev/null | head -1)
            if [[ -n "$LATEST" ]]; then
                MODEL="${LATEST%.zip}"
                evaluate "sac" "$scenario" "$seed" "$MODEL"
            else
                echo "  [WARN] SAC ${scenario}/seed_${seed}: no final_model found"
            fi
        ) &
        NJOBS=$((NJOBS + 1))
    done
done
wait

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Generate summary table
# ══════════════════════════════════════════════════════════════════════════════
echo -e "\n▸ Phase 4: Generating results summary …"
$PYTHON - << 'PYEOF'
import pandas as pd
from pathlib import Path

csv_path = Path("results/sweep_results.csv")
if not csv_path.exists():
    print("No results CSV found."); exit(1)

df = pd.read_csv(csv_path)
print(f"\nTotal runs: {len(df)}")
print(f"Algos:      {sorted(df.algo.unique())}")
print(f"Scenarios:  {sorted(df.scenario.unique())}")
print()

# Aggregate: mean ± std across seeds
cols = ["total_reward", "mean_pue", "max_temp_A", "tracking_rmse_kw", "survived"]
agg = df.groupby(["scenario", "algo"])[cols].agg(["mean", "std"]).round(2)
print(agg.to_string())
print()

# LaTeX-ready table
print("═══ LaTeX table rows (paste into paper) ═══")
for scenario in ["default", "scenario_a", "scenario_b", "scenario_c"]:
    for algo in ["random", "rule_based", "ppo", "sac"]:
        sub = df[(df.scenario == scenario) & (df.algo == algo)]
        if sub.empty:
            continue
        r   = sub.total_reward
        pue = sub.mean_pue
        trk = sub.tracking_rmse_kw
        vr  = sub.viol_rate
        sr  = sub.survived
        print(f"  {scenario:12} & {algo:12} & "
              f"${r.mean():.0f} \\pm {r.std():.0f}$ & "
              f"${pue.mean():.3f} \\pm {pue.std():.3f}$ & "
              f"${trk.mean():.0f} \\pm {trk.std():.0f}$ & "
              f"${vr.mean():.2f}$ & "
              f"${sr.mean():.2f}$ \\\\")

# Save summary too
summary = df.groupby(["scenario", "algo"])[["total_reward","mean_pue","max_temp_A",
    "tracking_rmse_kw","viol_rate","survived"]].agg(["mean","std"]).round(3)
summary.to_csv("results/sweep_summary.csv")
print(f"\nSummary saved to results/sweep_summary.csv")
PYEOF

echo -e "\n═══════════════════════════════════════════════════════════════"
echo "  Sweep complete.  Results: ${RESULTS_CSV}"
echo "═══════════════════════════════════════════════════════════════"
