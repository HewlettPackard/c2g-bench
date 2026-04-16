"""
evaluation/generate_ha_plots.py  —  HA Benchmark Visualisation
================================================================
Reads results CSVs and produces:
  1. Pareto frontier  (reward vs. hard-violation rate)
  2. Radar chart      (multi-metric per agent)
  3. LaTeX-ready comparison table
  4. Shield overhead violin plot

Usage
-----
  python evaluation/generate_ha_plots.py                                     # all defaults
  python evaluation/generate_ha_plots.py --csv results/ha_benchmark_results.csv
  python evaluation/generate_ha_plots.py --format pdf --dpi 300
"""
from __future__ import annotations
import argparse, textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Lazy-import matplotlib so tests don't need it
_MPL_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    _MPL_AVAILABLE = True
except ImportError:
    pass


# ── Tier colours ──────────────────────────────────────────────────
TIER_COLOURS = {
    # Tier 1: hard shields (green family)
    "simplex":   "#2ca02c",
    "cbf":       "#17becf",
    "hj":        "#1f77b4",
    "mpcsf":     "#9467bd",
    # Tier 2: soft constraints (orange family)
    "ppo_lag":   "#ff7f0e",
    "cpo":       "#d62728",
    "reward_shaping": "#e377c2",
    # Tier 3: neuro-symbolic (gold)
    "ha_c2g":    "#bcbd22",
    # Tier 3 ablations (teal family)
    "cbm_only":  "#66c2a5",
    "cbm_gate":  "#fc8d62",
    # Standard baselines (grey family)
    "ppo":       "#7f7f7f",
    "sac":       "#aec7e8",
    "random":    "#c7c7c7",
    "rule_based":"#8c564b",
    "bang_bang":  "#c49c94",
    "pid":       "#f7b6d2",
}

TIER_LABELS = {
    "simplex": "Simplex",
    "cbf": "CBF-PPO",
    "hj": "HJ-PPO",
    "mpcsf": "MPCSF-PPO",
    "ppo_lag": "PPO-Lag",
    "cpo": "CPO",
    "reward_shaping": "Shield-RS",
    "ha_c2g": "HA-C2G",
    "cbm_only": "CBM-Only",
    "cbm_gate": "CBM+Gate",
    "ppo": "PPO",
    "sac": "SAC",
    "random": "Random",
    "rule_based": "Rule-Based",
}


def _require_mpl():
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plots.  pip install matplotlib")


def load_results(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a results CSV."""
    df = pd.read_csv(csv_path)
    # Normalise agent column name
    for col in ("agent", "algo"):
        if col in df.columns:
            df = df.rename(columns={col: "agent"})
            break
    return df


# ═══════════════════════════════════════════════════════════════════
# 1. Pareto Frontier: Reward vs. Hard Violation Rate
# ═══════════════════════════════════════════════════════════════════

def plot_pareto(
    df: pd.DataFrame,
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 200,
    scenario: Optional[str] = None,
):
    """Scatter plot of mean_reward vs. hard_violation_rate with Pareto front."""
    _require_mpl()

    sub = df.copy()
    if scenario:
        sub = sub[sub.scenario == scenario]

    # Group by agent → mean ± std
    reward_col = "mean_reward" if "mean_reward" in sub.columns else "total_reward"
    viol_col = "hard_violation_rate" if "hard_violation_rate" in sub.columns else "thermal_viol_rate"

    if viol_col not in sub.columns:
        print(f"  SKIP pareto: no '{viol_col}' column")
        return

    grp = sub.groupby("agent").agg(
        r_mean=(reward_col, "mean"),
        r_std=(reward_col, "std"),
        v_mean=(viol_col, "mean"),
        v_std=(viol_col, "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in grp.iterrows():
        name = row["agent"]
        col = TIER_COLOURS.get(name, "#333333")
        label = TIER_LABELS.get(name, name)
        ax.errorbar(
            row.v_mean, row.r_mean,
            xerr=row.v_std if not np.isnan(row.v_std) else 0,
            yerr=row.r_std if not np.isnan(row.r_std) else 0,
            fmt="o", color=col, markersize=10,
            capsize=4, label=label, zorder=5,
        )

    # Draw Pareto front
    pts = grp[["v_mean", "r_mean"]].values
    pareto_idx = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i == j:
                continue
            if pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1]:
                if pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1]:
                    dominated = True
                    break
        if not dominated:
            pareto_idx.append(i)

    if pareto_idx:
        pf = pts[pareto_idx]
        order = np.argsort(pf[:, 0])
        ax.plot(pf[order, 0], pf[order, 1], "k--", alpha=0.4, linewidth=1.5,
                label="Pareto front")

    ax.set_xlabel("Hard Violation Rate ↓", fontsize=12)
    ax.set_ylabel("Mean Reward ↑", fontsize=12)
    title = "Pareto Frontier: Safety vs. Performance"
    if scenario:
        title += f" ({scenario})"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"pareto_frontier{'_' + scenario if scenario else ''}.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════
# 2. Radar Chart: Multi-Metric Comparison
# ═══════════════════════════════════════════════════════════════════

def plot_radar(
    df: pd.DataFrame,
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 200,
    agents: Optional[list[str]] = None,
):
    """Radar chart comparing agents across all metrics."""
    _require_mpl()
    from matplotlib.patches import FancyBboxPatch

    # Pick metrics that exist
    metric_candidates = [
        "mean_reward", "total_reward", "tracking_rmse",
        "thermal_viol_rate", "hard_violation_rate",
        "throughput_ratio", "survival_rate", "survived",
        "shield_intervention_rate", "constraint_margin",
        "worst_case_margin", "computational_overhead_ms",
    ]
    metrics = [m for m in metric_candidates if m in df.columns]
    if len(metrics) < 3:
        print("  SKIP radar: fewer than 3 metrics available")
        return

    # Average across scenarios and seeds
    grp = df.groupby("agent")[metrics].mean()

    if agents:
        grp = grp.loc[grp.index.isin(agents)]

    # Normalise each metric to [0, 1] (higher = better for display)
    normalised = grp.copy()
    # For "bad-if-high" metrics, invert
    invert = {"thermal_viol_rate", "hard_violation_rate",
              "shield_intervention_rate", "computational_overhead_ms",
              "tracking_rmse"}
    for col in metrics:
        mn, mx = grp[col].min(), grp[col].max()
        if mx - mn < 1e-10:
            normalised[col] = 0.5
        else:
            normalised[col] = (grp[col] - mn) / (mx - mn)
            if col in invert:
                normalised[col] = 1.0 - normalised[col]

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for agent_name in normalised.index:
        vals = normalised.loc[agent_name].values.tolist()
        vals += vals[:1]
        col = TIER_COLOURS.get(agent_name, "#333333")
        label = TIER_LABELS.get(agent_name, agent_name)
        ax.plot(angles, vals, "o-", linewidth=2, label=label, color=col)
        ax.fill(angles, vals, alpha=0.08, color=col)

    ax.set_xticks(angles[:-1])
    # Pretty labels
    pretty = {
        "mean_reward": "Reward",
        "total_reward": "Total Reward",
        "tracking_rmse": "Tracking ↓",
        "thermal_viol_rate": "Therm. Viol ↓",
        "hard_violation_rate": "Hard Viol ↓",
        "throughput_ratio": "Throughput",
        "survival_rate": "Survival",
        "survived": "Survival",
        "shield_intervention_rate": "Shield Int. ↓",
        "constraint_margin": "Margin",
        "worst_case_margin": "Worst Margin",
        "computational_overhead_ms": "Overhead ↓",
    }
    ax.set_xticklabels([pretty.get(m, m) for m in metrics], fontsize=10)
    ax.set_title("Multi-Metric Radar Comparison", fontsize=14, pad=30)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0.0), fontsize=9)

    fig.tight_layout()
    out = out_dir / f"radar_chart.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════
# 3. Shield Overhead Violin Plot
# ═══════════════════════════════════════════════════════════════════

def plot_overhead(
    df: pd.DataFrame,
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 200,
):
    """Violin plot of computational overhead per shield type."""
    _require_mpl()

    if "computational_overhead_ms" not in df.columns:
        print("  SKIP overhead: no 'computational_overhead_ms' column")
        return

    agents = df.agent.unique()
    data = []
    labels = []
    colours = []
    for a in sorted(agents):
        vals = df[df.agent == a]["computational_overhead_ms"].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(TIER_LABELS.get(a, a))
            colours.append(TIER_COLOURS.get(a, "#333333"))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data, positions=range(len(data)), showmeans=True,
                          showmedians=True)

    for i, pc in enumerate(parts.get("bodies", [])):
        pc.set_facecolor(colours[i])
        pc.set_alpha(0.6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Overhead (ms/step)")
    ax.set_title("Computational Overhead by Shield Type")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"shield_overhead.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════
# 4. LaTeX Comparison Table
# ═══════════════════════════════════════════════════════════════════

def generate_latex_table(df: pd.DataFrame, out_dir: Path):
    """Generate a LaTeX-ready comparison table."""
    metrics = []
    candidate_pairs = [
        ("mean_reward", "Reward"),
        ("hard_violation_rate", "Violation"),
        ("thermal_viol_rate", "Therm. Viol"),
        ("shield_intervention_rate", "Shield Int."),
        ("constraint_margin", "Margin"),
        ("survival_rate", "Survival"),
        ("survived", "Survival"),
        ("computational_overhead_ms", "Overhead (ms)"),
    ]
    for col, name in candidate_pairs:
        if col in df.columns:
            metrics.append((col, name))

    if not metrics:
        print("  SKIP latex: no metrics found")
        return

    grp = df.groupby("agent")

    # Build table
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{High-Assurance Benchmark Results (mean $\\pm$ std across scenarios and seeds)}")
    lines.append("\\label{tab:ha_benchmark}")
    col_spec = "l" + "c" * len(metrics)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header = "Agent & " + " & ".join(name for _, name in metrics) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Sort agents by tier
    tier_order = [
        "random", "rule_based", "bang_bang", "pid",
        "ppo", "sac",
        "ppo_lag", "cpo", "reward_shaping",
        "simplex", "cbf", "hj", "mpcsf",
        "ha_c2g",
    ]
    agent_names = sorted(df.agent.unique(), key=lambda x: tier_order.index(x) if x in tier_order else 99)

    for agent in agent_names:
        sub = df[df.agent == agent]
        label = TIER_LABELS.get(agent, agent)
        cells = [label]
        for col, _ in metrics:
            mean = sub[col].mean()
            std = sub[col].std()
            if std is None or np.isnan(std):
                cells.append(f"${mean:.3f}$")
            else:
                cells.append(f"${mean:.3f} \\pm {std:.3f}$")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    text = "\n".join(lines)
    out = out_dir / "ha_benchmark_table.tex"
    out.write_text(text)
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════
# 5. Per-Scenario Bar Chart
# ═══════════════════════════════════════════════════════════════════

def plot_per_scenario_bars(
    df: pd.DataFrame,
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 200,
):
    """Grouped bar chart: reward by agent, grouped by scenario."""
    _require_mpl()

    reward_col = "mean_reward" if "mean_reward" in df.columns else "total_reward"
    if reward_col not in df.columns:
        return

    scenarios = sorted(df.scenario.unique())
    agents = sorted(df.agent.unique())
    n_agents = len(agents)
    n_scenarios = len(scenarios)

    fig, ax = plt.subplots(figsize=(max(12, n_agents * 1.5), 6))
    x = np.arange(n_scenarios)
    width = 0.8 / n_agents

    for i, agent in enumerate(agents):
        vals = []
        for s in scenarios:
            sub = df[(df.agent == agent) & (df.scenario == s)]
            vals.append(sub[reward_col].mean() if len(sub) > 0 else 0)
        col = TIER_COLOURS.get(agent, "#333333")
        label = TIER_LABELS.get(agent, agent)
        ax.bar(x + i * width, vals, width, label=label, color=col, alpha=0.85)

    ax.set_xticks(x + width * n_agents / 2)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylabel(reward_col.replace("_", " ").title())
    ax.set_title("Performance by Scenario")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"per_scenario_bars.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate HA benchmark plots")
    parser.add_argument(
        "--csv", nargs="+",
        default=["results/ha_benchmark_results.csv", "results/sweep_results.csv"],
        help="Results CSV file(s) to load",
    )
    parser.add_argument("--out_dir", default="results/figures", help="Output directory")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--agents", nargs="+", default=None,
        help="Subset of agents for radar chart",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all CSVs
    dfs = []
    for csv_path in args.csv:
        p = Path(csv_path)
        if p.exists():
            print(f"Loading {p} …")
            dfs.append(load_results(p))
        else:
            print(f"  SKIP: {p} not found")

    if not dfs:
        print("No results CSVs found. Run the benchmark first.")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df)}, Agents: {sorted(df.agent.unique())}\n")

    # Generate all plots
    print("Generating plots …")
    plot_pareto(df, out_dir, args.format, args.dpi)
    for scenario in df.scenario.unique():
        plot_pareto(df, out_dir, args.format, args.dpi, scenario=scenario)
    plot_radar(df, out_dir, args.format, args.dpi, agents=args.agents)
    plot_overhead(df, out_dir, args.format, args.dpi)
    plot_per_scenario_bars(df, out_dir, args.format, args.dpi)
    generate_latex_table(df, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
