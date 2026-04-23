"""Plot ep/mean_reward and ep/total_reward from an episode_metrics CSV."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="outputs/sac_default/seed_42/2026-04-23_18-16-10/episode_metrics.csv",
        help="Path to episode_metrics.csv",
    )
    parser.add_argument("--output", default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df["episode"], df["ep/mean_reward"], marker="o", markersize=3)
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Episode Mean Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="grey", linestyle="--", linewidth=0.8)

    axes[1].plot(df["episode"], df["ep/total_reward"], marker="o", markersize=3, color="tab:orange")
    axes[1].set_ylabel("Total Reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_title("Episode Total Reward")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.8)

    fig.suptitle("SAC Training Rewards", fontsize=14)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
