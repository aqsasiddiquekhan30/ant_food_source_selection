import params as p
import resource_site_b as rsb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def agreement_time(xa, xb):
    decided= xa+xb 
    with np.errstate(divide='ignore', invalid= "ignore"):
        share_A = np.where(decided > 0, xa / np.where(decided > 0, decided, 1), 0)
        eligible = decided >= p.MIN_DECIDED_RATIO
    agree_A = eligible & (share_A >= p.AGREEMENT_THRESHOLD)
    agree_B = eligible & (share_A <= (1 - p.AGREEMENT_THRESHOLD))
    reached = agree_A | agree_B
    if not np.any(reached):
        return np.nan
    return int(np.argmax(reached))
 
 
def run_experiments(n_values=p.NUM_VALUES, n_seeds=p.NUM_SEEDS, n_steps=None):
    n_steps = p.NUMBER_OF_STEPS if n_steps is None else n_steps
    records = []
    for n in n_values:
        for seed in range(n_seeds):
            xa, xb, xu, *_ = rsb.run_two_site_environment(n_robots=n, n_steps=n_steps, seed=seed)
            agree_time = agreement_time(xa, xb)
            records.append({
                "N": n, "seed": seed,
                "final_xa": xa[-1], "final_xb": xb[-1], "final_xu": xu[-1],
                "agreement_time": agree_time,
                "converged": not np.isnan(agree_time),
                "correct_choice": xa[-1] > xb[-1],
            })
            print(f"N={n:3d} seed={seed:2d}  final_xa={xa[-1]:.2f} "
                  f"final_xb={xb[-1]:.2f} final_xu={xu[-1]:.2f}  agreement_time={agree_time}")
    return pd.DataFrame.from_records(records)
 
 
def summarize(df):
    return df.groupby("N").agg(
        mean_final_xa=("final_xa", "mean"), std_final_xa=("final_xa", "std"),
        mean_final_xb=("final_xb", "mean"), std_final_xb=("final_xb", "std"),
        mean_final_xu=("final_xu", "mean"), std_final_xu=("final_xu", "std"),
        consensus_rate=("converged", "mean"),
        mean_consensus_time=("agreement_time", lambda s: s.dropna().mean() if s.notna().any() else np.nan),
        decision_accuracy=("correct_choice", "mean"),
    ).reset_index()
 
 
def plot_results(df_summary, save_path="experiments_results.png"):
    n_values = df_summary["N"].tolist()
    x = np.arange(len(n_values))
    width = 0.6
 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
 
    axes[0].bar(x, df_summary["mean_final_xa"], width, label="xa", color="tab:green")
    axes[0].bar(x, df_summary["mean_final_xb"], width, bottom=df_summary["mean_final_xa"],
                label="xb", color="tab:red")
    axes[0].bar(x, df_summary["mean_final_xu"], width,
                bottom=df_summary["mean_final_xa"] + df_summary["mean_final_xb"],
                label="xu", color="tab:gray")
    axes[0].set_xticks(x); axes[0].set_xticklabels(n_values)
    axes[0].set_xlabel("Swarm size (N)"); axes[0].set_ylabel("Mean final fraction")
    axes[0].set_title("Final Opinion State vs Swarm Size")
    axes[0].legend(fontsize=8)
 
    axes[1].bar(x, df_summary["decision_accuracy"], width, color="tab:blue")
    axes[1].set_xticks(x); axes[1].set_xticklabels(n_values)
    axes[1].set_xlabel("Swarm size (N)"); axes[1].set_ylabel("Fraction with final xa > final xb")
    axes[1].set_title("Decision Accuracy vs Swarm Size")
    axes[1].set_ylim(0, 1.05)
 
    axes[2].bar(x, df_summary["consensus_rate"], width, color="tab:purple", alpha=0.6)
    axes[2].set_xticks(x); axes[2].set_xticklabels(n_values)
    axes[2].set_xlabel("Swarm size (N)"); axes[2].set_ylabel("Consensus rate")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Consensus Rate vs Swarm Size\n(90% of decided robots agreeing)")
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    return fig