import os
import numpy as np
import matplotlib.pyplot as plt 
import params as p
import basic_environment as be
import resource_site_a as rsa
import resource_site_b as rsb
import macro_model as m
import anaylsis_reports as ar

def if_exist_output_dir():
    os.makedirs(p.OUTPUT_DIR, exist_ok=True)
    
def run_basic_env_plots():
    print("\n=== Stage 1: basic environment (single site, no opinions) ===")
    found_ratio, traj, final_positions = be.run_base_environment()
    print(f"Final fraction that found the site: {found_ratio[-1]:.2%}")
    
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(found_ratio, color="tab:blue")
    axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Fraction found")
    axes[0].set_title("Stage 1: Cumulative Site Discovery")
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)

    axes[1].scatter(final_positions[:, 0], final_positions[:, 1],
                     c="tab:blue", s=20, alpha=0.6, label="Robots")
    axes[1].scatter(*p.RESOURCE_SITE_A, c="tab:red", s=200, marker="*",
                     label="Site A", zorder=5)
    axes[1].plot(traj[:, 0], traj[:, 1], color="gray", alpha=0.3, linewidth=0.7,
                 label="Sample path")
    axes[1].set_xlim(0, p.ARENA_SIZE); axes[1].set_ylim(0, p.ARENA_SIZE)
    axes[1].set_aspect("equal"); axes[1].set_title("Stage 1: Arena Snapshot")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(p.OUTPUT_DIR, "basic_environment.png")
    plt.savefig(path, dpi=150); plt.close(figure)
    print(f"Saved {path}")


def run_resource_site_a_plots():
    print("\n=== Stage 2: single site A with pheromone + recruitment ===")
    frac_A, total_pher, grid, final_positions, final_states = rsa.run_pheronome_environment()
    print(f"Final fraction supporting A: {frac_A[-1]:.2%}")

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(frac_A, color="tab:green")
    axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Fraction supporting A")
    axes[0].set_title("Stage 2: Opinion Dynamics")
    axes[0].set_ylim(0, 1.05); axes[0].grid(alpha=0.3)

    axes[1].plot(total_pher, color="tab:orange")
    axes[1].set_xlabel("Timestep"); axes[1].set_ylabel("Total pheromone")
    axes[1].set_title("Stage 2: Total Pheromone Over Time"); axes[1].grid(alpha=0.3)

    im = axes[2].imshow(grid.T, origin="lower", extent=[0, p.ARENA_SIZE, 0, p.ARENA_SIZE], cmap="Oranges")
    colors = np.where(final_states == p.SUPPORT_A, "green", "blue")
    axes[2].scatter(final_positions[:, 0], final_positions[:, 1],c=colors, s=15, alpha=0.7, edgecolors="k", linewidths=0.3)
    axes[2].scatter(*p.RESOURCE_SITE_A, c="red", s=200, marker="*", zorder=5)
    axes[2].set_xlim(0, p.ARENA_SIZE); axes[2].set_ylim(0, p.ARENA_SIZE)
    axes[2].set_aspect("equal")
    axes[2].set_title("Stage 2: Final Pheromone Field + States")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(p.OUTPUT_DIR, "single_site_pheromone.png")
    plt.savefig(path, dpi=150); plt.close(figure)
    print(f"Saved {path}")
        
        
def run_resource_site_b_plot():
    print("\n=== Stage 3: two competing sites (A vs B) ===")
    xA, xB, xU, grid_A, grid_B, final_positions, final_states = rsb.run_two_site_environment(
        n_robots=p.NUMBER_OF_ROBOTS, n_steps=p.NUMBER_OF_STEPS, seed=p.RNG_SEED)
    print(f"Final: xA={xA[-1]:.2%}, xB={xB[-1]:.2%}, xU={xU[-1]:.2%}")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(xA, color="tab:green", label="xA (Support A)")
    axes[0].plot(xB, color="tab:red", label="xB (Support B)")
    axes[0].plot(xU, color="tab:gray", label="xU (Undecided)", alpha=0.6)
    axes[0].set_xlabel("Timestep"); axes[0].set_ylabel("Fraction of population")
    axes[0].set_title("Stage 3: Collective Opinion Dynamics")
    axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    combined = grid_A - grid_B
    vmax = max(np.abs(combined).max(), 1e-6)
    im = axes[1].imshow(combined.T, origin="lower", extent=[0, p.ARENA_SIZE, 0, p.ARENA_SIZE],cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    colors = np.array(["tab:blue", "tab:green", "tab:red"])[final_states]
    axes[1].scatter(final_positions[:, 0], final_positions[:, 1],c=colors, s=15, alpha=0.8, edgecolors="k", linewidths=0.3)
    axes[1].scatter(*p.RESOURCE_SITE_A, c="darkgreen", s=200, marker="*", zorder=5, label="Site A")
    axes[1].scatter(*p.RESOURCE_SITE_B, c="darkred", s=200, marker="*", zorder=5, label="Site B")
    axes[1].set_xlim(0, p.ARENA_SIZE); axes[1].set_ylim(0, p.ARENA_SIZE)
    axes[1].set_aspect("equal"); axes[1].set_title("Stage 3: Pheromone Balance")
    axes[1].legend(loc="upper center", fontsize=8, bbox_to_anchor=(0.5, -0.08), ncol=2)

    plt.tight_layout()
    path = os.path.join(p.OUTPUT_DIR, "two_site_competition.png")
    plt.savefig(path, dpi=150); plt.close(figure)
    print(f"Saved {path}")


def model_plot():
    print("\n=== Stage 4: macro model fit + validation ===")
    path = os.path.join(p.OUTPUT_DIR, "macro_validation.png")
    params_fit, figure = m.run_and_plot(save_path=path)
    plt.close(figure)
    print(f"Saved {path}")
    return params_fit


def run_analysis_and_plot():
    print("\n=== Stage 5: swarm-size experiments ===")
    df_raw = ar.run_experiments()
    df_summary = ar.summarize(df_raw)

    df_raw.to_csv(os.path.join(p.OUTPUT_DIR, "stage5_raw_results.csv"), index=False)
    df_summary.to_csv(os.path.join(p.OUTPUT_DIR, "stage5_summary_results.csv"), index=False)

    print("\n=== Stage 5 summary table ===")
    print(df_summary.to_string(index=False))

    path = os.path.join(p.OUTPUT_DIR, "stage5_swarm_size_experiments.png")
    figure = ar.plot_results(df_summary, save_path=path)
    plt.close(figure)
    print(f"Saved {path}")


if __name__ == "__main__":
    if_exist_output_dir()
    run_basic_env_plots()
    run_resource_site_a_plots()
    run_resource_site_b_plot()
    model_plot()
    run_analysis_and_plot()
    print(f"\nAll stages complete. Outputs saved in ./{p.OUTPUT_DIR}/")
