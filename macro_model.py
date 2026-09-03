import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import params as p
import resource_site_b as rsb

def get_abm_trajectory(num_of_robots=None, number_of_steps = None, seed=None):
    xa, xb, xu, grid_A, grid_B, positions, states = rsb.run_two_site_environment(n_robots=num_of_robots, n_steps = number_of_steps, seed=seed)
    return xa, xb, xu

def smoothing(xa, xb, xu, dt=1.0, window=20):
    number_of_windows = len(xa) // window
    smoothed = []
    for arr in (xa, xb, xu):
        trimmed = arr[: number_of_windows * window]
        smoothed.append(trimmed.reshape(number_of_windows, window).mean(axis=1))
    xa_smoothed, xb_smoothed, xu_smoothed = smoothed
    smoothed_time_axis = np.arange(len(xa_smoothed)) * window * dt
    dxa_dt = np.diff(xa_smoothed) / (window * dt)
    dxb_dt = np.diff(xb_smoothed) / (window * dt)
    xa_mid = xa_smoothed[:-1]
    xb_mid = xb_smoothed[:-1]
    xu_mid = xu_smoothed[:-1]
    design_A = np.stack([xu_mid, xa_mid *xu_mid, -xa_mid], axis=1)
    result_A = np.linalg.lstsq(design_A, dxa_dt, rcond=None)
    coeffs_A = result_A[0]
    alpha_A = coeffs_A[0]
    rho_A = coeffs_A[1]
    gamma_A = coeffs_A[2]
    design_B = np.stack([xu_mid, xb_mid *xu_mid, -xb_mid], axis=1)    
    result_B = np.linalg.lstsq(design_B, dxb_dt, rcond=None)
    coeffs_B = result_B[0]
    alpha_B = coeffs_B[0]
    rho_B = coeffs_B[1]
    gamma_B = coeffs_B[2]
    
    alpha_A= max(alpha_A,0)
    rho_A = max(rho_A,0)
    gamma_A = max(gamma_A, 0)
    alpha_B= max(alpha_B,0)
    rho_B = max(rho_B,0)
    gamma_B = max(gamma_B, 0)
    
    params_fit = dict(alpha_A=alpha_A, rho_A=rho_A, gamma_A=gamma_A,alpha_B=alpha_B, rho_B=rho_B, gamma_B=gamma_B)
    return params_fit, ( smoothed_time_axis, xa_smoothed, xb_smoothed, xu_smoothed)


def macro_odes(t, y, params_fit):
    xa = y[0]
    xb = y[1]
    xu = 1-xa-xb
    dxa = params_fit["alpha_A"] * xu + params_fit["rho_A"] * xa * xu - params_fit["gamma_A"] * xa
    dxb =  params_fit["alpha_B"] * xu + params_fit["rho_B"] * xb * xu - params_fit["gamma_B"] * xb
    return [dxa, dxb]

def run_model(params_fit, time_span, y0, num_points=500):
    time_evaluation = np.linspace(time_span[0], time_span[1], num_points)
    sol = solve_ivp(macro_odes, time_span, y0, args = (params_fit,), t_eval=time_evaluation, method="RK45")
    xa_ode = sol.y[0]
    xb_ode = sol.y[1]
    xu_ode = 1 - xa_ode - xb_ode
    return sol.t, xa_ode, xb_ode, xu_ode

def run_and_plot(save_path="macro_model_results.png"):
    print("Regenerating ABM trajectory...")
    xA, xB, xU = get_abm_trajectory(num_of_robots=p.NUMBER_OF_ROBOTS,
                                     number_of_steps=p.NUMBER_OF_STEPS, seed=p.RNG_SEED)

    print("Fitting macro parameters via least squares...")
    params_fit, (t_s, xA_s, xB_s, xU_s) = smoothing(xA, xB, xU, dt=1.0, window=20)

    print("\nFitted macroscopic parameters:")
    for k, v in params_fit.items():
        print(f"  {k} = {v:.5f}")

    print("\nIntegrating macro ODE with fitted parameters...")
    y0 = [xA[0], xB[0]]
    t_ode, xA_ode, xB_ode, xU_ode = run_model(params_fit, (0, p.NUMBER_OF_STEPS), y0)

    fig, ax = plt.subplots(figsize=(9, 6))
    t_abm = np.arange(len(xA))
    ax.plot(t_abm, xA, color="tab:green", alpha=0.35, linewidth=1, label="ABM: xA (raw)")
    ax.plot(t_abm, xB, color="tab:red", alpha=0.35, linewidth=1, label="ABM: xB (raw)")
    ax.plot(t_abm, xU, color="tab:gray", alpha=0.25, linewidth=1, label="ABM: xU (raw)")
    ax.plot(t_ode, xA_ode, color="tab:green", linewidth=2.5, linestyle="--", label="ODE: xA (fitted)")
    ax.plot(t_ode, xB_ode, color="tab:red", linewidth=2.5, linestyle="--", label="ODE: xB (fitted)")
    ax.plot(t_ode, xU_ode, color="tab:gray", linewidth=2.5, linestyle="--", label="ODE: xU (fitted)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Fraction of population")
    ax.set_title("Micro (ABM) vs Macro (ODE) Validation")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    xA_ode_interp = np.interp(t_s, t_ode, xA_ode)
    xB_ode_interp = np.interp(t_s, t_ode, xB_ode)
    mae_A = np.mean(np.abs(xA_s - xA_ode_interp))
    mae_B = np.mean(np.abs(xB_s - xB_ode_interp))
    print(f"Mean absolute error (smoothed ABM vs ODE): xA={mae_A:.4f}, xB={mae_B:.4f}")

    return params_fit, fig