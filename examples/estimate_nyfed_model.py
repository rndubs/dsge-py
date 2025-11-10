"""
NYFed Model 1002 - Bayesian Estimation using SMC.

This script estimates the NYFed DSGE model using Sequential Monte Carlo (SMC)
with synthetic or real data.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsge.estimation.smc import SMCSampler
from dsge.filters.kalman import kalman_filter
from models.nyfed_model_1002 import create_nyfed_model


def estimate_nyfed_model(
    data_path: str = "data/nyfed_synthetic_data.csv",
    n_particles: int = 1000,
    n_phi: int = 100,
    save_dir: str = "results/nyfed_estimation",
    subset_params: bool = True,
    verbose: bool = True,
):
    """
    Estimate NYFed Model 1002 using SMC.

    Parameters
    ----------
    data_path : str
        Path to CSV file with observable data
    n_particles : int
        Number of particles for SMC
    n_phi : int
        Number of tempering stages
    save_dir : str
        Directory to save results
    subset_params : bool
        If True, estimate only a subset of parameters (for faster testing)
    verbose : bool
        Print progress information

    Returns:
    -------
    dict
        Estimation results including posterior samples and diagnostics
    """
    # Load data
    if verbose:
        pass

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if verbose:
        pass

    # Convert to numpy array
    y = data.values

    # Create model
    if verbose:
        pass

    model = create_nyfed_model()

    if verbose:
        pass

    # Select parameters to estimate
    if subset_params:
        # Estimate a subset of key parameters for faster testing
        params_to_estimate = [
            "sigma_c",  # Consumption risk aversion
            "h",  # Habit formation
            "psi_1",  # Taylor rule: inflation response
            "psi_2",  # Taylor rule: output gap response
            "rho_r",  # Interest rate smoothing
            "rho_z",  # Technology shock persistence
            "rho_b",  # Preference shock persistence
            "sigma_r",  # Monetary policy shock std
            "sigma_z",  # Technology shock std
            "sigma_b",  # Preference shock std
        ]

        if verbose:
            for _p in params_to_estimate:
                pass
    else:
        # Estimate all parameters with priors
        params_to_estimate = [p.name for p in model.parameters.parameters if p.prior is not None]
        if verbose:
            pass

    # Define log-likelihood function
    def log_likelihood(theta):
        """Compute log-likelihood for parameter vector theta."""
        try:
            # Update model parameters
            param_dict = model.parameters.to_dict()

            # Update only estimated parameters
            for i, param_name in enumerate(params_to_estimate):
                param_dict[param_name] = theta[i]

            # Get system matrices
            mats = model.system_matrices(param_dict)

            # Solve model
            from dsge.solvers.linear import solve_linear_model

            solution, info = solve_linear_model(
                Gamma0=mats["Gamma0"],
                Gamma1=mats["Gamma1"],
                Psi=mats["Psi"],
                Pi=mats["Pi"],
                n_states=model.spec.n_states,
            )

            # Check solution validity
            if solution is None:
                return -np.inf

            # Check stability
            max_eigval = np.max(np.abs(info["eigenvalues"]))
            if max_eigval > 1.05:  # Allow some tolerance
                return -np.inf

            # Get measurement equation
            Z, D = model.measurement_equation(param_dict)

            # Compute log-likelihood using Kalman filter
            T_mat = solution.T
            R_mat = solution.R
            C_vec = solution.C

            # Measurement error covariance (small for synthetic data)
            H = np.eye(model.spec.n_observables) * 1e-6

            # Shock covariance
            Q = np.eye(model.spec.n_shocks)

            # Initial state distribution
            x0 = C_vec
            P0 = np.eye(model.spec.n_states) * 0.1

            # Run Kalman filter
            ll, _, _, _ = kalman_filter(
                y=y, T=T_mat, R=R_mat, C=C_vec, Z=Z, D=D, Q=Q, H=H, x0=x0, P0=P0
            )

            return ll

        except Exception:
            # Return very low likelihood for any errors
            return -np.inf

    # Set up SMC sampler
    if verbose:
        pass

    # Get priors for estimated parameters
    param_objects = [model.parameters[name] for name in params_to_estimate]

    sampler = SMCSampler(
        log_likelihood=log_likelihood,
        priors=[p.prior for p in param_objects],
        n_particles=n_particles,
        n_phi=n_phi,
        target_ess=0.5,
        mutation_steps=1,
    )

    # Run estimation
    if verbose:
        pass

    start_time = time.time()

    results = sampler.sample()

    time.time() - start_time

    if verbose:
        pass

    # Extract results
    posterior_samples = results["particles"]
    weights = results["weights"]
    log_evidence = results["log_evidence"]

    # Compute posterior statistics
    posterior_mean = np.average(posterior_samples, weights=weights, axis=0)
    posterior_std = np.sqrt(
        np.average((posterior_samples - posterior_mean) ** 2, weights=weights, axis=0)
    )

    # Print results
    if verbose:



        for i, param_name in enumerate(params_to_estimate):
            model.parameters[param_name]
            posterior_mean[i]
            posterior_std[i]


    # Save results
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Save posterior samples
    posterior_df = pd.DataFrame(posterior_samples, columns=params_to_estimate)
    posterior_df["weight"] = weights
    posterior_df.to_csv(f"{save_dir}/posterior_samples.csv", index=False)

    # Save summary
    summary = pd.DataFrame(
        {
            "parameter": params_to_estimate,
            "prior_mean": [model.parameters[p].value for p in params_to_estimate],
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
        }
    )
    summary.to_csv(f"{save_dir}/posterior_summary.csv", index=False)

    if verbose:
        pass

    return {
        "posterior_samples": posterior_samples,
        "weights": weights,
        "log_evidence": log_evidence,
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "param_names": params_to_estimate,
        "model": model,
        "data": data,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate NYFed DSGE Model")

    parser.add_argument(
        "--data", type=str, default="data/nyfed_synthetic_data.csv", help="Path to data CSV file"
    )
    parser.add_argument("--particles", type=int, default=1000, help="Number of SMC particles")
    parser.add_argument("--stages", type=int, default=100, help="Number of tempering stages")
    parser.add_argument(
        "--output", type=str, default="results/nyfed_estimation", help="Output directory"
    )
    parser.add_argument(
        "--full", action="store_true", help="Estimate all parameters (default: subset only)"
    )

    args = parser.parse_args()

    results = estimate_nyfed_model(
        data_path=args.data,
        n_particles=args.particles,
        n_phi=args.stages,
        save_dir=args.output,
        subset_params=not args.full,
        verbose=True,
    )
