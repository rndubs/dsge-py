"""
Generate Synthetic Data for NYFed Model 1002.

This script creates synthetic observable data from the NYFed DSGE model
for testing estimation and forecasting without requiring FRED API access.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsge.solvers.linear import solve_linear_model
from models.nyfed_model_1002 import create_nyfed_model


def generate_synthetic_data(
    T: int = 200,
    shock_std: float = 0.01,
    seed: int = 42,
    save_path: str = "data/nyfed_synthetic_data.csv",
):
    """
    Generate synthetic observable data from NYFed model.

    Parameters
    ----------
    T : int
        Number of time periods (default: 200 quarters = 50 years)
    shock_std : float
        Standard deviation of structural shocks
    seed : int
        Random seed for reproducibility
    save_path : str
        Path to save synthetic data CSV

    Returns:
    -------
    pd.DataFrame
        Synthetic observable data
    """
    np.random.seed(seed)

    # Create and solve model
    model = create_nyfed_model()


    # Solve model
    mats = model.system_matrices()

    solution, _info = solve_linear_model(
        Gamma0=mats["Gamma0"],
        Gamma1=mats["Gamma1"],
        Psi=mats["Psi"],
        Pi=mats["Pi"],
        n_states=model.spec.n_states,
    )


    # Simulate states
    n_states = model.spec.n_states
    n_shocks = model.spec.n_shocks

    states = np.zeros((T, n_states))
    shocks = np.random.randn(T, n_shocks) * shock_std

    for t in range(1, T):
        states[t] = solution.C + solution.T @ states[t - 1] + solution.R @ shocks[t]


    # Generate observables
    Z, D = model.measurement_equation()

    observables = states @ Z.T + D


    # Create DataFrame
    start_date = "1960-01-01"
    dates = pd.date_range(start_date, periods=T, freq="Q")

    data = pd.DataFrame(observables, index=dates, columns=model.spec.observable_names)

    # Print summary statistics

    # Save data
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path)


    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic NYFed model data")
    parser.add_argument(
        "--periods", type=int, default=200, help="Number of time periods (default: 200)"
    )
    parser.add_argument(
        "--shock-std", type=float, default=0.01, help="Shock standard deviation (default: 0.01)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output", type=str, default="data/nyfed_synthetic_data.csv", help="Output CSV path"
    )

    args = parser.parse_args()

    data = generate_synthetic_data(
        T=args.periods, shock_std=args.shock_std, seed=args.seed, save_path=args.output
    )
