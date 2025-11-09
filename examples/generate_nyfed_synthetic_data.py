"""
Generate Synthetic Data for NYFed Model 1002

This script creates synthetic observable data from the NYFed DSGE model
for testing estimation and forecasting without requiring FRED API access.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.nyfed_model_1002 import create_nyfed_model
from dsge.solvers.linear import solve_linear_model


def generate_synthetic_data(
    T: int = 200,
    shock_std: float = 0.01,
    seed: int = 42,
    save_path: str = 'data/nyfed_synthetic_data.csv'
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

    Returns
    -------
    pd.DataFrame
        Synthetic observable data
    """
    print("="*80)
    print("NYFed Model 1002 - Synthetic Data Generation")
    print("="*80)

    np.random.seed(seed)

    # Create and solve model
    print("\n1. Creating NYFed model...")
    model = create_nyfed_model()

    print(f"   States: {model.spec.n_states}")
    print(f"   Observables: {model.spec.n_observables}")
    print(f"   Shocks: {model.spec.n_shocks}")

    # Solve model
    print("\n2. Solving model at calibrated parameters...")
    mats = model.system_matrices()

    solution, info = solve_linear_model(
        Gamma0=mats['Gamma0'],
        Gamma1=mats['Gamma1'],
        Psi=mats['Psi'],
        Pi=mats['Pi'],
        n_states=model.spec.n_states
    )

    print(f"   Solution: {info['condition']}")
    print(f"   Max eigenvalue: {np.max(np.abs(info['eigenvalues'])):.6f}")

    # Simulate states
    print(f"\n3. Simulating {T} periods...")
    n_states = model.spec.n_states
    n_shocks = model.spec.n_shocks

    states = np.zeros((T, n_states))
    shocks = np.random.randn(T, n_shocks) * shock_std

    for t in range(1, T):
        states[t] = solution.C + solution.T @ states[t-1] + solution.R @ shocks[t]

    print(f"   State simulation complete")
    print(f"   Max state value: {np.max(np.abs(states)):.4f}")

    # Generate observables
    print("\n4. Generating observables...")
    Z, D = model.measurement_equation()

    observables = states @ Z.T + D

    print(f"   Observable matrix: {Z.shape}")
    print(f"   Observables generated: {observables.shape}")

    # Create DataFrame
    start_date = '1960-01-01'
    dates = pd.date_range(start_date, periods=T, freq='Q')

    data = pd.DataFrame(
        observables,
        index=dates,
        columns=model.spec.observable_names
    )

    # Print summary statistics
    print("\n5. Summary statistics:")
    print(data.describe())

    # Save data
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path)
        print(f"\n✓ Synthetic data saved to: {save_path}")

    print("\n" + "="*80)
    print("Data generation complete!")
    print("="*80)
    print(f"\nPeriod: {data.index[0]} to {data.index[-1]}")
    print(f"Observations: {len(data)}")
    print(f"Variables: {len(data.columns)}")

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic NYFed model data')
    parser.add_argument('--periods', type=int, default=200,
                        help='Number of time periods (default: 200)')
    parser.add_argument('--shock-std', type=float, default=0.01,
                        help='Shock standard deviation (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='data/nyfed_synthetic_data.csv',
                        help='Output CSV path')

    args = parser.parse_args()

    data = generate_synthetic_data(
        T=args.periods,
        shock_std=args.shock_std,
        seed=args.seed,
        save_path=args.output
    )
