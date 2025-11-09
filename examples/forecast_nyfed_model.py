"""
NYFed Model 1002 - Forecasting Example

Generate forecasts from estimated NYFed model with uncertainty bands.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.nyfed_model_1002 import create_nyfed_model
from dsge.solvers.linear import solve_linear_model
from dsge.forecasting import forecast_observables, forecast_from_posterior
from dsge.filters.kalman import kalman_filter


def forecast_nyfed_model(
    data_path: str = 'data/nyfed_synthetic_data.csv',
    posterior_path: Optional[str] = 'results/nyfed_estimation/posterior_samples.csv',
    horizon: int = 20,
    n_paths: int = 1000,
    save_dir: str = 'results/nyfed_forecasts',
    use_posterior: bool = False,
    verbose: bool = True
):
    """
    Generate forecasts for NYFed Model 1002.

    Parameters
    ----------
    data_path : str
        Path to observable data CSV
    posterior_path : str, optional
        Path to posterior samples (if use_posterior=True)
    horizon : int
        Forecast horizon in quarters
    n_paths : int
        Number of simulation paths
    save_dir : str
        Directory to save forecast results
    use_posterior : bool
        If True, incorporate parameter uncertainty from posterior
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Forecast results
    """
    print("="*80)
    print("NYFed Model 1002 - Forecasting")
    print("="*80)

    # Load data
    if verbose:
        print(f"\n1. Loading data from {data_path}...")

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    y = data.values

    if verbose:
        print(f"   Observations: {len(data)}")
        print(f"   Last date: {data.index[-1]}")

    # Create model
    if verbose:
        print("\n2. Creating NYFed model...")

    model = create_nyfed_model()

    # Solve model at calibrated parameters
    if verbose:
        print("\n3. Solving model...")

    mats = model.system_matrices()

    solution, info = solve_linear_model(
        Gamma0=mats['Gamma0'],
        Gamma1=mats['Gamma1'],
        Psi=mats['Psi'],
        Pi=mats['Pi'],
        n_states=model.spec.n_states
    )

    if verbose:
        print(f"   Solution: {info['condition']}")

    # Get measurement equation
    Z, D = model.measurement_equation()

    # Run Kalman filter to get final state
    if verbose:
        print("\n4. Running Kalman filter to estimate final state...")

    T_mat = solution.T
    R_mat = solution.R
    C_vec = solution.C

    H = np.eye(model.spec.n_observables) * 1e-6
    Q = np.eye(model.spec.n_shocks)

    x0 = C_vec
    P0 = np.eye(model.spec.n_states) * 0.1

    # Run Kalman filter (returns KalmanFilter object)
    kf_result = kalman_filter(
        y=y,
        T=T_mat,
        R=R_mat,
        Z=Z,
        D=D,
        Q=Q,
        H=H,
        a0=x0,
        P0=P0
    )

    # Get final filtered state
    x_T = kf_result.filtered_states[-1]

    if verbose:
        print(f"   Log-likelihood: {kf_result.log_likelihood:.2f}")

    # Generate forecasts
    if use_posterior and posterior_path:
        if verbose:
            print(f"\n5. Generating forecasts with posterior uncertainty...")
            print(f"   Loading posterior from {posterior_path}")

        # Load posterior samples
        posterior_df = pd.read_csv(posterior_path)
        posterior_samples = posterior_df.drop('weight', axis=1).values
        weights = posterior_df['weight'].values

        forecast_result = forecast_from_posterior(
            posterior_samples=posterior_samples,
            posterior_weights=weights,
            model=model,
            x_T=x_T,
            horizon=horizon,
            n_forecast_paths=100,
            n_posterior_draws=min(100, len(posterior_samples)),
            seed=42
        )

    else:
        if verbose:
            print(f"\n5. Generating unconditional forecasts...")

        forecast_result = forecast_observables(
            T=T_mat,
            R=R_mat,
            C=C_vec,
            Z=Z,
            D=D,
            x_T=x_T,
            horizon=horizon,
            n_paths=n_paths,
            seed=42
        )

    if verbose:
        print(f"   Forecast horizon: {horizon} quarters")
        print(f"   Simulation paths: {forecast_result.paths.shape[0]}")

    # Create forecast dates
    last_date = pd.to_datetime(data.index[-1])
    forecast_dates = pd.date_range(
        last_date + pd.DateOffset(months=3),
        periods=horizon,
        freq='Q'
    )

    # Save forecasts
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Mean forecast
    mean_df = pd.DataFrame(
        forecast_result.mean,
        index=forecast_dates,
        columns=model.spec.observable_names
    )
    mean_df.to_csv(f'{save_dir}/forecast_mean.csv')

    # Uncertainty bands
    if forecast_result.bands:
        for level, (lower, upper) in forecast_result.bands.items():
            lower_df = pd.DataFrame(
                lower,
                index=forecast_dates,
                columns=model.spec.observable_names
            )
            upper_df = pd.DataFrame(
                upper,
                index=forecast_dates,
                columns=model.spec.observable_names
            )

            lower_df.to_csv(f'{save_dir}/forecast_lower_{int(level*100)}.csv')
            upper_df.to_csv(f'{save_dir}/forecast_upper_{int(level*100)}.csv')

    if verbose:
        print(f"\n✓ Forecasts saved to {save_dir}/")

    # Plot forecasts for key variables
    if verbose:
        print("\n6. Creating forecast plots...")

    key_vars = ['obs_gdp_growth', 'obs_infl_pce', 'obs_ffr']

    fig, axes = plt.subplots(len(key_vars), 1, figsize=(12, 8))

    for i, var in enumerate(key_vars):
        ax = axes[i]

        var_idx = model.spec.observable_names.index(var)

        # Historical data (last 40 periods)
        hist_data = data[var].iloc[-40:]
        ax.plot(hist_data.index, hist_data.values, 'k-', linewidth=2, label='Historical')

        # Forecast mean
        ax.plot(forecast_dates, forecast_result.mean[:, var_idx], 'b-', linewidth=2, label='Forecast')

        # Uncertainty bands
        if forecast_result.bands:
            for level, (lower, upper) in forecast_result.bands.items():
                alpha = 0.3 if level == 0.68 else 0.15
                ax.fill_between(
                    forecast_dates,
                    lower[:, var_idx],
                    upper[:, var_idx],
                    alpha=alpha,
                    color='blue',
                    label=f'{int(level*100)}% band'
                )

        ax.axvline(x=last_date, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel(var)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/forecast_plot.png', dpi=300, bbox_inches='tight')

    if verbose:
        print(f"   Plot saved to {save_dir}/forecast_plot.png")

    print("\n" + "="*80)
    print("Forecasting complete!")
    print("="*80)

    return {
        'forecast': forecast_result,
        'data': data,
        'model': model,
        'forecast_dates': forecast_dates,
    }


if __name__ == "__main__":
    import argparse
    from typing import Optional

    parser = argparse.ArgumentParser(description='Forecast NYFed DSGE Model')

    parser.add_argument('--data', type=str, default='data/nyfed_synthetic_data.csv',
                        help='Path to data CSV file')
    parser.add_argument('--posterior', type=str, default=None,
                        help='Path to posterior samples CSV')
    parser.add_argument('--horizon', type=int, default=20,
                        help='Forecast horizon in quarters')
    parser.add_argument('--paths', type=int, default=1000,
                        help='Number of simulation paths')
    parser.add_argument('--output', type=str, default='results/nyfed_forecasts',
                        help='Output directory')
    parser.add_argument('--use-posterior', action='store_true',
                        help='Incorporate posterior uncertainty')

    args = parser.parse_args()

    results = forecast_nyfed_model(
        data_path=args.data,
        posterior_path=args.posterior,
        horizon=args.horizon,
        n_paths=args.paths,
        save_dir=args.output,
        use_posterior=args.use_posterior,
        verbose=True
    )
