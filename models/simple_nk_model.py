"""
Simple 3-Equation New Keynesian Model

A basic New Keynesian model with:
- IS curve (consumption Euler equation)
- Phillips curve
- Taylor rule

This serves as a simpler example before the full NYFed model.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from src.dsge.models.base import DSGEModel, ModelSpecification
from src.dsge.models.parameters import Parameter, ParameterSet, Prior


class SimpleNKModel(DSGEModel):
    """
    Simple 3-equation New Keynesian model.

    State variables: [y_t, pi_t, r_t, y_t-1, pi_t-1, r_t-1, e_y_t, e_pi_t, e_r_t]
    Controls: none (all variables are states for simplicity)
    Shocks: [eps_y, eps_pi, eps_r]

    Equations:
    1. IS curve: y_t = E_t[y_t+1] - 1/sigma * (r_t - E_t[pi_t+1]) + e_y_t
    2. Phillips: pi_t = beta * E_t[pi_t+1] + kappa * y_t + e_pi_t
    3. Taylor rule: r_t = rho_r * r_t-1 + (1-rho_r)*(phi_pi*pi_t + phi_y*y_t) + e_r_t

    Shock processes:
    e_y_t = rho_y * e_y_t-1 + sigma_y * eps_y_t
    e_pi_t = rho_pi * e_pi_t-1 + sigma_pi * eps_pi_t
    e_r_t = sigma_r * eps_r_t
    """

    def __init__(self):
        """Initialize the simple NK model."""
        # Define model dimensions
        # States: y, pi, r, y_lag, pi_lag, r_lag, e_y, e_pi, e_r
        n_states = 9
        n_controls = 0  # All variables treated as states
        n_shocks = 3  # IS shock, AS shock, MP shock
        n_observables = 3  # Output, inflation, rate

        state_names = ['y', 'pi', 'r', 'y_lag', 'pi_lag', 'r_lag', 'e_y', 'e_pi', 'e_r']
        shock_names = ['eps_y', 'eps_pi', 'eps_r']
        observable_names = ['obs_y', 'obs_pi', 'obs_r']

        spec = ModelSpecification(
            n_states=n_states,
            n_controls=n_controls,
            n_shocks=n_shocks,
            n_observables=n_observables,
            state_names=state_names,
            control_names=[],
            shock_names=shock_names,
            observable_names=observable_names
        )

        super().__init__(spec)

    def _setup_parameters(self):
        """Define model parameters."""
        # Household parameters
        self.parameters.add(Parameter(
            name='sigma',
            value=1.5,
            fixed=False,
            description='Intertemporal elasticity of substitution'
        ))

        self.parameters.add(Parameter(
            name='beta',
            value=0.99,
            fixed=True,
            description='Discount factor'
        ))

        # Firm parameters
        self.parameters.add(Parameter(
            name='kappa',
            value=0.1,
            fixed=False,
            description='Slope of Phillips curve'
        ))

        # Policy parameters
        self.parameters.add(Parameter(
            name='phi_pi',
            value=1.5,
            fixed=False,
            description='Taylor rule coefficient on inflation'
        ))

        self.parameters.add(Parameter(
            name='phi_y',
            value=0.5,
            fixed=False,
            description='Taylor rule coefficient on output'
        ))

        self.parameters.add(Parameter(
            name='rho_r',
            value=0.75,
            fixed=False,
            description='Interest rate smoothing'
        ))

        # Shock persistence
        self.parameters.add(Parameter(
            name='rho_y',
            value=0.5,
            fixed=False,
            description='IS shock persistence'
        ))

        self.parameters.add(Parameter(
            name='rho_pi',
            value=0.5,
            fixed=False,
            description='Cost-push shock persistence'
        ))

        # Shock standard deviations
        self.parameters.add(Parameter(
            name='sigma_y',
            value=0.1,
            fixed=False,
            description='IS shock std dev'
        ))

        self.parameters.add(Parameter(
            name='sigma_pi',
            value=0.1,
            fixed=False,
            description='Cost-push shock std dev'
        ))

        self.parameters.add(Parameter(
            name='sigma_r',
            value=0.1,
            fixed=False,
            description='Monetary policy shock std dev'
        ))

    def system_matrices(self, params: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute linearized system matrices.

        System: Γ0 * s_t = Γ1 * s_{t-1} + Ψ * ε_t + Π * η_t

        where s_t = [y_t, pi_t, r_t, y_lag_t, pi_lag_t, r_lag_t, e_y_t, e_pi_t, e_r_t]'
        and η_t = [E_t[y_t+1] - y_t+1, E_t[pi_t+1] - pi_t+1]' are expectation errors
        """
        # Get parameter values
        p = self.parameters.to_dict() if params is None else params
        sigma = p['sigma']
        beta = p['beta']
        kappa = p['kappa']
        phi_pi = p['phi_pi']
        phi_y = p['phi_y']
        rho_r = p['rho_r']
        rho_y = p['rho_y']
        rho_pi = p['rho_pi']
        sigma_y = p['sigma_y']
        sigma_pi = p['sigma_pi']
        sigma_r = p['sigma_r']

        n = self.spec.n_states
        n_shocks = self.spec.n_shocks
        n_eta = 2  # Two expectation errors: E[y_t+1], E[pi_t+1]

        # Initialize matrices
        Gamma0 = np.zeros((n, n))
        Gamma1 = np.zeros((n, n))
        Psi = np.zeros((n, n_shocks))
        Pi = np.zeros((n, n_eta))

        # Equation indices
        idx_y, idx_pi, idx_r = 0, 1, 2
        idx_y_lag, idx_pi_lag, idx_r_lag = 3, 4, 5
        idx_e_y, idx_e_pi, idx_e_r = 6, 7, 8

        # Shock indices
        idx_eps_y, idx_eps_pi, idx_eps_r = 0, 1, 2

        # Expectation error indices
        idx_eta_y, idx_eta_pi = 0, 1

        # ===================================================================
        # Equation 1: IS curve
        # y_t = E_t[y_t+1] - 1/sigma * (r_t - E_t[pi_t+1]) + e_y_t
        # => y_t = E_t[y_t+1] - 1/sigma * r_t + 1/sigma * E_t[pi_t+1] + e_y_t
        # => y_t - E_t[y_t+1] + 1/sigma * r_t - 1/sigma * E_t[pi_t+1] - e_y_t = 0
        # ===================================================================
        Gamma0[idx_y, idx_y] = 1.0
        Gamma0[idx_y, idx_r] = 1.0 / sigma
        Gamma0[idx_y, idx_e_y] = -1.0

        # Expectation terms
        Pi[idx_y, idx_eta_y] = -1.0  # -E_t[y_t+1]
        Pi[idx_y, idx_eta_pi] = -1.0 / sigma  # -1/sigma * E_t[pi_t+1]

        # ===================================================================
        # Equation 2: Phillips curve
        # pi_t = beta * E_t[pi_t+1] + kappa * y_t + e_pi_t
        # => pi_t - beta * E_t[pi_t+1] - kappa * y_t - e_pi_t = 0
        # ===================================================================
        Gamma0[idx_pi, idx_pi] = 1.0
        Gamma0[idx_pi, idx_y] = -kappa
        Gamma0[idx_pi, idx_e_pi] = -1.0

        # Expectation terms
        Pi[idx_pi, idx_eta_pi] = -beta  # -beta * E_t[pi_t+1]

        # ===================================================================
        # Equation 3: Taylor rule
        # r_t = rho_r * r_t-1 + (1-rho_r)*(phi_pi*pi_t + phi_y*y_t) + e_r_t
        # => r_t - rho_r * r_t-1 - (1-rho_r)*phi_pi*pi_t - (1-rho_r)*phi_y*y_t - e_r_t = 0
        # ===================================================================
        Gamma0[idx_r, idx_r] = 1.0
        Gamma0[idx_r, idx_pi] = -(1 - rho_r) * phi_pi
        Gamma0[idx_r, idx_y] = -(1 - rho_r) * phi_y
        Gamma0[idx_r, idx_e_r] = -1.0
        Gamma1[idx_r, idx_r_lag] = rho_r

        # ===================================================================
        # Equations 4-6: Lag definitions
        # y_lag_t = y_t-1
        # pi_lag_t = pi_t-1
        # r_lag_t = r_t-1
        # ===================================================================
        Gamma0[idx_y_lag, idx_y_lag] = 1.0
        Gamma1[idx_y_lag, idx_y] = 1.0

        Gamma0[idx_pi_lag, idx_pi_lag] = 1.0
        Gamma1[idx_pi_lag, idx_pi] = 1.0

        Gamma0[idx_r_lag, idx_r_lag] = 1.0
        Gamma1[idx_r_lag, idx_r] = 1.0

        # ===================================================================
        # Equations 7-9: Shock processes
        # e_y_t = rho_y * e_y_t-1 + sigma_y * eps_y_t
        # e_pi_t = rho_pi * e_pi_t-1 + sigma_pi * eps_pi_t
        # e_r_t = sigma_r * eps_r_t
        # ===================================================================
        Gamma0[idx_e_y, idx_e_y] = 1.0
        Gamma1[idx_e_y, idx_e_y] = rho_y
        Psi[idx_e_y, idx_eps_y] = sigma_y

        Gamma0[idx_e_pi, idx_e_pi] = 1.0
        Gamma1[idx_e_pi, idx_e_pi] = rho_pi
        Psi[idx_e_pi, idx_eps_pi] = sigma_pi

        Gamma0[idx_e_r, idx_e_r] = 1.0
        Psi[idx_e_r, idx_eps_r] = sigma_r

        return {
            'Gamma0': Gamma0,
            'Gamma1': Gamma1,
            'Psi': Psi,
            'Pi': Pi
        }

    def measurement_equation(self, params: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Measurement equation: observables = Z * states + D

        obs_y_t = y_t
        obs_pi_t = pi_t
        obs_r_t = r_t
        """
        n_obs = self.spec.n_observables
        n_states = self.spec.n_states

        Z = np.zeros((n_obs, n_states))
        D = np.zeros(n_obs)

        # Output observable
        Z[0, 0] = 1.0  # obs_y = y

        # Inflation observable
        Z[1, 1] = 1.0  # obs_pi = pi

        # Interest rate observable
        Z[2, 2] = 1.0  # obs_r = r

        return Z, D

    def steady_state(self, params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute steady state.

        In this log-linearized model, all variables are deviations from steady state,
        so the steady state is zero.
        """
        return np.zeros(self.spec.n_states)


def create_simple_nk_model() -> SimpleNKModel:
    """Factory function to create a Simple NK model instance."""
    return SimpleNKModel()


if __name__ == "__main__":
    # Example usage
    model = create_simple_nk_model()
    print(f"Model: Simple New Keynesian")
    print(f"States: {model.spec.n_states}")
    print(f"Shocks: {model.spec.n_shocks}")
    print(f"Observables: {model.spec.n_observables}")
    print(f"Parameters: {len(model.parameters)}")

    # Test system matrices
    mats = model.system_matrices()
    print(f"\nGamma0 shape: {mats['Gamma0'].shape}")
    print(f"Gamma1 shape: {mats['Gamma1'].shape}")
    print(f"Psi shape: {mats['Psi'].shape}")
    print(f"Pi shape: {mats['Pi'].shape}")

    # Test measurement equation
    Z, D = model.measurement_equation()
    print(f"\nZ shape: {Z.shape}")
    print(f"D shape: {D.shape}")
