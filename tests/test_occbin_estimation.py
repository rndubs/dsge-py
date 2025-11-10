"""Tests for OccBin model estimation."""

import numpy as np
import pytest

from dsge import DSGEModel, ModelSpecification, Parameter, Prior
from dsge.estimation.likelihood import log_likelihood_occbin
from dsge.estimation.occbin_estimation import OccBinSMCResults, OccBinSMCSampler, estimate_occbin
from dsge.solvers.linear import solve_linear_model
from dsge.solvers.occbin import OccBinSolver, create_zlb_constraint


class SimpleZLBModel(DSGEModel):
    """
    Simple 1D model for testing ZLB estimation.

    Normal regime: x_t = φ*x_{t-1} + ε_t
    ZLB regime: x_t = 0
    """

    def __init__(self, at_zlb: bool = False) -> None:
        spec = ModelSpecification(
            n_states=1,
            n_controls=0,
            n_shocks=1,
            n_observables=1,
            state_names=["x"],
            shock_names=["eps"],
            observable_names=["x_obs"],
        )
        self.at_zlb = at_zlb
        super().__init__(spec)

    def _setup_parameters(self) -> None:
        """Define parameters."""
        self.parameters.add(
            Parameter(
                name="phi",
                value=0.8,
                description="AR(1) persistence",
                fixed=False,
                bounds=(0.0, 0.99),
                prior=Prior("beta", {"alpha": 16, "beta": 4}),  # mean = 0.8
            )
        )

        self.parameters.add(
            Parameter(
                name="sigma",
                value=0.1,
                description="Shock std",
                fixed=False,
                bounds=(0.01, 0.5),
                prior=Prior("invgamma", {"shape": 2.0, "scale": 0.2}),
            )
        )

    def system_matrices(self, params=None):
        """Construct system matrices."""
        if params is not None:
            self.parameters.set_values(params)

        φ = self.parameters["phi"]

        if self.at_zlb:
            # ZLB: x_t = 0
            Γ0 = np.array([[1.0]])
            Γ1 = np.array([[0.0]])
            Ψ = np.array([[0.0]])
        else:
            # Normal: x_t = φ*x_{t-1} + ε_t
            Γ0 = np.array([[1.0]])
            Γ1 = np.array([[φ]])
            Ψ = np.array([[1.0]])

        Π = np.array([[1e-10]])

        return {"Gamma0": Γ0, "Gamma1": Γ1, "Psi": Ψ, "Pi": Π}

    def measurement_equation(self, params=None):
        """Observe state directly."""
        Z = np.eye(1)
        D = np.zeros(1)
        return Z, D

    def shock_covariance(self, params=None):
        """Shock variance."""
        if params is not None:
            self.parameters.set_values(params)
        σ = self.parameters["sigma"]
        return np.array([[σ**2]])

    def measurement_error_covariance(self, params=None):
        """Tiny measurement error."""
        return np.eye(1) * 1e-8


def simulate_zlb_data(T: int = 50, seed: int = 123) -> np.ndarray:
    """Simulate data from ZLB model."""
    np.random.seed(seed)

    # Create models
    model_normal = SimpleZLBModel(at_zlb=False)
    model_zlb = SimpleZLBModel(at_zlb=True)

    # True parameters
    true_phi = 0.85
    true_sigma = 0.15
    model_normal.parameters["phi"] = true_phi
    model_normal.parameters["sigma"] = true_sigma
    model_zlb.parameters["phi"] = true_phi
    model_zlb.parameters["sigma"] = true_sigma

    # Solve both regimes
    sys_normal = model_normal.system_matrices()
    solution_normal, _ = solve_linear_model(
        sys_normal["Gamma0"], sys_normal["Gamma1"], sys_normal["Psi"], sys_normal["Pi"], n_states=1
    )

    sys_zlb = model_zlb.system_matrices()
    solution_zlb, _ = solve_linear_model(
        sys_zlb["Gamma0"], sys_zlb["Gamma1"], sys_zlb["Psi"], sys_zlb["Pi"], n_states=1
    )

    # Create OccBin solver
    zlb_constraint = create_zlb_constraint(variable_index=0, bound=-0.3)
    occbin = OccBinSolver(solution_normal, solution_zlb, zlb_constraint)

    # Generate shocks - include negative shock to hit ZLB
    shocks = np.random.randn(T, 1) * true_sigma
    shocks[20] = -0.8  # Large negative shock

    # Solve with OccBin
    initial_state = np.zeros(1)
    result = occbin.solve(initial_state, shocks, T)

    return result.states


def test_log_likelihood_occbin() -> None:
    """Test OccBin likelihood evaluation."""
    # Create models
    model_M1 = SimpleZLBModel(at_zlb=False)
    model_M2 = SimpleZLBModel(at_zlb=True)

    # Set parameters
    model_M1.parameters["phi"] = 0.8
    model_M1.parameters["sigma"] = 0.1
    model_M2.parameters["phi"] = 0.8
    model_M2.parameters["sigma"] = 0.1

    # Create constraint
    constraint = create_zlb_constraint(variable_index=0, bound=0.0)

    # Generate simple data
    np.random.seed(42)
    data = np.random.randn(30, 1) * 0.1

    # Evaluate likelihood
    log_lik = log_likelihood_occbin(model_M1, model_M2, constraint, data, max_iter=20)

    # Should be finite
    assert np.isfinite(log_lik)
    # Should be negative (log probability)
    assert log_lik < 0


def test_log_likelihood_occbin_unstable_params() -> None:
    """Test that unstable parameters return -inf."""
    model_M1 = SimpleZLBModel(at_zlb=False)
    model_M2 = SimpleZLBModel(at_zlb=True)

    # Set unstable parameters (phi > 1)
    params = np.array([1.1, 0.1])  # phi=1.1 is explosive
    constraint = create_zlb_constraint(variable_index=0, bound=0.0)

    np.random.seed(42)
    data = np.random.randn(30, 1) * 0.1

    log_lik = log_likelihood_occbin(model_M1, model_M2, constraint, data, params, max_iter=20)

    # Should return -inf for unstable solution
    assert log_lik == -np.inf


def test_occbin_smc_sampler_initialization() -> None:
    """Test OccBin SMC sampler initialization."""
    model_M1 = SimpleZLBModel(at_zlb=False)
    model_M2 = SimpleZLBModel(at_zlb=True)
    constraint = create_zlb_constraint(variable_index=0, bound=0.0)

    sampler = OccBinSMCSampler(
        model_M1=model_M1, model_M2=model_M2, constraint=constraint, n_particles=100
    )

    assert sampler.n_particles == 100
    assert sampler.model_M1 is model_M1
    assert sampler.model_M2 is model_M2
    assert sampler.constraint is constraint


def test_occbin_estimation_basic() -> None:
    """Test basic OccBin estimation with synthetic data."""
    # Generate data
    np.random.seed(456)
    T = 40
    data = simulate_zlb_data(T=T, seed=456)

    # Create models for estimation
    model_M1 = SimpleZLBModel(at_zlb=False)
    model_M2 = SimpleZLBModel(at_zlb=True)

    # Set initial (wrong) parameters
    model_M1.parameters["phi"] = 0.7
    model_M1.parameters["sigma"] = 0.2
    model_M2.parameters["phi"] = 0.7
    model_M2.parameters["sigma"] = 0.2

    # Create constraint
    constraint = create_zlb_constraint(variable_index=0, bound=-0.3)

    # Run estimation with few particles for speed
    results = estimate_occbin(
        model_M1=model_M1,
        model_M2=model_M2,
        constraint=constraint,
        data=data,
        n_particles=50,  # Small for speed
        n_mh_steps=1,
        max_filter_iter=20,
        verbose=False,
    )

    # Check results structure
    assert isinstance(results, OccBinSMCResults)
    assert results.particles.shape == (50, 2)  # 50 particles, 2 params
    assert results.weights.shape == (50,)
    assert np.isfinite(results.log_evidence)
    assert results.n_iterations > 0
    assert 0 <= results.acceptance_rate <= 1

    # Check that weights sum to 1
    assert np.allclose(np.sum(results.weights), 1.0)

    # Check regime diagnostics
    assert results.regime_diagnostics is not None
    assert "n_particles" in results.regime_diagnostics


def test_occbin_estimation_parameter_recovery() -> None:
    """Test that estimation recovers approximately correct parameters."""
    # Generate data with known parameters
    np.random.seed(789)
    T = 60
    data = simulate_zlb_data(T=T, seed=789)

    # True parameters: phi=0.85, sigma=0.15

    # Create models
    model_M1 = SimpleZLBModel(at_zlb=False)
    model_M2 = SimpleZLBModel(at_zlb=True)

    # Create constraint
    constraint = create_zlb_constraint(variable_index=0, bound=-0.3)

    # Run estimation
    results = estimate_occbin(
        model_M1=model_M1,
        model_M2=model_M2,
        constraint=constraint,
        data=data,
        n_particles=100,  # More particles for better recovery
        n_mh_steps=2,
        max_filter_iter=20,
        verbose=False,
    )

    # Compute posterior mean
    posterior_mean = np.average(results.particles, weights=results.weights, axis=0)
    phi_est = posterior_mean[0]
    sigma_est = posterior_mean[1]

    # Should be reasonably close to true values (allowing for uncertainty)
    # With small sample and few particles, we just check they're in reasonable range
    assert 0.5 < phi_est < 0.95, f"phi estimate {phi_est} out of range"
    assert 0.05 < sigma_est < 0.3, f"sigma estimate {sigma_est} out of range"


def test_occbin_estimation_convergence() -> None:
    """Test that SMC converges (reaches phi=1)."""
    # Simple data
    np.random.seed(101)
    data = simulate_zlb_data(T=30, seed=101)

    model_M1 = SimpleZLBModel(at_zlb=False)
    model_M2 = SimpleZLBModel(at_zlb=True)
    constraint = create_zlb_constraint(variable_index=0, bound=-0.3)

    # Run with more stages allowed
    sampler = OccBinSMCSampler(
        model_M1=model_M1,
        model_M2=model_M2,
        constraint=constraint,
        n_particles=50,
        n_phi=50,  # Allow many stages
    )

    results = sampler.sample(data, n_mh_steps=1, verbose=False)

    # Should complete in reasonable number of iterations
    assert results.n_iterations > 0
    assert results.n_iterations <= 50  # Allow using all stages if needed


def test_occbin_estimation_with_no_regime_switch() -> None:
    """Test estimation when constraint never binds."""
    # Generate data that stays positive (never hits ZLB)
    np.random.seed(42)
    T = 30
    data = np.abs(np.random.randn(T, 1) * 0.1)  # All positive

    model_M1 = SimpleZLBModel(at_zlb=False)
    model_M2 = SimpleZLBModel(at_zlb=True)

    # Set bound very low so it never binds
    constraint = create_zlb_constraint(variable_index=0, bound=-10.0)

    results = estimate_occbin(
        model_M1=model_M1,
        model_M2=model_M2,
        constraint=constraint,
        data=data,
        n_particles=50,
        n_mh_steps=1,
        verbose=False,
    )

    # Should still work
    assert isinstance(results, OccBinSMCResults)
    assert np.isfinite(results.log_evidence)


def test_estimate_occbin_function() -> None:
    """Test the convenience function estimate_occbin."""
    np.random.seed(111)
    data = simulate_zlb_data(T=30, seed=111)

    model_M1 = SimpleZLBModel(at_zlb=False)
    model_M2 = SimpleZLBModel(at_zlb=True)
    constraint = create_zlb_constraint(variable_index=0, bound=-0.3)

    # Use convenience function
    results = estimate_occbin(model_M1, model_M2, constraint, data, n_particles=50, verbose=False)

    assert isinstance(results, OccBinSMCResults)
    assert results.particles.shape[0] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
