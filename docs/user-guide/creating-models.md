# Creating Custom DSGE Models

This guide shows you how to implement your own DSGE model using the framework. We'll progress from simple to complex examples.

## Model Specification Workflow

1. **Define the economic model**: Write down equilibrium conditions
2. **Log-linearize**: Derive log-linear system around steady state
3. **Express in canonical form**: Convert to Γ₀ E[x_t] = Γ₁ x_{t-1} + Ψ ε_t + Π η_t
4. **Implement `DSGEModel` class**: Code the model in Python
5. **Test**: Verify solution, simulate, compute impulse responses

## Example 1: AR(1) Model

Let's start with the simplest model: an autoregressive process.

### Economic Model

$$x_t = \rho x_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma^2)$$

### Canonical Form

This is already linear, so:
- Γ₀ = [1]
- Γ₁ = [ρ]
- Ψ = [1]
- Π = [0] (no expectational errors)

### Implementation

```python
import numpy as np
from dsge import DSGEModel, ModelSpecification, Parameter, Prior

class AR1Model(DSGEModel):
    """Simple AR(1) process for demonstration."""

    def __init__(self):
        # Define model dimensions
        spec = ModelSpecification(
            n_states=1,          # Just x_t
            n_controls=0,        # No forward-looking variables
            n_shocks=1,          # One structural shock
            n_observables=1,     # Observe x_t directly
            state_names=['x'],
            shock_names=['eps'],
            observable_names=['y']
        )
        super().__init__(spec)
        self.name = "AR(1) Model"

    def _setup_parameters(self):
        """Define parameters with priors."""

        # Persistence parameter: ρ ∈ (0, 1)
        self.parameters.add(Parameter(
            name='rho',
            value=0.9,
            bounds=(0.0, 0.999),
            fixed=False,
            prior=Prior('beta', {'alpha': 18, 'beta': 2})
        ))

        # Shock standard deviation: σ > 0
        self.parameters.add(Parameter(
            name='sigma',
            value=0.1,
            bounds=(0.001, 1.0),
            fixed=False,
            prior=Prior('invgamma', {'shape': 2.0, 'scale': 0.2})
        ))

    def system_matrices(self, params=None):
        """Return Γ₀, Γ₁, Ψ, Π matrices."""

        if params is not None:
            self.parameters.set_values(params)

        rho = self.parameters['rho']

        return {
            'Gamma0': np.array([[1.0]]),
            'Gamma1': np.array([[rho]]),
            'Psi': np.array([[1.0]]),
            'Pi': np.array([[1e-10]])  # Numerical zero for stability
        }

    def measurement_equation(self, params=None):
        """Return Z matrix and D vector: y_t = Z x_t + D"""

        # Perfect observation: y_t = x_t
        Z = np.array([[1.0]])
        D = np.array([0.0])
        return Z, D

    def shock_covariance(self, params=None):
        """Return shock covariance matrix Q."""

        if params is not None:
            self.parameters.set_values(params)

        sigma = self.parameters['sigma']
        return np.array([[sigma**2]])
```

### Testing

```python
from dsge.solvers.linear import solve_linear_model, simulate

# Create model
model = AR1Model()

# Solve
mats = model.system_matrices()
solution, info = solve_linear_model(
    mats['Gamma0'], mats['Gamma1'],
    mats['Psi'], mats['Pi'],
    n_states=1
)

print(f"Stable: {solution.is_stable}")
print(f"Eigenvalue: {solution.eigenvalues[0]:.3f}")  # Should be 0.9

# Simulate
solution.Z, solution.D = model.measurement_equation()
solution.Q = model.shock_covariance()

states, obs = simulate(solution, n_periods=100, random_seed=42)

import matplotlib.pyplot as plt
plt.plot(obs)
plt.title('AR(1) Simulation')
plt.xlabel('Time')
plt.ylabel('x_t')
plt.show()
```

## Example 2: Three-Equation New Keynesian Model

Now let's implement a standard New Keynesian model with forward-looking expectations.

### Economic Model

The canonical 3-equation NK model:

**IS Curve (output gap):**
$$y_t = E_t[y_{t+1}] - \sigma (r_t - E_t[\pi_{t+1}]) + \varepsilon_t^y$$

**Phillips Curve (inflation):**
$$\pi_t = \beta E_t[\pi_{t+1}] + \kappa y_t + \varepsilon_t^\pi$$

**Taylor Rule (monetary policy):**
$$r_t = \rho_r r_{t-1} + (1-\rho_r)(\phi_\pi \pi_t + \phi_y y_t) + \varepsilon_t^r$$

### State Vector

We need to track:
- Current variables: $y_t, \pi_t, r_t$
- Lags: $r_{t-1}$
- Shocks: $\varepsilon_t^y, \varepsilon_t^\pi, \varepsilon_t^r$

Total states: 7

### Implementation

```python
class SimpleNKModel(DSGEModel):
    """3-equation New Keynesian model."""

    def __init__(self):
        spec = ModelSpecification(
            n_states=9,  # y, π, r, r_lag, eps_y, eps_pi, eps_r, y_lag, pi_lag
            n_controls=0,
            n_shocks=3,
            n_observables=3,
            state_names=['y', 'pi', 'r', 'r_lag',
                         'eps_y', 'eps_pi', 'eps_r',
                         'y_lag', 'pi_lag'],
            shock_names=['shock_y', 'shock_pi', 'shock_r'],
            observable_names=['obs_y', 'obs_pi', 'obs_r']
        )
        super().__init__(spec)
        self.name = "Simple NK Model"

    def _setup_parameters(self):
        # IS curve
        self.parameters.add(Parameter(
            'sigma', 1.0, (0.1, 10.0), False,
            Prior('gamma', {'shape': 2.0, 'rate': 2.0})
        ))

        # Phillips curve
        self.parameters.add(Parameter(
            'beta', 0.99, fixed=True  # Discount factor (calibrated)
        ))
        self.parameters.add(Parameter(
            'kappa', 0.1, (0.01, 1.0), False,
            Prior('gamma', {'shape': 2.0, 'rate': 20.0})
        ))

        # Taylor rule
        self.parameters.add(Parameter(
            'phi_pi', 1.5, (1.0, 3.0), False,
            Prior('normal', {'mean': 1.5, 'std': 0.25})
        ))
        self.parameters.add(Parameter(
            'phi_y', 0.5, (0.0, 2.0), False,
            Prior('gamma', {'shape': 4.0, 'rate': 8.0})
        ))
        self.parameters.add(Parameter(
            'rho_r', 0.8, (0.0, 0.99), False,
            Prior('beta', {'alpha': 16, 'beta': 4})
        ))

        # Shock standard deviations
        for shock in ['y', 'pi', 'r']:
            self.parameters.add(Parameter(
                f'sigma_{shock}', 0.01, (0.001, 0.1), False,
                Prior('invgamma', {'shape': 2.0, 'scale': 0.02})
            ))

    def system_matrices(self, params=None):
        if params is not None:
            self.parameters.set_values(params)

        # Extract parameters
        sigma = self.parameters['sigma']
        beta = self.parameters['beta']
        kappa = self.parameters['kappa']
        phi_pi = self.parameters['phi_pi']
        phi_y = self.parameters['phi_y']
        rho_r = self.parameters['rho_r']

        n = 9
        Gamma0 = np.zeros((n, n))
        Gamma1 = np.zeros((n, n))
        Psi = np.zeros((n, 3))
        Pi = np.zeros((n, 3))

        # Equation 1: IS curve
        # y_t = E[y_{t+1}] - σ(r_t - E[π_{t+1}]) + ε_y
        Gamma0[0, 0] = 1.0          # y_t
        Gamma0[0, 2] = sigma        # σ r_t
        Pi[0, 0] = -1.0             # -E[y_{t+1}]
        Pi[0, 1] = sigma            # σ E[π_{t+1}]
        Gamma0[0, 4] = -1.0         # -ε_y

        # Equation 2: Phillips curve
        # π_t = β E[π_{t+1}] + κ y_t + ε_π
        Gamma0[1, 1] = 1.0          # π_t
        Gamma0[1, 0] = -kappa       # -κ y_t
        Pi[1, 1] = -beta            # -β E[π_{t+1}]
        Gamma0[1, 5] = -1.0         # -ε_π

        # Equation 3: Taylor rule
        # r_t = ρ_r r_{t-1} + (1-ρ_r)(φ_π π_t + φ_y y_t) + ε_r
        Gamma0[2, 2] = 1.0                          # r_t
        Gamma1[2, 3] = rho_r                        # ρ_r r_{t-1}
        Gamma0[2, 1] = -(1 - rho_r) * phi_pi       # -(1-ρ_r)φ_π π_t
        Gamma0[2, 0] = -(1 - rho_r) * phi_y        # -(1-ρ_r)φ_y y_t
        Gamma0[2, 6] = -1.0                         # -ε_r

        # Equation 4: Lag of interest rate
        Gamma0[3, 3] = 1.0          # r_{t-1}^t
        Gamma1[3, 2] = 1.0          # r_{t-1}

        # Equations 5-7: Shock processes (AR(1) with persistence = 0)
        # ε_y,t = shock_y,t
        Gamma0[4, 4] = 1.0
        Psi[4, 0] = 1.0

        # ε_π,t = shock_π,t
        Gamma0[5, 5] = 1.0
        Psi[5, 1] = 1.0

        # ε_r,t = shock_r,t
        Gamma0[6, 6] = 1.0
        Psi[6, 2] = 1.0

        # Equations 8-9: Lags of y and π
        Gamma0[7, 7] = 1.0
        Gamma1[7, 0] = 1.0

        Gamma0[8, 8] = 1.0
        Gamma1[8, 1] = 1.0

        return {
            'Gamma0': Gamma0,
            'Gamma1': Gamma1,
            'Psi': Psi,
            'Pi': Pi
        }

    def measurement_equation(self, params=None):
        # Observe y, π, r with measurement error
        Z = np.zeros((3, 9))
        Z[0, 0] = 1.0  # obs_y = y
        Z[1, 1] = 1.0  # obs_π = π
        Z[2, 2] = 1.0  # obs_r = r
        D = np.zeros(3)
        return Z, D

    def shock_covariance(self, params=None):
        if params is not None:
            self.parameters.set_values(params)

        Q = np.diag([
            self.parameters['sigma_y']**2,
            self.parameters['sigma_pi']**2,
            self.parameters['sigma_r']**2
        ])
        return Q
```

### Computing Impulse Responses

```python
from dsge.solvers.linear import compute_irf

# Solve model
model = SimpleNKModel()
mats = model.system_matrices()
solution, info = solve_linear_model(
    mats['Gamma0'], mats['Gamma1'],
    mats['Psi'], mats['Pi'],
    n_states=9
)

# Compute IRF to monetary policy shock
shock_index = 2  # Monetary policy shock
irf = compute_irf(solution, shock_index, periods=20, shock_size=1.0)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
variables = ['Output', 'Inflation', 'Interest Rate']

for i, (ax, var) in enumerate(zip(axes, variables)):
    ax.plot(irf[:, i])
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_title(f'{var} Response to MP Shock')
    ax.set_xlabel('Quarters')
    ax.grid(True)

plt.tight_layout()
plt.show()
```

## Example 3: Model with OccBin Constraint

Now let's add a Zero Lower Bound constraint.

### Model Modification

The ZLB constraint is:
$$r_t \geq 0$$

When binding, the Taylor rule is replaced with:
$$r_t = 0$$

### Implementation

```python
class ZLBNKModel(DSGEModel):
    """NK model with optional ZLB constraint."""

    def __init__(self, zlb_binding=False):
        self.zlb_binding = zlb_binding
        # ... (same spec as SimpleNKModel)

    def system_matrices(self, params=None):
        # ... (same as before until Taylor rule)

        if not self.zlb_binding:
            # Normal Taylor rule (same as before)
            Gamma0[2, 2] = 1.0
            Gamma1[2, 3] = rho_r
            Gamma0[2, 1] = -(1 - rho_r) * phi_pi
            Gamma0[2, 0] = -(1 - rho_r) * phi_y
            Gamma0[2, 6] = -1.0
        else:
            # ZLB binding: r_t = 0
            Gamma0[2, 2] = 1.0
            # No other terms - rate is fixed at zero

        # ... (rest of equations same)
```

### Using OccBin Solver

```python
from dsge.solvers.occbin import OccBinSolver, create_zlb_constraint

# Create both regime models
model_M1 = ZLBNKModel(zlb_binding=False)  # Normal
model_M2 = ZLBNKModel(zlb_binding=True)   # ZLB

# Solve both
solution_M1, _ = solve_linear_model(...)
solution_M2, _ = solve_linear_model(...)

# Create constraint
zlb_constraint = create_zlb_constraint(
    variable_index=2,  # Interest rate is 3rd state
    bound=0.0
)

# Create OccBin solver
solver = OccBinSolver(solution_M1, solution_M2, zlb_constraint)

# Simulate large negative shock
initial_state = np.zeros(9)
shocks = np.zeros((50, 3))
shocks[0, 0] = -5.0  # Large negative demand shock

result = solver.solve(initial_state, shocks, T=50)

# Plot regime-switching paths
fig, axes = plt.subplots(3, 1, figsize=(10, 9))

for i, var in enumerate(['Output', 'Inflation', 'Interest Rate']):
    axes[i].plot(result.paths[:, i])
    axes[i].axhline(0, color='black', linestyle='--', linewidth=0.5)
    axes[i].set_ylabel(var)
    axes[i].grid(True)

    # Shade ZLB periods
    zlb_periods = result.regime_sequence == 1
    axes[i].fill_between(range(50), -10, 10,
                          where=zlb_periods,
                          alpha=0.2, color='red',
                          label='ZLB Binding')

axes[0].legend()
axes[2].set_xlabel('Quarters')
plt.tight_layout()
plt.show()
```

## Best Practices

### Model Design

1. **Start Simple**: Begin with a minimal version, test, then add features
2. **Modular Equations**: Group related equations (production, consumption, etc.)
3. **Clear Comments**: Document each equation with economic interpretation
4. **Symbolic Names**: Use parameter names from the literature

### State Vector Organization

Recommended ordering:
1. Forward-looking variables (outputs, prices, etc.)
2. Predetermined variables (capital, etc.)
3. Lags of endogenous variables
4. Shocks
5. Shock MA terms
6. Measurement errors

### Parameter Priors

| Parameter Type | Distribution | Example |
|----------------|--------------|---------|
| Persistence (0,1) | Beta | `Prior('beta', {'alpha': 18, 'beta': 2})` for ρ≈0.9 |
| Positive unbounded | Gamma | `Prior('gamma', {'shape': 2, 'rate': 4})` |
| Standard deviations | Inverse Gamma | `Prior('invgamma', {'shape': 2, 'scale': 0.02})` |
| Unbounded | Normal | `Prior('normal', {'mean': 1.5, 'std': 0.25})` |
| Bounded interval | Uniform | `Prior('uniform', {'lower': 0, 'upper': 1})` |

### Testing Checklist

Before using your model:

```python
def test_model(model):
    """Comprehensive model tests."""

    # 1. Dimension checks
    mats = model.system_matrices()
    assert mats['Gamma0'].shape == (model.spec.n_states, model.spec.n_states)
    assert mats['Psi'].shape == (model.spec.n_states, model.spec.n_shocks)

    # 2. Solution exists and is stable
    solution, info = solve_linear_model(
        mats['Gamma0'], mats['Gamma1'],
        mats['Psi'], mats['Pi'],
        n_states=model.spec.n_states
    )
    assert solution.is_stable, "Solution is unstable!"
    print(f"Max eigenvalue: {np.max(np.abs(solution.eigenvalues)):.4f}")

    # 3. Simulation doesn't explode
    solution.Z, solution.D = model.measurement_equation()
    solution.Q = model.shock_covariance()
    states, obs = simulate(solution, n_periods=200, random_seed=42)
    assert np.all(np.isfinite(obs)), "Simulation exploded!"
    assert np.max(np.abs(obs)) < 100, "Simulation values unreasonable!"

    # 4. IRFs are well-behaved
    for shock_idx in range(model.spec.n_shocks):
        irf = compute_irf(solution, shock_idx, periods=40)
        assert np.all(np.isfinite(irf)), f"IRF {shock_idx} has non-finite values!"
        assert np.max(np.abs(irf)) < 10, f"IRF {shock_idx} too large!"

    # 5. Prior densities are finite
    test_params = model.parameters.get_values()
    prior_density = model.parameters.log_prior_density()
    assert np.isfinite(prior_density), "Prior density not finite at initial values!"

    print("✓ All tests passed!")

# Run tests
test_model(SimpleNKModel())
```

## Common Issues and Solutions

### Issue 1: "Blanchard-Kahn conditions not satisfied"

**Cause**: Model is indeterminate (infinite solutions) or explosive (no stable solution)

**Debugging:**
```python
print(f"Eigenvalues: {solution.eigenvalues}")
print(f"Stable eigenvalues: {np.sum(np.abs(solution.eigenvalues) < 1)}")
print(f"Should equal n_states - n_controls = {model.spec.n_states - model.spec.n_controls}")
```

**Common fixes:**
- Check Taylor principle: φ_π > 1
- Verify expectational errors (Π matrix) are correct
- Ensure capital and other state variables are predetermined

### Issue 2: Simulation explodes

**Cause**: Unstable eigenvalue or numerical overflow

**Debugging:**
```python
print(f"Transition matrix T largest eigenvalue: {np.max(np.abs(np.linalg.eigvals(solution.T)))}")
print(f"Shock matrix R norm: {np.linalg.norm(solution.R)}")
```

**Fixes:**
- Check shock standard deviations aren't too large
- Verify T matrix eigenvalues are inside unit circle
- Inspect parameter values for unreasonable magnitudes

### Issue 3: Prior density is -inf

**Cause**: Parameter value outside prior support

**Debugging:**
```python
for param in model.parameters.values():
    if not param.fixed:
        density = param.prior.log_pdf(param.value)
        print(f"{param.name}: value={param.value:.4f}, log_pdf={density:.2f}")
```

**Fix:** Adjust initial parameter values or prior specifications

## Advanced Topics

### Custom Steady State

For models with non-zero steady state:

```python
def get_steady_state(self, params=None):
    """Compute deterministic steady state."""

    if params is not None:
        self.parameters.set_values(params)

    # Extract parameters
    alpha = self.parameters['alpha']
    delta = self.parameters['delta']
    # ...

    # Solve steady state system
    r_ss = 1/beta - 1 + delta
    k_over_l = (r_ss / alpha)**(1/(alpha - 1))
    # ...

    # Return as dict
    return {
        'k': k_ss,
        'l': l_ss,
        'y': y_ss,
        # ...
    }
```

### Measurement Errors

Add i.i.d. measurement errors:

```python
def measurement_equation(self, params=None):
    # ... (Z, D as before)

    # Measurement error covariance
    H = np.diag([
        self.parameters['me_y']**2,
        self.parameters['me_pi']**2,
        self.parameters['me_r']**2
    ])

    return Z, D, H  # Return H as well
```

## Next Steps

- **[Using Models](using-models.md)**: Learn how to estimate and forecast with your model
- **[NYFed Model](../models/nyfed.md)**: Study a complex real-world example
- **[API Reference](../api.md)**: Detailed documentation of all classes and functions
