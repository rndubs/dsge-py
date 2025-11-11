# DSGE-PY Documentation

Welcome to the DSGE-PY documentation! This package provides a comprehensive Python framework for Dynamic Stochastic General Equilibrium (DSGE) modeling with support for occasionally binding constraints.

```{toctree}
:maxdepth: 2
:caption: Getting Started

introduction
features
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user-guide/using-models
user-guide/creating-models
```

```{toctree}
:maxdepth: 2
:caption: Model Documentation

models/nyfed
```

```{toctree}
:maxdepth: 2
:caption: Developer Documentation

implementation-notes
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api
```

## Quick Links

### For New Users

- **[Introduction](introduction.md)**: Learn about the package philosophy and capabilities
- **[Features Overview](features.md)**: Survey of all features with API links
- **[Using Existing Models](user-guide/using-models.md)**: Start using models right away

### For Model Developers

- **[Creating Custom Models](user-guide/creating-models.md)**: Step-by-step guide to building your own DSGE model
- **[API Reference](api.md)**: Complete API documentation

### Model Specifications

- **[NYFed DSGE Model 1002](models/nyfed.md)**: Medium-scale model with financial frictions

## What is DSGE-PY?

DSGE-PY is a **framework-centric** toolkit for DSGE modeling in Python. Unlike single-model implementations, DSGE-PY provides reusable infrastructure that works across multiple model specifications.

### Core Capabilities

- **Linear DSGE Solution**: Blanchard-Kahn solver with Schur decomposition
- **Kalman Filtering**: State-space filtering and smoothing with missing data handling
- **Bayesian Estimation**: Sequential Monte Carlo (SMC) for parameter estimation
- **OccBin Support**: Occasionally binding constraints (Zero Lower Bound, credit constraints, etc.)
- **Forecasting**: Unconditional, conditional, and posterior-based forecasts

### Implemented Models

- **Simple New Keynesian**: 3-equation pedagogical model
- **NYFed Model 1002**: 67 parameters, 13 observables, financial frictions
- **Smets-Wouters (2007)**: 41 parameters, 7 observables, canonical medium-scale model

## Installation

### Using uv (recommended)

```bash
git clone https://github.com/rndubs/dsge-py.git
cd dsge-py
uv sync
```

### Using pip

```bash
git clone https://github.com/rndubs/dsge-py.git
cd dsge-py
pip install -e .
```

## Quick Start

```python
import numpy as np
from dsge import DSGEModel, ModelSpecification, Parameter, Prior
from dsge import solve_linear_model, estimate_dsge

# Define a simple AR(1) model
class AR1Model(DSGEModel):
    def __init__(self):
        spec = ModelSpecification(
            n_states=1, n_controls=0, n_shocks=1, n_observables=1,
            state_names=['x'], shock_names=['eps'], observable_names=['y']
        )
        super().__init__(spec)

    def _setup_parameters(self):
        self.parameters.add(Parameter(
            'rho', 0.9, fixed=False,
            prior=Prior('beta', {'alpha': 18, 'beta': 2})
        ))
        self.parameters.add(Parameter(
            'sigma', 0.1, fixed=False,
            prior=Prior('invgamma', {'shape': 2.0, 'scale': 0.2})
        ))

    def system_matrices(self, params=None):
        if params is not None:
            self.parameters.set_values(params)
        return {
            'Gamma0': np.array([[1.0]]),
            'Gamma1': np.array([[self.parameters['rho']]]),
            'Psi': np.array([[1.0]]),
            'Pi': np.array([[1e-10]])
        }

    def measurement_equation(self, params=None):
        return np.array([[1.0]]), np.array([0.0])

    def shock_covariance(self, params=None):
        if params is not None:
            self.parameters.set_values(params)
        return np.array([[self.parameters['sigma']**2]])

# Solve model
model = AR1Model()
solution, info = solve_linear_model(
    **model.system_matrices(), n_states=1
)

# Estimate from data
results = estimate_dsge(model, data, n_particles=500)
```

See the [User Guide](user-guide/using-models.md) for more examples.

## Project Status

**Current Version**: 0.2.0

- ✅ Core framework complete (solution, filtering, estimation)
- ✅ Full OccBin support (solver + filtering)
- ✅ NYFed Model 1002 implemented and validated
- ✅ Smets-Wouters (2007) implemented
- ✅ 134 comprehensive tests (all passing)
- 📝 Documentation in progress

## Contributing

Contributions are welcome! Areas for contribution:
- Additional model examples
- Performance optimization
- Documentation improvements
- Additional solution methods

## License

MIT License - see LICENSE file for details.

## Indices and Tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
