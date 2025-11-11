# DSGE-PY

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/rndubs/dsge-py/actions/workflows/ci.yml/badge.svg)](https://github.com/rndubs/dsge-py/actions/workflows/ci.yml)
[![Documentation](https://github.com/rndubs/dsge-py/actions/workflows/docs.yml/badge.svg)](https://github.com/rndubs/dsge-py/actions/workflows/docs.yml)

> **⚠️ PROOF OF CONCEPT DISCLAIMER**
>
> This project is a proof of concept that was entirely written by Claude (Anthropic's AI assistant). It is intended solely for research, demonstration, and educational purposes.
>
> **This software should NOT be used to make any policy decisions, economic forecasts, or real-world financial decisions.** While the implementation follows established economic modeling techniques, it has not undergone the rigorous validation and peer review required for use in actual economic policy or analysis.

A Python framework for Dynamic Stochastic General Equilibrium (DSGE) models with support for occasionally binding constraints.

## Features

- Linear DSGE solution (Blanchard-Kahn)
- Bayesian estimation (Sequential Monte Carlo)
- Occasionally binding constraints (OccBin)
- Kalman filtering and forecasting
- Production-ready models: NYFed Model 1002, Smets-Wouters (2007)

## Installation

```bash
git clone https://github.com/rndubs/dsge-py.git
cd dsge-py
uv sync  # or: pip install -e .
```

For FRED data access, get a free API key at https://fred.stlouisfed.org/ and set:
```bash
export FRED_API_KEY=your_key_here
```

## Quick Start

```python
from models.nyfed_model_1002 import create_nyfed_model
from dsge.solvers.linear import solve_linear_model

# Load and solve the NYFed model
model = create_nyfed_model()
solution, info = solve_linear_model(**model.system_matrices(), n_states=48)

print(f"Stable: {solution.is_stable}")
```

See `examples/` directory for complete examples including AR(1), New Keynesian, and Zero Lower Bound models.

## Documentation

Full documentation available in `docs/`:
- **Introduction**: Package philosophy and motivation
- **Features**: Complete feature overview
- **User Guides**: Using existing models and creating custom models
- **Model Documentation**: NYFed Model 1002 specification
- **API Reference**: Complete API documentation

Build docs: `cd docs && make html`

## Development

See [PLAN.md](PLAN.md) for development roadmap and [BACKGROUND.md](BACKGROUND.md) for project objectives.

**Status**: 134 tests passing, NYFed and Smets-Wouters models implemented and validated.

## License

MIT License
