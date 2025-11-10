# Introduction

## What is DSGE-PY?

DSGE-PY is a comprehensive Python framework for specifying, solving, filtering, and estimating Dynamic Stochastic General Equilibrium (DSGE) models. The package provides modern tools for macroeconomic modeling with a focus on **reusability**, **modularity**, and **ease of use**.

## Motivation

### The Challenge

DSGE models are essential tools in modern macroeconomics, used by central banks, research institutions, and academics for policy analysis, forecasting, and understanding economic dynamics. However, implementing these models involves significant technical challenges:

- **Complex solution methods** requiring numerical linear algebra
- **State-space filtering** for likelihood evaluation
- **Bayesian estimation** with computationally intensive algorithms
- **Occasionally binding constraints** (like the Zero Lower Bound) requiring specialized solvers

Existing tools either focus on specific models or require researchers to implement estimation infrastructure from scratch for each new model.

### The Solution: A Framework-Centric Approach

DSGE-PY adopts a **framework-centric** rather than **model-centric** approach. Instead of providing a single model implementation, DSGE-PY provides reusable infrastructure that works across multiple model specifications.

**Key Benefits:**

- **Separation of concerns**: Model specification is independent from solution and estimation methods
- **Consistent methodology**: The same estimation engine works across different models
- **Reduced duplication**: Implement your model once, leverage existing estimation infrastructure
- **Easy comparison**: Compare different models using identical estimation procedures
- **Future-proof**: Methodological improvements benefit all models simultaneously

## Project Philosophy

### Framework vs. Model Implementation

Think of DSGE-PY like a statistical software package (Stan, PyMC) rather than a single model port. Just as PyMC can estimate many different statistical models, DSGE-PY can estimate many different DSGE models using consistent infrastructure.

**This project is for you if you want to:**
- Estimate multiple DSGE models with consistent methodology
- Develop new DSGE models without reimplementing solution/estimation code
- Compare different model specifications fairly
- Build on a tested, documented codebase
- Focus on economics rather than numerical methods

**This project may not be for you if you:**
- Only need to run a specific existing model for production forecasting
- Require maximum computational performance (use Julia/C++)
- Need cutting-edge solution methods (we focus on proven, stable algorithms)

## Core Capabilities

### 1. Linear DSGE Models

Solve and estimate standard linear DSGE models using the Blanchard-Kahn method:

- Rational expectations solution via Schur decomposition
- Determinacy analysis and eigenvalue diagnostics
- Kalman filtering for state inference and likelihood evaluation
- Sequential Monte Carlo (SMC) for Bayesian parameter estimation

### 2. Occasionally Binding Constraints (OccBin)

Handle models with regime-switching constraints:

- Implementation of the Guerrieri-Iacoviello (2015) algorithm
- Automatic regime detection and switching
- Regime-aware filtering (both perfect foresight and particle filter approaches)
- Applications: Zero Lower Bound, credit constraints, capacity constraints

### 3. Real-World Applications

The framework includes implementations of production-ready models:

- **NYFed DSGE Model 1002**: Medium-scale model with financial frictions (67 parameters, 13 observables)
- **Smets-Wouters (2007)**: Canonical medium-scale DSGE model (41 parameters, 7 observables)
- Simple pedagogical models for learning and testing

## Design Principles

### Modularity

Clean separation between components:
- **Models** (`dsge.models`): Specify equilibrium conditions and parameters
- **Solvers** (`dsge.solvers`): Compute rational expectations solutions
- **Filters** (`dsge.filters`): Perform state-space filtering
- **Estimation** (`dsge.estimation`): Bayesian parameter estimation
- **Forecasting** (`dsge.forecasting`): Generate forecasts with uncertainty

### Extensibility

Easy to add new models by inheriting from the `DSGEModel` base class. The framework handles solution, filtering, and estimation automatically.

### Testing

Comprehensive test suite with 134 tests covering all core functionality. All examples are tested and documented.

### Documentation

Multiple documentation levels:
- This user guide for high-level concepts
- API reference for detailed function signatures
- Example scripts demonstrating complete workflows
- Academic references for methodological details

## Who Should Use DSGE-PY?

### Researchers

- Estimate existing DSGE models on your own data
- Develop new model specifications quickly
- Compare alternative model formulations
- Experiment with different estimation methods

### Students

- Learn DSGE modeling with working examples
- Understand solution and estimation algorithms through clean implementations
- Progress from simple (AR(1)) to complex (NYFed) models
- Well-documented code suitable for educational purposes

### Policy Analysts

- Run forecasts from estimated DSGE models
- Perform scenario analysis and policy experiments
- Generate uncertainty bands around forecasts
- Extend models with new features relevant to policy questions

## Next Steps

- **[Features Overview](features.md)**: Detailed look at framework capabilities
- **[User Guide - Using Models](user-guide/using-models.md)**: Start working with existing models
- **[User Guide - Creating Models](user-guide/creating-models.md)**: Build your own DSGE model
- **[API Reference](api.md)**: Complete function and class documentation

## References

The framework builds on established methods from the DSGE literature:

- **Blanchard & Kahn (1980)**: Solution of linear rational expectations models
- **Kalman (1960)**, **Rauch et al. (1965)**: State-space filtering and smoothing
- **Herbst & Schorfheide (2014)**: Sequential Monte Carlo for DSGE estimation
- **Guerrieri & Iacoviello (2015)**: Toolkit for occasionally binding constraints
- **Del Negro et al. (2015)**: NYFed DSGE model specification

See the complete bibliography in our [references](references.bib) file.
