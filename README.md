# Python DSGE Estimation Framework with OccBin Support

## Executive Summary

This project aims to develop a consistent estimation framework for Dynamic Stochastic General Equilibrium (DSGE) models in Python, with particular emphasis on models featuring occasionally binding constraints (OccBin). The New York Federal Reserve DSGE model serves as the primary application and validation case for this estimation infrastructure. This document summarizes the key resources and findings from initial research.

## Project Objectives

### Primary Goal
Develop a consistent, reusable estimation strategy that can be applied across different DSGE model specifications, enabling researchers to:
- Define models using a standardized specification format
- Estimate models using a unified Bayesian estimation engine
- Incorporate occasionally binding constraints (OccBin) seamlessly
- Maintain separation between model definition and estimation methodology

### Specific Deliverables
1. **Estimation Framework**: Implement a modular estimation infrastructure based on Sequential Monte Carlo (SMC) or equivalent Bayesian methods
2. **Model Specification Layer**: Create a consistent interface for defining DSGE model equations, parameters, and calibrations
3. **OccBin Integration**: Integrate occasionally binding constraints solver functionality into the estimation workflow
4. **NYFed Model Application**: Convert and estimate the New York Federal Reserve DSGE model as a demonstration of the framework's capabilities
5. **Documentation**: Provide clear examples of how to specify and estimate different DSGE models using the framework

## Key Resources Identified

### 1. NYFed DSGE Model (Source: Julia)

**Repository**: [FRBNY-DSGE/DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl)
**Documentation**: https://frbny-dsge.github.io/DSGE.jl/latest/

**Key Features**:
- Solves and estimates Dynamic Stochastic General Equilibrium models
- Includes the New York Fed DSGE Model (Version 1002)
- Uses Sequential Monte Carlo (SMC) for Bayesian estimation
- Actively maintained by FRBNY research team
- Written in Julia v1.x

**Related Ecosystem**:
- `StateSpaceRoutines.jl` - Kalman filtering and smoothing algorithms
- `SMC.jl` - Sequential Monte Carlo sampling for estimation
- `ModelConstructors.jl` - Abstract model type definitions
- `FredData.jl` - Economic data retrieval

### 2. Python DSGE Packages

#### Option A: Ed Herbst's dsge (Minimal)

**Repository**: [eph/dsge](https://github.com/eph/dsge)
**Installation**: `conda install dsge -c eherbst`

**Characteristics**:
- Simple Python 3+ package for DSGE models
- Originally forked from Pablo Winant's dolo package
- Written primarily for personal research use
- Minimally maintained, may contain bugs
- Companion `smc` package available for Bayesian estimation
- Author: Ed Herbst (Federal Reserve Board)

**Status**: ⚠️ Personal research tool, not production-ready

#### Option B: pydsge (Recommended)

**Repository**: [gboehl/pydsge](https://github.com/gboehl/pydsge)
**Documentation**: https://pydsge.readthedocs.io/

**Key Features**:
- ✅ **Already supports occasionally binding constraints (OccBin)**
- Solves, filters, and estimates linear DSGE models
- Handles Zero Lower Bound (ZLB) and other regime-switching constraints
- Based on published research: Boehl & Strobel (2023, *Journal of Economic Dynamics and Control*)
- Parser originally forked from Ed Herbst's work
- Actively maintained

**Status**: ✅ Production-ready with OccBin support

### 3. OccBin Toolkit (Reference Implementation)

**Original Repository**: [lucaguerrieri/occbin](https://github.com/lucaguerrieri/occbin)
**Platform**: MATLAB/Dynare
**Paper**: Guerrieri & Iacoviello (2015), "OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily", *Journal of Monetary Economics*

**Methodology**:
- Piecewise linear perturbation approach
- Handles regime-switching models where constraints occasionally bind
- Two-regime framework: constraint slack vs. constraint binding
- Compatible with Dynare modeling environment

**Applications**:
- Zero lower bound on nominal interest rates
- Occasionally binding credit constraints
- Factor mobility limitations
- Other regime-dependent constraints

### 4. Alexander Richter & Nathaniel Throckmorton's Work

**Alexander W. Richter**:
- Vice President, Federal Reserve Bank of Dallas
- Website: alexrichterecon.com
- GitHub: Multiple accounts found (economist vs. other developers)

**Nathaniel A. Throckmorton**:
- Assistant Professor, William & Mary
- Website: nathrock.github.io

**Relevant Publications**:
1. "The Zero Lower Bound and Estimation Accuracy" - Compares nonlinear vs. piecewise linear methods
2. "The Zero Lower Bound and Endogenous Uncertainty" (*Economic Journal*, 2018)
3. "Accuracy, Speed and Robustness of Policy Function Iteration"

**Replication Code**:
- Available on OpenICPSR: "Data and Code for: A Simple Explanation of Countercyclical Uncertainty"
- MATLAB implementations using time iteration with linear interpolation
- User-friendly suite of functions for OccBin models

## Project Philosophy: Framework-Centric vs. Model-Centric Approach

This project adopts a **framework-centric** approach rather than a **model-centric** approach. Understanding this distinction is crucial for architectural decisions.

### Framework-Centric Approach (This Project)
**Goal**: Build reusable estimation infrastructure applicable to multiple DSGE models

**Characteristics**:
- Separates model specification from estimation methodology
- Estimation engine is model-agnostic
- Researchers can plug in different model specifications
- Consistent estimation strategy across models
- NYFed model serves as validation case

**Analogy**: Building a statistical software package (like Stan or PyMC) that can estimate many different models, rather than implementing one specific model.

**Benefits**:
- Enables comparative analysis across models
- Reduces duplication of estimation code
- Facilitates methodological improvements (update estimation engine once, benefit all models)
- Supports reproducibility through consistent methodology

### Model-Centric Approach (Not This Project)
**Goal**: Port a specific model (NYFed DSGE) from Julia to Python

**Characteristics**:
- Focus on getting one model working in Python
- Estimation code may be tightly coupled to model specification
- Optimization for single use case
- Faster initial development

**When Appropriate**:
- Production deployment of specific model for forecasting
- Replicating published results for one model
- Quick prototyping or proof-of-concept

### Implications for Development

The framework-centric approach requires:
1. **More upfront design work**: Define clear interfaces between modules
2. **Greater initial development effort**: Build generalized rather than specialized code
3. **Broader validation requirements**: Test with multiple models, not just one
4. **Better long-term value**: Infrastructure can support future research needs

## Critical Findings

### 1. OccBin Python Implementation Already Exists ✅

Contrary to the initial assumption ("I don't think the Ocbin solver for python has been written yet"), **pydsge already implements OccBin functionality** for Python. This significantly changes the project scope.

### 2. Two Viable Development Paths

The choice of development path has significant implications for achieving the goal of a consistent estimation framework across multiple DSGE models.

**Path A: Build on pydsge**
- Leverage existing OccBin implementation (already functional)
- Faster time to working prototype for NYFed model
- Trade-off: Less control over estimation methodology
- Challenge: Framework is tightly integrated; harder to separate model specification from estimation
- Best suited for: Single-model estimation projects

**Path B: Extend Ed Herbst's dsge Package** (Aligned with Framework Goals)
- Modular architecture: `dsge` package for model specification, `smc` package for estimation
- Clear separation of concerns enables consistent estimation across different models
- Requires implementing OccBin solver (significant development effort)
- Can leverage MATLAB reference implementations (Guerrieri/Iacoviello, Richter/Throckmorton)
- Higher development cost but greater flexibility for multi-model framework
- Best suited for: Building reusable estimation infrastructure

**Path C: Hybrid Approach**
- Use pydsge's OccBin implementation as reference
- Integrate with Ed Herbst's modular framework (dsge + smc)
- Extract and adapt OccBin solver to work with dsge package specifications
- Balances development effort with framework goals

### 3. Reference Implementation Sources

For understanding OccBin methodology:
1. **MATLAB Reference**: Guerrieri/Iacoviello's original toolkit
2. **Academic Implementation**: Richter/Throckmorton's replication packages
3. **Python Reference**: pydsge source code (already working)

## Recommended Next Steps

### Phase 1: Framework Architecture Decision

1. **Evaluate Ed Herbst's dsge + smc Framework**
   - Review `dsge` package architecture and model specification format
   - Analyze `smc` package estimation capabilities
   - Assess extensibility for OccBin integration
   - Document current limitations and gaps

2. **Analyze pydsge OccBin Implementation**
   - Study OccBin solver algorithms and implementation details
   - Identify key components that could be extracted/adapted
   - Evaluate compatibility with modular estimation framework

3. **Architecture Decision**
   - Select development path (A, B, or C) based on framework requirements
   - Define module boundaries: model specification, solver, estimation
   - Plan integration strategy for OccBin functionality

### Phase 2: Estimation Framework Development

4. **Design Model Specification Interface**
   - Define standard format for DSGE model equations
   - Establish parameter and calibration structures
   - Create validation and testing protocols
   - Design for extensibility (multiple model types)

5. **Implement/Extend Estimation Engine**
   - Integrate or enhance SMC-based Bayesian estimation
   - Ensure estimation code is model-agnostic
   - Implement convergence diagnostics
   - Add support for parallel computation

6. **Integrate OccBin Solver**
   - Implement or adapt piecewise linear perturbation solver
   - Integrate regime-switching logic into estimation workflow
   - Validate against MATLAB reference implementations
   - Optimize for computational performance

### Phase 3: NYFed Model Application

7. **Convert NYFed Model Specification**
   - Analyze DSGE.jl model structure (equations, parameters, observables)
   - Map Julia-specific features to Python equivalents
   - Implement model in framework's specification format
   - Validate model solution against Julia implementation

8. **Estimate NYFed Model**
   - Prepare data inputs (FRED or equivalent sources)
   - Configure estimation parameters (priors, SMC settings)
   - Execute estimation using the framework
   - Compare results with published DSGE.jl estimates

9. **Validation and Testing**
   - Establish testing strategy (unit tests, integration tests)
   - Validate estimation results against known benchmarks
   - Document performance metrics and computational requirements
   - Create example notebooks demonstrating framework usage

### Phase 4: Documentation and Generalization

10. **Framework Documentation**
    - Write user guide for model specification
    - Document estimation methodology and configuration
    - Provide examples with multiple model types (not just NYFed)
    - Create developer documentation for extending framework

11. **Demonstrate Generalizability**
    - Implement at least one additional model (e.g., Smets-Wouters)
    - Show consistent estimation methodology across models
    - Document differences in model-specific requirements
    - Validate framework's reusability claims

## Technical Considerations

### Language Migration Challenges
- Julia uses multiple dispatch; Python uses object-oriented patterns
- Julia's type system vs. Python's duck typing
- Performance implications (Julia is JIT-compiled)
- Dependency management differences

### Estimation Framework
- SMC.jl (Julia) vs. SMC capabilities in Python
- Particle filtering implementations
- Bayesian estimation infrastructure

### OccBin Implementation Details
- Regime detection algorithms
- Convergence criteria for piecewise solutions
- Numerical stability considerations
- Computational performance optimization

## References

### Repositories
- FRBNY-DSGE/DSGE.jl: https://github.com/FRBNY-DSGE/DSGE.jl
- eph/dsge: https://github.com/eph/dsge
- gboehl/pydsge: https://github.com/gboehl/pydsge
- lucaguerrieri/occbin: https://github.com/lucaguerrieri/occbin

### Key Papers
- Guerrieri, L. & Iacoviello, M. (2015). "OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding Constraints Easily", *Journal of Monetary Economics*
- Boehl, G. & Strobel, F. (2023). "Estimation of DSGE models with occasionally binding constraints", *Journal of Economic Dynamics and Control*
- Richter, A.W. & Throckmorton, N.A. (2018). "The Zero Lower Bound and Endogenous Uncertainty", *Economic Journal*

### Documentation
- DSGE.jl Docs: https://frbny-dsge.github.io/DSGE.jl/latest/
- pydsge Docs: https://pydsge.readthedocs.io/
- Ed Herbst's Software: https://edherbst.net/research/software/

---

**Last Updated**: 2025-11-09
**Status**: Requirements Definition Phase
**Approach**: Framework-Centric (Consistent Estimation Methodology)
