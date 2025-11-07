# DSGE Model Conversion: Julia to Python

## Executive Summary

This project aims to convert the New York Federal Reserve DSGE model from Julia to Python, with support for occasionally binding constraints (OccBin). This document summarizes the key resources and findings from initial research.

## Project Objectives

1. Convert the NYFed DSGE.jl model from Julia to Python
2. Integrate with Ed Herbst's Python DSGE estimation framework
3. Implement OccBin (occasionally binding constraints) solver functionality
4. Create a consistent methodology for estimating DSGE models with occasionally binding constraints

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

## Critical Findings

### 1. OccBin Python Implementation Already Exists ✅

Contrary to the initial assumption ("I don't think the Ocbin solver for python has been written yet"), **pydsge already implements OccBin functionality** for Python. This significantly changes the project scope.

### 2. Two Viable Development Paths

**Path A: Use pydsge as Foundation** (Recommended)
- Leverage existing OccBin implementation
- Focus on converting NYFed model specification to pydsge format
- Extend pydsge if needed for NYFed-specific features
- Faster time to working prototype

**Path B: Extend Ed Herbst's dsge Package**
- Build OccBin solver from scratch
- Port MATLAB reference implementations
- More control but significantly more development work
- Higher risk due to minimal package maintenance

### 3. Reference Implementation Sources

For understanding OccBin methodology:
1. **MATLAB Reference**: Guerrieri/Iacoviello's original toolkit
2. **Academic Implementation**: Richter/Throckmorton's replication packages
3. **Python Reference**: pydsge source code (already working)

## Recommended Next Steps

1. **Explore pydsge Capabilities**
   - Review documentation and examples
   - Test OccBin functionality with toy models
   - Assess compatibility with NYFed model structure

2. **Analyze DSGE.jl Structure**
   - Identify core model equations and calibration
   - Map Julia-specific features to Python equivalents
   - Document estimation procedures

3. **Architecture Decision**
   - Determine if pydsge can be used as-is or requires extension
   - Evaluate whether to fork pydsge or use as dependency
   - Plan integration strategy

4. **Implementation Planning**
   - Create detailed specification mapping (Julia → Python)
   - Identify testing/validation strategy
   - Establish development milestones

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

**Last Updated**: 2025-11-07
**Status**: Initial Research Phase
