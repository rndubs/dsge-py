# Model Implementation Notes

This document contains important context about model implementations in dsge-py.

## Parameter Values: Priors vs Posteriors

### General Principle

The models in this package use **posterior estimates** from Bayesian estimation as default parameter values, not prior means. This choice is deliberate:

1. **Smets-Wouters (2007)**: Uses posterior estimates from Table 1A and 1B of the original paper
2. **NYFed Model 1002**: Uses **prior means** (appropriate for an estimation framework)
3. **Simple NK Model**: Uses standard textbook calibrations

### Rationale

For **reference models** (Smets-Wouters), using posterior estimates ensures:
- Impulse response functions match published results
- Model dynamics reflect empirical fit
- Serves as proper validation benchmark

For **estimation frameworks** (NYFed Model 1002), using prior means allows:
- Users to run Bayesian estimation and obtain their own posteriors
- Flexibility for different data vintages
- Appropriate starting point for SMC/MCMC algorithms

## Smets-Wouters (2007) Implementation

### Key Parameter Updates (2025-11-11)

The following parameters were corrected to match published posterior estimates:

**Shock Persistence** (CRITICAL fixes):
- `crhoms`: 0.0 → **0.3000** (monetary policy shock)
- `crhopinf`: 0.0 → **0.8692** (price markup shock)
- `crhow`: 0.0 → **0.9546** (wage markup shock)

**Price/Wage Rigidity**:
- `cprobp`: 0.6 → **0.7813** (Calvo price stickiness)
- `cprobw`: 0.8087 → **0.7937** (Calvo wage stickiness)
- `cindp`: 0.47 → **0.3291** (price indexation)
- `cindw`: 0.3243 → **0.4425** (wage indexation)

**Preferences**:
- `csigma`: 1.5 → **1.2312** (risk aversion)
- `chabb`: 0.6361 → **0.7205** (habit formation)
- `csigl`: 1.9423 → **2.8401** (inverse Frisch elasticity)

**Policy Rule**:
- `crpi`: 1.488 → **1.7985** (inflation response)
- `crr`: 0.8762 → **0.8258** (interest rate smoothing)
- `cry`: 0.0593 → **0.0893** (output gap response)
- `crdy`: 0.2347 → **0.2239** (output growth response)

**Other Parameters**: Multiple shock persistence and standard deviation parameters updated to match Table 1B.

### Verification Sources

Parameter values verified against:
1. Smets & Wouters (2007) AER paper, Tables 1A and 1B
2. Johannes Pfeifer's Dynare replication (GitHub: JohannesPfeifer/DSGE_mod)

See model docstring in `models/smets_wouters_2007.py` for complete references.

## NYFed Model 1002 Implementation

### Parameter Source

All 67 parameters use **prior means** from FRBNY's DSGE.jl specification. This is verified against:
- DSGE.jl source code: `src/models/representative/m1002/m1002.jl`
- Official documentation: DSGE_Model_Documentation_1002.pdf

### Key Differences from DSGE.jl "Defaults"

DSGE.jl's default values are **posterior estimates** from FRBNY data. Notable differences between priors and posteriors include:

- **Calvo prices** (ζ_p): 0.50 (prior) vs 0.8940 (posterior) - much higher price stickiness from data
- **Calvo wages** (ζ_w): 0.50 (prior) vs 0.9291 (posterior) - very sticky wages empirically
- **Risk aversion** (σ_c): 1.50 (prior) vs 0.8719 (posterior) - lower than prior belief
- **Habit** (h): 0.70 (prior) vs 0.5347 (posterior) - less habit than prior
- **Risk premium persistence** (ρ_b): 0.50 (prior) vs 0.9410 (posterior) - much more persistent

This is **expected**—Bayesian estimation updates priors based on data.

### Usage Notes

To replicate FRBNY forecasts:
1. Estimate the model on FRED data using the SMC framework
2. Use specific data vintages matching FRBNY's publication dates
3. Match subspecification if needed (baseline vs alternatives)

See model docstring in `models/nyfed_model_1002.py` for complete references to DSGE.jl source files.

## Testing After Parameter Changes

Always run the full test suite after modifying parameter values:

```bash
uv run pytest tests/test_smets_wouters_model.py -v
uv run pytest tests/test_nyfed_model.py -v
```

All tests should pass. If IRF signs change or model becomes unstable, verify:
1. Parameter value matches source (check for typos)
2. Prior distribution specification is correct
3. Model equations haven't changed

## References

### Smets-Wouters (2007)
- Paper: https://www.aeaweb.org/articles?id=10.1257/aer.97.3.586
- Dynare code: https://github.com/JohannesPfeifer/DSGE_mod/tree/master/Smets_Wouters_2007

### NYFed Model 1002
- DSGE.jl: https://github.com/FRBNY-DSGE/DSGE.jl/tree/main/src/models/representative/m1002
- Documentation: https://github.com/FRBNY-DSGE/DSGE.jl/blob/main/docs/DSGE_Model_Documentation_1002.pdf

---

**Last Updated**: 2025-11-11
**Status**: Smets-Wouters parameters corrected to match posterior estimates
