"""
Example usage of the St. Louis Fed DSGE model.

This script demonstrates how to:
1. Create the model
2. Inspect its structure
3. Get system matrices
4. (Future) Solve and simulate

Note: Full solution requires completing the equilibrium system implementation.
See models/IMPLEMENTING_STLOUISFED.md for details.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from models.stlouisfed_dsge import create_stlouisfed_dsge


def main():
    """Demonstrate St. Louis Fed DSGE model usage."""
    print("=" * 80)
    print("ST. LOUIS FED DSGE MODEL - EXAMPLE USAGE")
    print("=" * 80)
    print()

    # ========================================================================
    # 1. Create the model
    # ========================================================================
    print("1. Creating St. Louis Fed DSGE model...")
    model = create_stlouisfed_dsge()
    print(f"   ✓ Model created successfully")
    print()

    # ========================================================================
    # 2. Inspect model structure
    # ========================================================================
    print("2. Model Structure:")
    print(f"   States:      {model.spec.n_states}")
    print(f"   Shocks:      {model.spec.n_shocks}")
    print(f"   Observables: {model.spec.n_observables}")
    print()

    print("   Key Features:")
    print("   • Two-agent heterogeneity (workers & capitalists)")
    print("   • Explicit fiscal sector (G, B, tax)")
    print("   • Sticky prices and wages")
    print("   • Capital accumulation with adjustment costs")
    print()

    # ========================================================================
    # 3. Display key parameters
    # ========================================================================
    print("3. Key Parameters:")
    print()

    print("   Heterogeneity:")
    print(f"   • Worker share (lambda_w):     {model.parameters['lambda_w']:.4f}")
    print(f"   • Portfolio adj. cost (psiH):  {model.parameters['psiH']:.4f}")
    print()

    print("   Preferences:")
    print(f"   • Discount factor (betta):     {model.parameters['betta']:.4f}")
    print(f"   • Risk aversion (sigma_c):     {model.parameters['sigma_c']:.4f}")
    print(f"   • Inverse Frisch (varrho):     {model.parameters['varrho']:.4f}")
    print()

    print("   Production:")
    print(f"   • Capital share (alp):         {model.parameters['alp']:.4f}")
    print(f"   • Depreciation (delta):        {model.parameters['delta']:.4f}")
    print(f"   • Inv. adj. cost (phiX):       {model.parameters['phiX']:.4f}")
    print()

    print("   Policy:")
    print(f"   • Interest rate smoothing:     {model.parameters['rho_r']:.4f}")
    print(f"   • Taylor coef. inflation:      {model.parameters['theta_pie']:.4f}")
    print(f"   • Fiscal response to debt:     {model.parameters['phi_tauT_B']:.4f}")
    print()

    print("   Steady State:")
    print(f"   • Govt spending/GDP (gy):      {model.parameters['gy']:.4f}")
    print(f"   • Debt/GDP (BYss):             {model.parameters['BYss']:.4f}")
    print(f"   • Labor share (LSss):          {model.parameters['LSss']:.4f}")
    print()

    # ========================================================================
    # 4. Get system matrices
    # ========================================================================
    print("4. System Matrices:")
    print("   Computing Gamma0, Gamma1, Psi, Pi...")

    try:
        matrices = model.system_matrices()
        print(f"   ✓ Matrices computed successfully")
        print()
        print(f"   Matrix dimensions:")
        print(f"   • Gamma0: {matrices['Gamma0'].shape}")
        print(f"   • Gamma1: {matrices['Gamma1'].shape}")
        print(f"   • Psi:    {matrices['Psi'].shape}")
        print(f"   • Pi:     {matrices['Pi'].shape}")
        print()

        # Check sparsity
        n_total = model.spec.n_states
        n_nonzero_g0 = np.count_nonzero(matrices['Gamma0'])
        n_nonzero_g1 = np.count_nonzero(matrices['Gamma1'])
        sparsity_g0 = 100 * (1 - n_nonzero_g0 / (n_total**2))
        sparsity_g1 = 100 * (1 - n_nonzero_g1 / (n_total**2))

        print(f"   Matrix sparsity:")
        print(f"   • Gamma0: {sparsity_g0:.1f}% sparse ({n_nonzero_g0} non-zero)")
        print(f"   • Gamma1: {sparsity_g1:.1f}% sparse ({n_nonzero_g1} non-zero)")
        print()

    except Exception as e:
        print(f"   ✗ Error computing matrices: {e}")
        print()

    # ========================================================================
    # 5. Display measurement equation
    # ========================================================================
    print("5. Measurement Equation:")
    print("   Computing Z matrix and D vector...")

    try:
        Z, D = model.measurement_equation()
        print(f"   ✓ Measurement equation computed")
        print()
        print(f"   Dimensions:")
        print(f"   • Z matrix: {Z.shape}  (maps states to observables)")
        print(f"   • D vector: {D.shape}  (constant terms)")
        print()

        # Check which observables are mapped
        n_mapped = np.sum(np.any(Z != 0, axis=1))
        print(f"   Observables with mappings: {n_mapped}/{model.spec.n_observables}")
        print()

    except Exception as e:
        print(f"   ✗ Error computing measurement equation: {e}")
        print()

    # ========================================================================
    # 6. Display observable variables
    # ========================================================================
    print("6. Observable Variables (FRED Series):")
    print()

    obs_descriptions = [
        ("dy", "GDP growth", "GDPC1"),
        ("dc", "Consumption growth", "PCECC96"),
        ("dinve", "Investment growth", "GPDIC1"),
        ("dg", "Govt spending growth", "GCEC1"),
        ("hours", "Hours worked", "HOANBS"),
        ("dw", "Wage growth", "COMPNFB"),
        ("infl", "Core PCE inflation", "PCEPILFE"),
        ("ffr", "Federal Funds Rate", "FEDFUNDS"),
        ("r10y", "10-Year Treasury", "GS10"),
        ("infl_exp", "Inflation expectations", "EXPINF10YR"),
        ("ls", "Labor share", "LABSHPUSA156NRUG"),
        ("debt_gdp", "Debt-to-GDP ratio", "GFDGDPA188S"),
        ("tax_gdp", "Tax-to-GDP ratio", "FGRECPT"),
    ]

    for i, (name, desc, fred) in enumerate(obs_descriptions, 1):
        print(f"   {i:2d}. {name:10s} - {desc:25s} ({fred})")
    print()

    # ========================================================================
    # 7. Next steps
    # ========================================================================
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("To make this model fully operational:")
    print()
    print("1. Complete equilibrium system implementation")
    print("   → See: models/IMPLEMENTING_STLOUISFED.md")
    print("   → Translate all ~40 equations from Dynare code")
    print("   → Handle 7 expectation errors properly")
    print()
    print("2. Solve the model")
    print("   → Use: solve_linear_model(matrices)")
    print("   → Check Blanchard-Kahn conditions")
    print("   → Verify stable dynamics")
    print()
    print("3. Prepare data")
    print("   → Download FRED series (requires API key)")
    print("   → Apply transformations")
    print("   → See: data/stlouisfed_fred_mapping.py")
    print()
    print("4. Estimate parameters")
    print("   → Use SMC estimation framework")
    print("   → Validate against posterior estimates")
    print()
    print("5. Generate forecasts")
    print("   → Unconditional and conditional forecasts")
    print("   → Incorporate uncertainty")
    print()
    print("=" * 80)
    print()

    print("References:")
    print("• Faria-e-Castro (2024): St. Louis Fed Working Paper 2024-014")
    print("• Cantore & Freund (2021): Journal of Monetary Economics, 119, 58-74")
    print("• Dynare code: github.com/ccantore/TANK-CW_Replication")
    print()


if __name__ == "__main__":
    main()
