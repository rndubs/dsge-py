# Completing the St. Louis Fed DSGE Model Implementation

This document provides a detailed guide for completing the full implementation of the St. Louis Fed DSGE model equilibrium system.

## Current Status

✅ **Complete:**
- Model structure (49 states, 7 shocks, 13 observables)
- All parameters defined with priors (40 parameters)
- FRED data mappings for all observables
- Measurement equations
- Test suite (20 tests passing)
- Documentation

⚠️ **Partial:**
- `system_matrices()` method: Simplified version with key equations
- Full equilibrium system needs all ~40 equations from Dynare code

## Implementation Guide

### Understanding the Canonical Form

The Sims (2002) solution method uses:

```
Γ0 * s_t = Γ1 * s_{t-1} + Ψ * ε_t + Π * η_t
```

Where:
- `s_t` = state vector at time t
- `ε_t` = structural shocks
- `η_t` = expectation errors for forward-looking variables

### Forward-Looking Variables (Expectation Errors Needed)

The TANK model has 7 forward-looking variables:

1. `UCS(+1)` - Capitalist marginal utility (in Euler equation)
2. `UCH(+1)` - Worker marginal utility (in Euler equation)
3. `PIE(+1)` - Inflation (in Fisher equation, Phillips curve, arbitrage)
4. `PIEW(+1)` - Wage inflation (in wage Phillips curve)
5. `Q(+1)` - Tobin's Q (in arbitrage condition)
6. `RK(+1)` - Rental rate (in arbitrage condition)
7. `IS(+1)` - Investment (in investment equation)

**Expectation Error Representation:**

For a forward-looking equation like:
```
y_t = E_t[y_{t+1}] + x_t
```

Rewrite as:
```
y_t = y_{t+1} - η_y_{t+1} + x_t
```

Where `η_y` is the expectation error. In the canonical form:
```
Gamma0[eq, idx['y']] = 1.0
Gamma0[eq, idx['x']] = -1.0
Pi[eq, eta_idx['y']] = 1.0  # Captures -E_t[y_{t+1}] = y_{t+1} - η
```

### Complete Equation List from Dynare Code

Location: `/tmp/TANK-CW_Replication/TANK models/Dynare master codes/medium scale model/tank_cw_ms.mod`

Lines 333-601 contain the linearized model block. Configuration for TANK-CW:
- `model_version = 1` (TANK-CW)
- `U_ls = 0` (capitalists don't work)
- `PAC = 1` (workers have portfolio adjustment costs)
- `gov_spending = 1` (with government)
- `debt_gdp_steadystate = 1` (with debt)
- `free_entry = 0` (profits > 0)

#### Household Block (9 equations)

1. **Marginal Utility, Capitalists** (Line 335):
   ```
   UCS = Pr - sigma_c*CS
   ```

2. **Euler, Capitalists** (Line 337):
   ```
   UCS = R + UCS(+1)
   ```
   - **Expectation error**: UCS(+1)

3. **MU Leisure, Capitalists** (Line 341):
   ```
   UHS = 0
   ```

4. **Labor Supply, Capitalists** (Line 343):
   ```
   HS = 0
   ```

5. **Marginal Utility, Workers** (Line 354):
   ```
   UCH = Pr - sigma_c*CH
   ```

6. **MU Leisure, Workers** (Line 357):
   ```
   UHH = varrho*HH
   ```

7. **Labor Supply, Workers** (Line 359):
   ```
   MRS = UHH - UCH
   ```

8. **Euler, Workers** (Line 369):
   ```
   UCH = R + UCH(+1) - (psiH/CHss)*BH
   ```
   - **Expectation error**: UCH(+1)

9. **Budget Constraint, Workers** (Line 382):
   ```
   CH + BH/CHss = (W+HH)*Wss*HHss/CHss - taxHss/CHss*taxH
                  + tauD/lambda*profits/CHss + BH(-1)*Rss/CHss
   ```

#### Firm Block (14 equations)

10. **Production Function** (Line 453):
    ```
    YW = (1-alp)*(Z+H) + alp*(U+K(-1))
    ```

11. **Real Output** (Line 455):
    ```
    Y = YW*(1+FY)
    ```
    - Note: In linearized form, `Y ≈ YW` if FY is small

12. **Marginal Product of Labor** (Line 457):
    ```
    MPL = YW - H
    ```

13. **Real Wage** (Line 459):
    ```
    W = MC + MPL
    ```

14. **Marginal Product of Capital** (Line 461):
    ```
    MPK = YW - U - K(-1)
    ```

15. **Rental Rate** (Line 463):
    ```
    RK = MC + MPK
    ```

16. **Profits** (Line 469):
    ```
    Y = (W+H)*Wss*Hss/Yss + (RK+U+K(-1))*RKss*Kss/Yss + profits/Yss
    ```

17. **Profits Distribution** (Line 474):
    ```
    profits = profitsS
    ```

18. **Labor Income** (Line 476):
    ```
    LI = W + H
    ```

19. **Labor Share** (Line 478):
    ```
    LS = W + H - Y
    ```

20. **Capital Share** (Line 480):
    ```
    KSh = RK + U + K(-1) - Y
    ```

21. **Variable Utilization** (Line 483):
    ```
    RK = gamma2/gamma1*U
    ```

22. **Capital Accumulation** (Line 485):
    ```
    KS = delta*(IS+ZI) + (1-delta)*KS(-1)
    ```

23. **Arbitrage Condition** (Line 487):
    ```
    Rn - PIE(+1) = betta*(1-delta)*Q(+1) + (1-betta*(1-delta))*RK(+1) - Q
    ```
    - **Expectation errors**: PIE(+1), Q(+1), RK(+1)

#### Investment Block (3 equations)

24. **Investment Equation** (Line 489):
    ```
    (1+1/Rss)*IS = 1/Rss*IS(+1) + IS(-1) + 1/(2*phiX)*(Q+ZI)
    ```
    - **Expectation error**: IS(+1)

25. **Investment Aggregation** (Line 492):
    ```
    I = IS
    ```

26. **Capital Aggregation** (Line 494):
    ```
    K = KS
    ```

#### Resource Constraint & Aggregation (3 equations)

27. **Resource Constraint** (Line 499):
    ```
    Y = C*cy + G*gy + I*iy + gamma1*Kss/Yss*U
    ```

28. **Fisher Equation** (Line 505):
    ```
    R = Rn - PIE(+1)
    ```
    - **Expectation error**: PIE(+1)

29. **Consumption Aggregation** (Line 507):
    ```
    C = lambda*CH + (1-lambda)*CS
    ```

30. **Labor Aggregation** (Line 511):
    ```
    H = HH
    ```
    - (Only workers supply labor when U_ls=0)

#### Price/Wage Dynamics (3 equations)

31. **Phillips Curve** (Line 519):
    ```
    PIE = betta*PIE(+1) + kappa*(MC+MS)
    ```
    - **Expectation error**: PIE(+1)

32. **Wage Phillips Curve** (Line 522):
    ```
    PIEW = betta*PIEW(+1) + kappaw*(MRS-W+WMS)
    ```
    - **Expectation error**: PIEW(+1)

33. **Wage Inflation Definition** (Line 526):
    ```
    PIEW = W - W(-1)
    ```

#### Policy Rules (5 equations)

34. **Taylor Rule** (Line 529):
    ```
    Rn = rho_r*Rn(-1) + (1-rho_r)*(theta_pie*PIE + theta_y*Y) + epsM
    ```

35. **Government Budget** (Line 540):
    ```
    B = (B(-1)+R(-1))*Rss + G*Gss/Bss - tax*taxss/Bss
    ```

36. **Bond Aggregation** (Line 553):
    ```
    B = BS + lambda*BH/Bss
    ```

37. **Tax Rule** (Line 565):
    ```
    tax = rho_tauT*tax(-1) + phi_tauT_B*B(-1) + phi_tauT_G*G
    ```

38. **Tax Distribution, Capitalists** (Line 592):
    ```
    taxS = (1-eta)/(1-lambda)*tax
    ```

39. **Tax Distribution, Workers** (Line 594):
    ```
    taxH = eta/lambda*tax
    ```

#### Shock Processes (6 equations)

40. **Technology** (Line 532):
    ```
    Z = rhoZ*Z(-1) + epsZ
    ```

41. **Government Spending** (Line 535):
    ```
    G = rhoG*G(-1) + epsG
    ```

42. **Price Markup** (Line 570):
    ```
    MS = rhoMS*MS(-1) + epsMS
    ```

43. **Wage Markup** (Line 572):
    ```
    WMS = rhoWMS*WMS(-1) + epsWMS
    ```

44. **Preference** (Line 575):
    ```
    Pr = rhoPr*Pr(-1) + epsPr
    ```

45. **MEI** (Line 577):
    ```
    ZI = rhoZI*ZI(-1) + epsZI
    ```

### Implementation Steps

1. **Create expectation error indices** (7 errors):
   ```python
   n_eta = 7
   eta_idx = {
       'UCS': 0, 'UCH': 1, 'PIE': 2, 'PIEW': 3,
       'Q': 4, 'RK': 5, 'IS': 6
   }
   Pi = np.zeros((n_total, n_eta))
   ```

2. **Implement each equation systematically**:
   - Put current period (t) variables in `Gamma0`
   - Put lagged (-1) variables in `Gamma1`
   - Put expectation errors in `Pi`
   - Put shock innovations in `Psi`

3. **Example: Capitalist Euler Equation**
   ```python
   # UCS = R + UCS(+1)
   # Rearranged: UCS - R - UCS(+1) = 0
   # With expectation error: UCS - R - (UCS_{t+1} - eta_UCS) = 0

   row = 1  # Equation number
   Gamma0[row, idx['UCS']] = 1.0
   Gamma0[row, idx['R']] = -1.0
   Pi[row, eta_idx['UCS']] = 1.0  # -(-eta) = +eta
   ```

4. **Handle lags properly**:
   ```python
   # For BH(-1) in worker budget:
   Gamma1[row, idx['BH']] = -Rss/CHss

   # For K(-1) in production:
   Gamma1[row, idx['K']] = -alp_calc
   ```

5. **Add lag definitions**:
   ```python
   # After all behavioral equations, define lags:
   # C_lag = C(-1) => C_lag_t = C_{t-1}
   Gamma0[row, idx['C_lag']] = 1.0
   Gamma1[row, idx['C']] = -1.0
   ```

### Testing Strategy

1. **Start with a subset**: Implement core equations first (Euler, production, Taylor rule)
2. **Test solution**: Use `solve_linear_model()` to check for:
   - Blanchard-Kahn conditions satisfied
   - Stable eigenvalues
   - No NaN/Inf in solution
3. **Add equations incrementally**: One block at a time
4. **Validate IRFs**: Compare to Dynare output

### Validation Against Dynare

After implementation, validate by:

1. **Running Dynare code**:
   ```matlab
   cd '/tmp/TANK-CW_Replication/TANK models/Dynare master codes/medium scale model'
   dynare tank_cw_ms
   ```

2. **Compare IRFs**: Extract impulse responses and compare

3. **Compare steady state**: Check that all SS relationships match

### Common Pitfalls

1. **Sign conventions**: Be careful with minus signs in linearized equations
2. **Steady state values**: Ensure all ratios (Wss*Hss/CHss, etc.) are computed correctly
3. **Expectation errors**: Forward-looking variables need Pi matrix entries
4. **Lag handling**: Use Gamma1 for variables at t-1
5. **Index management**: Double-check all state indices

### Resources

- **Dynare code**: `/tmp/TANK-CW_Replication/TANK models/Dynare master codes/medium scale model/tank_cw_ms.mod`
- **Example**: `/home/user/dsge-py/models/simple_nk_model.py` (shows expectation handling)
- **Sims (2002)**: "Solving Linear Rational Expectations Models"

### Estimated Time

- **Core equations** (15-20): 2-3 hours
- **Full system** (40+ equations): 4-6 hours
- **Testing & validation**: 2-3 hours
- **Total**: 8-12 hours for complete implementation

## Quick Start: Minimal Working Example

To get a minimal working version quickly:

1. Implement equations 1-10 (household + basic production)
2. Implement equations 27-30 (aggregation)
3. Implement equations 31-34 (pricing + policy)
4. Implement equations 40-45 (shocks)
5. Test with simple shock (e.g., technology)

This gives ~25 equations that form a closed system and can solve.

---

**Last Updated**: 2025-11-11
**For Questions**: Refer to Dynare code and Simple NK model example
