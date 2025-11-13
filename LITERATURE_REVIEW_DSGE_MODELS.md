# Comprehensive DSGE Models Literature Review

**Date:** January 2025
**Purpose:** Identify additional DSGE models for implementation in the dsge-py repository

---

## Executive Summary

This literature review identifies **30+ DSGE models** from the academic and central banking literature that could be implemented in the dsge-py framework. Models are organized by category:

1. **Central Bank DSGE Models** (12 models)
2. **Canonical Academic Models** (8 models)
3. **Financial Frictions Models** (4 models)
4. **Heterogeneous Agent Models** (3 models)
5. **Open Economy Models** (4 models)
6. **Environmental/Climate Models** (3 models)
7. **Labor Market Models** (3 models)
8. **Fiscal Policy Models** (3 models)
9. **Pandemic/Epidemiological Models** (2 models)
10. **Emerging Market Models** (3 models)
11. **Recent Innovations** (3 models)

---

## 1. Central Bank DSGE Models

### 1.1 European Central Bank (ECB) Models

#### NAWM II (New Area Wide Model)
- **Type:** Multi-country Euro Area model
- **Status:** Operational at ECB
- **Key Features:**
  - Open economy framework
  - Financial frictions
  - Multiple Euro Area countries
  - Sticky prices and wages
- **Purpose:** Euro Area forecasting and policy analysis
- **References:**
  - ECB Occasional Paper Series
  - Used actively for ECB policy decisions as of 2024

#### ECB-BASE
- **Type:** Semi-structural model
- **Status:** Operational at ECB
- **Key Features:**
  - Semi-structural approach
  - Combines theory and data fit
- **Purpose:** Forecasting and scenario analysis
- **References:**
  - ECB publications (2024)

#### QUEST III
- **Type:** European Commission model
- **Status:** Used by EC and ECB
- **Key Features:**
  - Multi-sector structure
  - Energy sector modeling
  - Climate policy integration
- **Purpose:** Fiscal policy analysis, structural reforms
- **References:**
  - European Commission technical documentation
  - Joint Quarterly Macroeconometric Model

### 1.2 Bank of England Models

#### COMPASS (Central Organising Model for Projection Analysis and Scenario Simulation)
- **Type:** Two-agent open economy DSGE
- **States:** ~50 (estimated)
- **Shocks:** Multiple (monetary, productivity, fiscal, foreign)
- **Observables:** ~13 (GDP, consumption, investment, inflation, wages, etc.)
- **Key Features:**
  - Two-agent framework (savers and hand-to-mouth consumers)
  - Open economy structure for the UK
  - Financial accelerator mechanism
  - Sticky prices and wages
  - **Recent enhancement (2024):** Energy sector integration for analyzing energy price shocks (Russia-Ukraine impact)
- **Purpose:** UK economic forecasting, monetary policy analysis
- **References:**
  - Burgess et al. (2013), "The Bank of England's Forecasting Platform: COMPASS, MAPS, EASE and the Suite of Models"
  - Bank of England Working Paper No. 471
  - Chan et al. (2024), "Energy and climate policy in a DSGE model of the United Kingdom"
  - Bank of England Macro Technical Paper No. 1 (June 2025)
- **Implementation Priority:** HIGH - Well-documented central bank model with recent innovations
- **Status:** Active model undergoing refinement following Bernanke Review (2024)

### 1.3 Bank of Japan Models

#### Q-JEM (Quarterly Japanese Economic Model)
- **Type:** Semi-structural large-scale model
- **Status:** Operational at BOJ (2019 version)
- **Key Features:**
  - Greater disaggregation of expenditure components
  - Detailed financial market information
  - Puts more emphasis on data fit than pure DSGE
  - Relaxes some theoretical discipline for better forecasting
- **Purpose:** Japanese economy forecasting and policy analysis
- **References:**
  - BOJ Working Paper 19-E-07 (2019)
  - "The Quarterly Japanese Economic Model (Q-JEM): 2019 version"
- **Implementation Priority:** MEDIUM - Large-scale model with unique semi-structural approach

#### M-JEM (Medium-scale Japanese Economic Model)
- **Type:** Pure DSGE model
- **Status:** Used alongside Q-JEM at BOJ
- **Key Features:**
  - More theoretically disciplined than Q-JEM
  - Standard New Keynesian features
  - Estimated with Bayesian methods
- **Purpose:** Policy scenario analysis
- **References:**
  - Bank of Japan research papers
- **Implementation Priority:** MEDIUM - Complements Q-JEM with more structural approach

### 1.4 Reserve Bank of Australia Models

#### MARTIN (Macroeconometric And Regional Transmission Impact Network)
- **Type:** Macroeconometric model (not pure DSGE)
- **Status:** Operational at RBA since 2018
- **Key Features:**
  - Describes relationships between key macroeconomic variables
  - More emphasis on empirical fit than theoretical restrictions
  - Used for forecasting and counterfactual scenarios
- **Purpose:** Australian economy forecasting
- **References:**
  - Ballantyne et al. (2019), "MARTIN Has Its Place: A Macroeconometric Model of the Australian Economy"
  - RBA Research Discussion Paper 2019-07
  - RBA Bulletin, March 2018
- **Implementation Priority:** LOW - Not a pure DSGE model

#### MSM (Multi-Sector Model)
- **Type:** Multi-sector DSGE model
- **Status:** Used for scenario analysis at RBA
- **Key Features:**
  - Built on consistent theoretical framework
  - Optimizing households and firms
  - Multiple sectors
  - Forward-looking expectations
  - Useful for structural policy analysis
- **Purpose:** Scenario analysis, structural change analysis
- **References:**
  - Rees, Smith and Hall (2016)
  - Gibbs, Hambur and Nodari (2018) - extensions
  - RBA Research Discussion Papers
- **Implementation Priority:** MEDIUM - Well-structured multi-sector approach

### 1.5 Bank of Canada Models

#### ToTEM III (Terms-of-Trade Economic Model)
- **Type:** Open economy DSGE
- **Status:** Operational (being replaced in 2024-2025)
- **Key Features:**
  - Enhanced household debt modeling
  - Housing market integration
  - Terms of trade dynamics
  - Small open economy framework
  - Financial frictions
- **Purpose:** Canadian economy forecasting and policy analysis
- **References:**
  - Corrigan et al. (2021), "ToTEM III: The Bank of Canada's Main DSGE Model for Projection and Policy Analysis"
  - Bank of Canada Technical Report 119
- **Implementation Priority:** MEDIUM-HIGH - Well-documented, modern features
- **Note:** Bank of Canada announced major model overhaul in 2024 with bounded rationality approach to replace ToTEM

### 1.6 Sveriges Riksbank (Sweden) Models

#### RAMSES (Riksbank Aggregated Macro model for Studies of the Economy of Sweden)
- **Type:** Small open economy DSGE
- **Status:** Operational (evolved into MAJA)
- **Key Features:**
  - Small open economy framework
  - Estimated with Bayesian techniques
  - Sticky prices and wages
  - Financial frictions
- **Purpose:** Swedish economy forecasting
- **References:**
  - Christiano, Trabandt and Walentin (2011)
  - Sveriges Riksbank Occasional Paper Series No. 12
- **Implementation Priority:** MEDIUM
- **Note:** RAMSES II was the second generation

#### MAJA (Modell för Allmän JämviktsAnalys)
- **Type:** Two-region DSGE model
- **Status:** Operational at Riksbank (2020+)
- **Key Features:**
  - Models Sweden and aggregate of main trading partners (Euro Area + US)
  - Better captures dependence on global developments
  - Built on CEE (2005) and Smets-Wouters (2003) foundations
  - Estimated on 1995Q2-2018Q4 data
- **Purpose:** Forecasting, scenarios, policy analysis
- **References:**
  - Sveriges Riksbank Working Paper 391 (2020)
  - "MAJA: A two-region DSGE model for Sweden and its main trading partners"
- **Implementation Priority:** HIGH - Modern two-region approach with good documentation

### 1.7 International Monetary Fund Models

#### GIMF (Global Integrated Monetary and Fiscal Model)
- **Type:** Multi-region DSGE model
- **Status:** Operational at IMF
- **Key Features:**
  - Multi-region structure (can model 3-10+ regions)
  - Forward-looking expectations
  - Comprehensive fiscal sector
  - Monetary policy rules
  - International spillovers and trade
  - Heterogeneous agents (allows for non-Ricardian households)
  - Financial accelerator mechanism
- **Purpose:** Global policy analysis, spillover analysis, risk assessment
- **References:**
  - Kumhof et al. (2010), "The Global Integrated Monetary and Fiscal Model (GIMF) – Theoretical Structure"
  - Anderson et al. (2013), "Getting to Know GIMF: The Simulation Properties of the Global Integrated Monetary and Fiscal Model"
  - IMF Working Papers
- **Implementation Priority:** HIGH - Influential multi-region model
- **Complexity:** Very high - multi-region requires significant calibration

#### GEM (Global Economy Model)
- **Type:** Multi-country DSGE model
- **Status:** Predecessor to GIMF
- **Key Features:**
  - Based on New Open Economy Macroeconomics (NOEM)
  - Multiple countries/regions
  - International spillover analysis
  - Representative agent framework
- **Purpose:** International policy analysis
- **References:**
  - IMF research papers (mid-2000s)
- **Implementation Priority:** LOW - Superseded by GIMF
- **Note:** Less suited for fiscal/debt analysis due to representative agent assumption

#### FSGM (Flexible System of Global Models)
- **Type:** Multi-module global model system
- **Status:** Used at IMF alongside GIMF
- **Key Features:**
  - Modular structure
  - Flexible country coverage
  - Trade modeling
  - Combines multiple modeling approaches
- **Purpose:** Trade analysis, spillover analysis
- **References:**
  - IMF Working Paper 15/64 (2015)
- **Implementation Priority:** LOW - More of a model suite than single DSGE

### 1.8 Philadelphia Fed Model

#### PRISM (Philadelphia Research Intertemporal Stochastic Model)
- **Type:** Medium-scale DSGE
- **Status:** Research model at Philadelphia Fed
- **States:** ~40 equations
- **Observables:** 6 (real GDP growth, consumption growth, investment growth, hours, core PCE inflation, interest rate)
- **Key Features:**
  - Estimated using Bayesian methods
  - Standard New Keynesian features
  - Used for forecasting and policy projects
- **Purpose:** Forecasting, policy analysis
- **References:**
  - Philadelphia Fed website (PRISM section)
  - Philadelphia Fed Research Department publications
- **Implementation Priority:** MEDIUM
- **Note:** PRISM-II is second-generation version under development
- **Important:** PRISM output is NOT an official Philadelphia Fed forecast

**Note on Already Implemented Models:**
- **NYFed Model 1002:** Already implemented in repository ✅
- **St. Louis Fed TANK Model:** Partially implemented in repository ⚠️
- **Cleveland Fed CLEMENTINE:** Initially implemented in repository ⚠️

---

## 2. Canonical Academic DSGE Models

### 2.1 Christiano-Eichenbaum-Evans (CEE) Model (2005)

- **Full Name:** "Nominal Rigidities and the Dynamic Effects of a Shock to Monetary Policy"
- **Type:** Medium-scale New Keynesian DSGE
- **Status:** Highly influential canonical model
- **Key Features:**
  - Sticky prices à la Calvo (firms)
  - Staggered wage contracts à la Calvo (3 quarters average duration)
  - Variable capital utilization
  - Habit formation in consumption
  - Costly investment adjustment
  - Prevents sharp rise in marginal costs after monetary expansion
  - Accounts for inflation inertia and output persistence
- **Estimation:** Matches impulse response functions from VARs
- **Purpose:** Understanding monetary policy transmission
- **References:**
  - Christiano, Eichenbaum, and Evans (2005), Journal of Political Economy, Vol. 113, No. 1
  - Christiano, Trabandt, and Walentin (2010), "DSGE Models for Monetary Policy Analysis," Handbook of Monetary Economics, Vol. 3
- **Implementation Priority:** VERY HIGH - Foundational model
- **Complexity:** Medium - Well-understood and documented

### 2.2 Basic New Keynesian (3-Equation) Model

- **Type:** Canonical textbook model
- **Status:** Foundational teaching model
- **Equations:**
  1. IS curve (consumption Euler equation)
  2. New Keynesian Phillips curve
  3. Taylor rule (monetary policy)
- **Key Features:**
  - Forward-looking expectations
  - Nominal rigidities (Calvo pricing)
  - Simple analytical solutions
- **Purpose:** Teaching, theoretical foundations
- **References:**
  - Galí (2008/2015), "Monetary Policy, Inflation, and the Business Cycle"
  - Woodford (2003), "Interest and Prices"
- **Implementation Priority:** MEDIUM - Already have Simple NK model ✅
- **Note:** Repository already has a simple NK model implemented

### 2.3 Real Business Cycle (RBC) Model

- **Type:** Baseline dynamic general equilibrium
- **Status:** Foundation for DSGE modeling
- **Key Features:**
  - Technology shocks
  - Flexible prices
  - Representative agent
  - Capital accumulation
  - Labor-leisure choice
- **Purpose:** Baseline for understanding business cycles
- **References:**
  - Kydland and Prescott (1982)
  - King and Rebelo (1999), "Resuscitating Real Business Cycles"
- **Implementation Priority:** LOW-MEDIUM - Useful as baseline
- **Complexity:** Low - Well-understood

### 2.4 Rotemberg Pricing Model

- **Type:** Alternative to Calvo pricing
- **Status:** Widely used variant
- **Key Features:**
  - Quadratic price adjustment costs (instead of Calvo)
  - Easier to solve (fewer state variables)
  - Similar aggregate dynamics to Calvo
- **Purpose:** Alternative sticky price modeling
- **References:**
  - Rotemberg (1982)
  - Frequently used in DSGE literature as alternative to Calvo
- **Implementation Priority:** LOW - Framework supports Calvo, could add Rotemberg variant
- **Complexity:** Low - Similar to Calvo implementation

**Note:** Repository already has **Smets-Wouters (2007)** model implemented ✅

---

## 3. Financial Frictions Models

### 3.1 Bernanke-Gertler-Gilchrist (BGG) Financial Accelerator (1999)

- **Type:** DSGE with financial frictions
- **Status:** Most influential pre-crisis financial frictions model
- **Key Features:**
  - Financial accelerator mechanism
  - Credit market frictions amplify shocks
  - External finance premium depends on borrower net worth
  - Costly state verification problem
  - Entrepreneurial borrowing
- **Mechanism:**
  - Shocks → Net worth changes → Credit conditions → Amplification
- **Purpose:** Understanding financial-real economy linkages
- **References:**
  - Bernanke, Gertler, and Gilchrist (1999), "The Financial Accelerator in a Quantitative Business Cycle Framework," Handbook of Macroeconomics
- **Implementation Priority:** HIGH - Foundational financial frictions model
- **Note:** Elements already in NYFed model ✅, but standalone BGG would be valuable

### 3.2 Gertler-Kiyotaki (2010, 2015) Models

#### GK (2010): Financial Intermediation Model
- **Type:** DSGE with banking sector
- **Status:** Influential post-crisis model
- **Key Features:**
  - Explicit financial intermediaries (banks)
  - Bank capital constraints
  - Runs and panics possible
  - Credit policy analysis
- **Purpose:** Understanding financial intermediation in business cycles
- **References:**
  - Gertler and Kiyotaki (2010), "Financial Intermediation and Credit Policy in Business Cycle Analysis," Handbook of Monetary Economics
- **Implementation Priority:** HIGH - Important for financial stability analysis

#### GK (2015): Shadow Banking and Rollover Crises
- **Type:** DSGE with shadow banking sector
- **Status:** Models financial crises
- **Key Features:**
  - Shadow banking sector
  - Rollover crises
  - Fire sales
  - Financial instability dynamics
- **Purpose:** Understanding financial crises
- **References:**
  - Gertler, Kiyotaki, and Prestipino (2016), "Wholesale Banking and Bank Runs in Macroeconomic Modeling of Financial Crises," Handbook of Macroeconomics, Vol. 2
- **Implementation Priority:** MEDIUM - More specialized

### 3.3 Kiyotaki-Moore (1997) Credit Cycles Model

- **Type:** DSGE with collateral constraints
- **Status:** Seminal financial frictions model
- **Key Features:**
  - Collateral constraints on borrowing
  - Asset prices affect borrowing capacity
  - Credit cycles
  - Quantity of loans constrained (vs. BGG price mechanism)
- **Purpose:** Understanding credit booms and busts
- **References:**
  - Kiyotaki and Moore (1997), "Credit Cycles," Journal of Political Economy
- **Implementation Priority:** MEDIUM - Alternative financial frictions approach
- **Complexity:** Medium

### 3.4 Climate Transition Risk with Financial Frictions

- **Type:** Environmental DSGE with financial sector
- **Status:** Recent research frontier (2020s)
- **Key Features:**
  - Climate externality
  - Carbon-intensive vs. green sectors
  - Financial frictions
  - Transition risk from abrupt climate policy
  - Financial stability implications
- **Purpose:** Understanding climate policy and financial stability
- **References:**
  - Vermeulen et al. (2021), "Climate Policy, Financial Frictions, and Transition Risk"
  - NBER Working Paper 28525
- **Implementation Priority:** MEDIUM - Emerging important topic
- **Complexity:** High - Combines multiple model features

---

## 4. Heterogeneous Agent Models

### 4.1 HANK (Heterogeneous Agent New Keynesian) - Kaplan-Moll-Violante (2018)

- **Type:** New Keynesian with heterogeneous agents
- **Status:** Highly influential recent model
- **Observables:** Similar to RANK models (GDP, consumption, investment, inflation, interest rate, etc.)
- **Key Features:**
  - Idiosyncratic income risk (leptokurtic income changes)
  - Two assets: liquid (low return) and illiquid (high return with transaction costs)
  - Heterogeneous marginal propensities to consume (MPC)
  - Realistic wealth distribution
  - Incomplete markets
  - General equilibrium effects dominate intertemporal substitution
- **Key Finding:**
  - Indirect effects through labor demand far outweigh direct effects (intertemporal substitution)
  - Stark contrast to Representative Agent NK (RANK) models
- **Purpose:** Understanding distributional effects of monetary policy
- **References:**
  - Kaplan, Moll, and Violante (2018), "Monetary Policy According to HANK," American Economic Review, Vol. 108(3), pp. 697-743
  - NBER Working Paper 21897
- **Implementation Priority:** VERY HIGH - Important for distributional analysis
- **Complexity:** HIGH - Requires solving for wealth distribution
- **Note:** Repository has St. Louis Fed TANK model (Two-Agent NK) ⚠️ which is simpler version

### 4.2 TANK (Two-Agent New Keynesian) Models

- **Type:** Simplified heterogeneous agent model
- **Status:** Practical alternative to full HANK
- **Key Features:**
  - Two agent types:
    - Ricardian/Patient households (save, smooth consumption)
    - Non-Ricardian/Impatient households (hand-to-mouth, high MPC)
  - Captures key MPC heterogeneity
  - Much simpler to solve than full HANK
  - General equilibrium
- **Purpose:** Distributional effects without full HANK complexity
- **References:**
  - Galí, López-Salido, and Vallés (2007)
  - Cantore and Freund (2021) - with fiscal policy
- **Implementation Priority:** MEDIUM - St. Louis Fed model in repo is TANK variant ⚠️
- **Note:** St. Louis Fed model (repository) is a TANK model with fiscal policy

### 4.3 Bewley-Huggett-Aiyagari Models

- **Type:** Incomplete markets with heterogeneous agents
- **Status:** Foundation for HANK models
- **Key Features:**
  - Idiosyncratic income risk
  - Borrowing constraints
  - Incomplete markets
  - Precautionary savings
  - Stationary distribution of wealth
- **Purpose:** Understanding wealth inequality and precautionary savings
- **References:**
  - Bewley (1986)
  - Huggett (1993)
  - Aiyagari (1994)
- **Implementation Priority:** LOW-MEDIUM - More foundational than policy-relevant
- **Complexity:** Medium-High
- **Note:** Partial equilibrium versions; HANK extends to general equilibrium

---

## 5. Open Economy Models

### 5.1 Galí-Monacelli (2005) Small Open Economy Model

- **Type:** Small open economy New Keynesian
- **Status:** Canonical open economy model
- **Key Features:**
  - Small open economy as part of continuum
  - Identical preferences, technology across economies
  - Calvo staggered price setting
  - Exchange rate dynamics
  - Terms of trade
  - Reveals trade-offs between inflation, output gap, and exchange rate stabilization
- **Purpose:** Monetary policy in open economies
- **References:**
  - Galí and Monacelli (2005), "Monetary Policy and Exchange Rate Volatility in a Small Open Economy," Review of Economic Studies, Vol. 72, pp. 707-734
- **Implementation Priority:** VERY HIGH - Canonical open economy baseline
- **Complexity:** Medium - Well-documented
- **Extensions:**
  - Monacelli (2005) - deviations from law of one price
  - Santacreu (2005) - non-tradable goods, habit, indexation

### 5.2 Two-Country Model

- **Type:** Two symmetric or asymmetric country DSGE
- **Status:** Standard for studying international spillovers
- **Key Features:**
  - Two countries with trade
  - Exchange rate determination
  - International asset markets
  - Spillover effects of policies
- **Purpose:** International policy coordination
- **References:**
  - Numerous variants in literature
  - Basis for multi-country models
- **Implementation Priority:** MEDIUM - Repository could add symmetric 2-country
- **Note:** MAJA (Riksbank) and GIMF (IMF) are multi-region variants

### 5.3 SOE with Commodity Exports (Medina-Soto Models)

- **Type:** Small open economy with commodity sector
- **Status:** Used by central banks in commodity-exporting countries
- **Key Features:**
  - Explicit commodity sector
  - Commodity price shocks
  - Fiscal rules linked to commodity revenues
  - Terms of trade dynamics
  - Dutch disease effects
- **Purpose:** Policy analysis for commodity exporters (Chile, Australia, Norway, Canada, etc.)
- **References:**
  - Medina and Soto (2005/2007), "Oil Shocks and Monetary Policy in an Estimated DSGE Model"
  - Medina, Munro, and Soto (2008), "What Drives the Current Account in Commodity Exporting Countries?"
  - Medina and Soto (2016), "Commodity prices and fiscal policy in a commodity exporting economy"
- **Implementation Priority:** HIGH - Important for many economies
- **Complexity:** Medium
- **Applications:** Chile (copper), Norway (oil), Australia (minerals), Canada (oil)

### 5.4 Emerging Market DSGE Models

- **Type:** SOE adapted for emerging markets
- **Status:** Used by EM central banks
- **Key Features:**
  - Foreign currency debt
  - Country risk premium
  - Sudden stops
  - Currency mismatches
  - Imperfect capital mobility
  - Liability dollarization
- **Purpose:** EM-specific policy challenges
- **References:**
  - Various central bank models (Chile, Colombia, Peru, South Africa, etc.)
  - Aguiar and Gopinath (2007) on EM business cycles
- **Implementation Priority:** MEDIUM - Specialized but important
- **Complexity:** Medium-High

---

## 6. Environmental and Climate DSGE Models

### 6.1 Heutel (2012) Environmental Business Cycle Model

- **Type:** RBC/NK model with pollution
- **Status:** Foundational E-DSGE model
- **Key Features:**
  - Pollution as state variable
  - Pollution externality
  - Environmental policy (taxes or cap)
  - Business cycle interactions with environmental policy
- **Purpose:** Optimal environmental policy over business cycle
- **References:**
  - Heutel (2012), "How should environmental policy respond to business cycles? Optimal policy under persistent productivity shocks," Review of Economic Dynamics
- **Implementation Priority:** HIGH - Foundational environmental model
- **Complexity:** Medium

### 6.2 Golosov-Hassler-Krusell-Tsyvinski (2014) Climate Model

- **Type:** Growth model with climate change
- **Status:** Influential climate-macro model
- **Key Features:**
  - Fossil fuel extraction and use
  - CO2 emissions and atmospheric concentration
  - Climate damages
  - Optimal carbon tax
- **Purpose:** Long-run climate policy analysis
- **References:**
  - Golosov, Hassler, Krusell, and Tsyvinski (2014), "Optimal Taxes on Fossil Fuel in General Equilibrium," Econometrica
- **Implementation Priority:** MEDIUM-HIGH - Important climate economics
- **Complexity:** Medium-High - Long-run dynamics

### 6.3 ECB Climate-Macro Model (Vermandel et al. 2024)

- **Type:** New Keynesian with climate externality
- **Status:** Recent central bank model
- **Key Features:**
  - Green and brown sectors
  - Carbon pricing
  - Green transition dynamics
  - Financial sector interactions
  - Climate shocks
- **Purpose:** Central bank climate policy analysis
- **References:**
  - Sahuc, Smets, and Vermandel (2024), "The New Keynesian Climate Model"
  - ECB Conference presentation (April 2024)
- **Implementation Priority:** HIGH - Cutting-edge central bank application
- **Complexity:** High
- **Note:** Also see Bank of England energy/climate model (Chan et al. 2024) ✅

---

## 7. Labor Market Models

**Note:** Repository already has **Galí (2010)** model with Mortensen-Pissarides labor search ✅

### 7.1 Mortensen-Pissarides Diamond (DMP) Search Model

- **Type:** Labor search and matching
- **Status:** Canonical labor market model
- **Key Features:**
  - Job vacancies posted by firms
  - Unemployed workers search
  - Matching function
  - Nash wage bargaining
  - Job destruction
  - Unemployment as equilibrium outcome
- **Challenges:** Shimer (2005) puzzle - low volatility of unemployment
- **Purpose:** Understanding labor market dynamics
- **References:**
  - Diamond (1982), Mortensen-Pissarides (1994)
  - Pissarides (2000), "Equilibrium Unemployment Theory"
  - Shimer (2005), "The Cyclical Behavior of Equilibrium Unemployment and Vacancies"
- **Implementation Priority:** MEDIUM - Galí (2010) already implements this ✅
- **Complexity:** Medium

### 7.2 Gertler-Trigari (2009) Model

- **Type:** New Keynesian with search frictions
- **Status:** Combines NK and DMP
- **Key Features:**
  - DMP labor search
  - Sticky prices (Calvo)
  - Monetary policy
  - Alternative wage determination (staggered Nash bargaining)
  - Better matches unemployment volatility
- **Purpose:** Monetary policy with labor frictions
- **References:**
  - Gertler and Trigari (2009), "Unemployment Fluctuations with Staggered Nash Wage Bargaining," Journal of Political Economy
- **Implementation Priority:** MEDIUM-HIGH - Important NK + labor frictions combination
- **Complexity:** Medium-High

### 7.3 Hall (2005) Wage Rigidity Model

- **Type:** DMP with wage rigidity
- **Status:** Response to Shimer puzzle
- **Key Features:**
  - DMP framework
  - Rigid wages (not Nash bargaining)
  - Higher unemployment volatility
- **Purpose:** Resolving unemployment volatility puzzle
- **References:**
  - Hall (2005), "Employment Fluctuations with Equilibrium Wage Stickiness," American Economic Review
- **Implementation Priority:** LOW-MEDIUM - Specialized labor model variant
- **Complexity:** Medium

---

## 8. Fiscal Policy Models

### 8.1 Leeper-Traum-Walker Fiscal Model

- **Type:** New Keynesian with detailed fiscal sector
- **Status:** Influential fiscal DSGE model
- **Key Features:**
  - Active vs. passive monetary policy regimes
  - Active vs. passive fiscal policy regimes
  - Regime-dependent fiscal multipliers
  - Debt dynamics
  - Multiple tax instruments
  - Government spending
- **Key Findings:**
  - Short-run multipliers ~1.4 across regimes
  - Long-run multipliers: 1.9 (passive money/active fiscal) vs. 0.7 (active money/passive fiscal)
  - Investment multipliers can be negative under active monetary policy
- **Purpose:** Understanding fiscal multipliers and policy regimes
- **References:**
  - Leeper, Traum, and Walker (2017), "Clearing Up the Fiscal Multiplier Morass," American Economic Review
  - Leeper, Plante, and Traum (2010), "Dynamics of Fiscal Financing in the United States," Journal of Econometrics
- **Implementation Priority:** VERY HIGH - Important for fiscal policy analysis
- **Complexity:** Medium-High

### 8.2 Coenen et al. Euro Area Fiscal Model

- **Type:** Multi-country Euro Area fiscal model
- **Status:** Used for Euro Area fiscal analysis
- **Key Features:**
  - Multiple Euro Area countries
  - Fiscal rules
  - Government debt sustainability
  - Spillovers across countries
- **Purpose:** Euro Area fiscal policy coordination
- **References:**
  - Coenen et al. (various ECB publications)
  - Used for evaluating Euro Area fiscal multipliers
- **Implementation Priority:** MEDIUM - Specialized to Euro Area
- **Complexity:** High - Multi-country

### 8.3 Fiscal Theory of the Price Level (FTPL) Models

- **Type:** DSGE with non-Ricardian fiscal regime
- **Status:** Alternative to monetary-dominant regime
- **Key Features:**
  - Fiscal policy determines price level
  - Passive monetary policy
  - Government budget constraint as equilibrium condition
  - Inflation depends on fiscal variables
- **Purpose:** Understanding fiscal-monetary interactions
- **References:**
  - Leeper (1991), Sims (1994), Woodford (1994, 1995)
  - Cochrane (various)
- **Implementation Priority:** MEDIUM - Important alternative paradigm
- **Complexity:** Medium

---

## 9. Pandemic and Epidemiological Models

### 9.1 Eichenbaum-Rebelo-Trabandt (ERT) SIR-Macro Model (2020-2022)

- **Type:** New Keynesian + SIR epidemiological model
- **Status:** Influential COVID-19 model
- **Key Features:**
  - SIR (Susceptible-Infected-Recovered) epidemiology
  - Endogenous infection through economic activity
  - Two-way feedback: epidemic → economy, economy → epidemic
  - Consumption and work activities spread virus
  - **Supply effect:** Workers reduce labor to avoid infection
  - **Demand effect:** Consumers reduce consumption to avoid infection
  - Containment policies (lockdowns)
- **Key Findings:**
  - Epidemic causes recession through voluntary social distancing
  - Fall in economic activity slows pandemic spread
  - Optimal policy balances health and economic outcomes
- **Purpose:** Understanding pandemic economics
- **References:**
  - Eichenbaum, Rebelo, and Trabandt (2021), "The Macroeconomics of Epidemics," Review of Financial Studies, Vol. 34(11), pp. 5149-5187
  - Eichenbaum, Rebelo, and Trabandt (2022), "Epidemics in the New Keynesian Model," Journal of Economic Dynamics and Control
  - NBER Working Paper 26882 (2020)
- **Implementation Priority:** HIGH - Important for future pandemic preparedness
- **Complexity:** Medium-High
- **Code Availability:** Epidemic-Macro Model Database (epi-mmb.com)

### 9.2 Extensions with Labor Market Heterogeneity

- **Type:** SIR-DSGE with skill heterogeneity
- **Status:** Recent research (2024)
- **Key Features:**
  - High-skilled vs. low-skilled workers
  - Different infection risks by occupation
  - Distributional impacts of pandemic
- **Purpose:** Understanding inequality in pandemic impacts
- **References:**
  - Chu (2024), "The impact of the COVID-19 pandemic on high-skilled and low-skilled labor markets in a DSGE model," Scottish Journal of Political Economy
- **Implementation Priority:** MEDIUM - Extension of baseline SIR-Macro
- **Complexity:** High

---

## 10. Emerging Market and Commodity Models

### 10.1 Aguiar-Gopinath Emerging Market Model

- **Type:** Small open economy for emerging markets
- **Status:** Influential EM business cycle model
- **Key Features:**
  - Trend shocks (not just cyclical)
  - Country risk premium
  - Sudden stops
  - Counter-cyclical current account
  - Emerging market business cycle properties
- **Purpose:** Understanding EM business cycles
- **References:**
  - Aguiar and Gopinath (2007), "Emerging Market Business Cycles: The Cycle is the Trend," Journal of Political Economy
- **Implementation Priority:** HIGH - Important for EM analysis
- **Complexity:** Medium

### 10.2 Commodity Exporter Models

Already covered in Section 5.3 (Medina-Soto models)

### 10.3 Sudden Stops Models

- **Type:** EM with financial crisis dynamics
- **Status:** Models sudden capital outflows
- **Key Features:**
  - Occasionally binding borrowing constraint
  - Collateral constraints
  - Sudden stops in capital flows
  - Current account reversals
  - Fire sales
- **Purpose:** Understanding EM financial crises
- **References:**
  - Mendoza (2010)
  - Bianchi (2011)
- **Implementation Priority:** MEDIUM - Specialized but important
- **Complexity:** High - Occasionally binding constraints (could use OccBin!)

---

## 11. Recent Methodological Innovations

### 11.1 Neural Network DSGE Solutions

- **Type:** Deep learning for solving DSGE models
- **Status:** Emerging research frontier (2023-2024)
- **Key Features:**
  - Neural networks approximate policy functions
  - Handles high-dimensional state spaces
  - Preserves non-linearity
  - Avoids curse of dimensionality
  - Hard constraints to satisfy economic conditions
- **Purpose:** Solving complex DSGE models
- **References:**
  - Beck et al. (2024), "Deep learning solutions of DSGE models," Central Bank of Luxembourg Technical Report
  - arXiv:2310.13436 (2024), "Non-linear approximations of DSGE models with neural-networks and hard-constraints"
- **Implementation Priority:** LOW - Methodological, not new model
- **Complexity:** Very High - Requires ML infrastructure
- **Note:** Could be framework enhancement rather than new model

### 11.2 Deep Reinforcement Learning in DSGE

- **Type:** RL agents replace optimization
- **Status:** Very recent research (2024)
- **Key Features:**
  - DRL agent learns vacancy posting (firms)
  - Replaces analytical optimization
  - Can generate higher volatility
  - Different dynamics than log-linearized solutions
- **Purpose:** Alternative solution method
- **References:**
  - Recent papers in search-matching models with DRL (2024)
- **Implementation Priority:** LOW - Methodological exploration
- **Complexity:** Very High

### 11.3 Bounded Rationality DSGE Models

- **Type:** DSGE with learning and bounded rationality
- **Status:** Growing research area, adopted by Bank of Canada (2024)
- **Key Features:**
  - Agents learn about economy structure
  - Deviations from rational expectations
  - Better matches expectation survey data
  - More realistic inflation dynamics
- **Purpose:** More realistic expectation formation
- **References:**
  - Bank of Canada's planned new model (2024 announcement)
  - Various academic papers on learning in DSGE
- **Implementation Priority:** MEDIUM - Important methodological direction
- **Complexity:** High

---

## 12. Summary Table: Implementation Priorities

| Priority | Model | Category | Complexity | Documentation |
|----------|-------|----------|------------|---------------|
| **VERY HIGH** | CEE (2005) | Canonical | Medium | Excellent |
| **VERY HIGH** | HANK (Kaplan-Moll-Violante) | Heterogeneous Agents | High | Excellent |
| **VERY HIGH** | Galí-Monacelli (2005) | Open Economy | Medium | Excellent |
| **VERY HIGH** | Leeper-Traum-Walker | Fiscal | Medium-High | Excellent |
| **HIGH** | COMPASS (Bank of England) | Central Bank | High | Very Good |
| **HIGH** | MAJA (Riksbank) | Central Bank, Multi-Region | High | Very Good |
| **HIGH** | GIMF (IMF) | Central Bank, Multi-Region | Very High | Very Good |
| **HIGH** | BGG (1999) Financial Accelerator | Financial Frictions | Medium-High | Excellent |
| **HIGH** | Gertler-Kiyotaki (2010) | Financial Frictions | High | Good |
| **HIGH** | Heutel (2012) E-DSGE | Environmental | Medium | Good |
| **HIGH** | Aguiar-Gopinath EM Model | Emerging Markets | Medium | Good |
| **HIGH** | Medina-Soto Commodity Model | Open Economy | Medium | Good |
| **HIGH** | ERT SIR-Macro Model | Pandemic | Medium-High | Excellent |
| **HIGH** | ECB Climate Model (2024) | Environmental | High | Good |
| **MEDIUM-HIGH** | ToTEM III (Bank of Canada) | Central Bank | High | Very Good |
| **MEDIUM-HIGH** | Gertler-Trigari (2009) | Labor | Medium-High | Good |
| **MEDIUM-HIGH** | Golosov et al. Climate | Environmental | Medium-High | Good |
| **MEDIUM** | Q-JEM (Bank of Japan) | Central Bank | Very High | Good |
| **MEDIUM** | M-JEM (Bank of Japan) | Central Bank | High | Fair |
| **MEDIUM** | MSM (RBA Multi-Sector) | Central Bank | Medium-High | Good |
| **MEDIUM** | PRISM (Philadelphia Fed) | Central Bank | Medium | Fair |
| **MEDIUM** | RAMSES (Riksbank) | Central Bank | Medium | Good |
| **MEDIUM** | Kiyotaki-Moore (1997) | Financial Frictions | Medium | Good |
| **MEDIUM** | Climate-Financial Frictions | Environmental | High | Fair |
| **MEDIUM** | Two-Country Model | Open Economy | Medium | Good |
| **MEDIUM** | Coenen et al. Euro Fiscal | Fiscal | High | Good |
| **MEDIUM** | FTPL Models | Fiscal | Medium | Good |
| **MEDIUM** | Pandemic Labor Heterogeneity | Pandemic | High | Fair |
| **MEDIUM** | EM Sudden Stops | Emerging Markets | High | Good |
| **MEDIUM** | Bounded Rationality DSGE | Methodology | High | Fair |

---

## 13. Sources and References

### Academic Journals
- American Economic Review
- Journal of Political Economy
- Review of Economic Studies
- Econometrica
- Review of Economic Dynamics
- Journal of Monetary Economics
- Journal of Economic Dynamics and Control
- Review of Financial Studies

### Book References
- Handbook of Monetary Economics (2010, 2011) - Elsevier, edited by Friedman and Woodford
- Handbook of Macroeconomics (1999, 2016) - Elsevier
- Galí (2008/2015), "Monetary Policy, Inflation, and the Business Cycle" - Princeton University Press
- Woodford (2003), "Interest and Prices" - Princeton University Press
- Herbst and Schorfheide (2016), "Bayesian Estimation of DSGE Models" - Princeton University Press

### Central Bank Publications
- Federal Reserve Banks (New York, Philadelphia, St. Louis, Cleveland, Dallas, etc.)
  - Staff Reports, Working Papers, Technical Reports
- European Central Bank
  - Working Paper Series, Occasional Paper Series
- Bank of England
  - Working Papers, Staff Working Papers, Quarterly Bulletin
- Bank of Japan
  - Working Paper Series
- Reserve Bank of Australia
  - Research Discussion Papers, Bulletin
- Bank of Canada
  - Technical Reports, Staff Working Papers
- Sveriges Riksbank
  - Working Paper Series, Occasional Paper Series, Staff Memos
- International Monetary Fund
  - Working Papers, Staff Reports

### NBER Working Papers
- Extensive DSGE model research published as NBER Working Papers
- Many eventually published in top journals

### Online Resources
- Dynare Model Database (DSGE_mod repository by Johannes Pfeifer)
- Epidemic-Macro Model Database (epi-mmb.com)
- Central bank websites with model documentation
- Author websites with replication codes

### Key Researchers and Their Websites
- Lawrence Christiano (Northwestern)
- Martin Eichenbaum (Northwestern)
- Charles Evans (Chicago Fed)
- Jordi Galí (CREI, Pompeu Fabra)
- Mark Gertler (NYU)
- Frank Schorfheide (UPenn)
- Chris Sims (Princeton)
- Michael Woodford (Columbia)
- Tommaso Monacelli (Bocconi)
- Giovanni Violante (Princeton)
- Benjamin Moll (LSE)
- Greg Kaplan (Chicago)
- Eric Leeper (Virginia)
- Mathias Trabandt (various)

### Replication Code Repositories
- Johannes Pfeifer's Dynare repository (GitHub)
- Author websites (replication files)
- Journal replication archives
- Central bank model codes (where available)

---

## 14. Recommendations for Implementation

### Phase 1: High-Priority Canonical Models (3-6 months)
1. **Christiano-Eichenbaum-Evans (2005)** - Foundational medium-scale NK
2. **Galí-Monacelli (2005)** - Canonical small open economy
3. **BGG Financial Accelerator (standalone)** - Financial frictions baseline
4. **Leeper-Traum-Walker** - Fiscal policy and multipliers

### Phase 2: Heterogeneous Agents (6-12 months)
1. **HANK (Kaplan-Moll-Violante 2018)** - Full heterogeneous agent model
2. Complete **St. Louis Fed TANK model** (already partially implemented)

### Phase 3: Important Central Bank Models (6-12 months)
1. **COMPASS (Bank of England)** - Two-agent with energy sector
2. **MAJA (Riksbank)** - Two-region model
3. **ToTEM III (Bank of Canada)** - Modern SOE with housing/debt

### Phase 4: Environmental and Labor Extensions (6-9 months)
1. **Heutel (2012)** - E-DSGE baseline
2. **ECB Climate Model (2024)** - Modern climate-macro
3. **Gertler-Trigari (2009)** - NK + search frictions

### Phase 5: Pandemic and Advanced Models (3-6 months)
1. **ERT SIR-Macro Model** - Pandemic economics
2. **GIMF** - Multi-region (ambitious, long-term project)
3. **Aguiar-Gopinath** - Emerging markets

### Quick Wins (1-3 months each)
- **RBC Model** - Baseline without nominal rigidities
- **Rotemberg pricing variant** - Alternative to Calvo
- **Two-country symmetric model** - International spillovers
- **Commodity exporter model (Medina-Soto)** - Resource economies

---

## 15. Models Already in Repository ✅

1. **Simple New Keynesian (NK) Model** - Complete ✅
2. **NYFed DSGE Model 1002** - Complete ✅
3. **Smets-Wouters (2007)** - Complete ✅
4. **Galí (2010) Labor Search Model** - Complete ✅
5. **St. Louis Fed TANK Model** - Partial ⚠️
6. **Cleveland Fed CLEMENTINE** - Initial ⚠️

**Repository Status:** 6 models (4 complete, 2 partial)

---

## 16. Conclusion

This literature review identified **30+ distinct DSGE models** across 11 categories. The models range from canonical academic benchmarks (CEE 2005, Smets-Wouters 2007 ✅) to cutting-edge central bank operational models (COMPASS, MAJA, GIMF) to emerging research frontiers (HANK, climate models, pandemic models).

### Key Takeaways:

1. **Central bank models** are becoming increasingly sophisticated, incorporating:
   - Financial frictions and banking sectors
   - Heterogeneous agents (two-agent and full HANK)
   - Environmental/energy sectors
   - Multi-region interactions
   - Occasionally binding constraints (ZLB, borrowing constraints)

2. **Heterogeneous agent models** (HANK, TANK) are the major recent innovation, showing monetary policy works very differently with realistic wealth distributions

3. **Climate/environmental DSGE** is a rapidly growing area with policy relevance

4. **Pandemic models** (SIR-Macro) showed DSGE models can be adapted to new challenges quickly

5. **Open economy models** remain crucial, especially for small open economies and commodity exporters

6. **Financial frictions** (BGG, Gertler-Kiyotaki) are now standard in central bank models

### Implementation Strategy:

The recommended implementation strategy prioritizes:
- **Canonical models** that establish credibility (CEE 2005, Galí-Monacelli 2005)
- **Policy-relevant extensions** (fiscal, climate, heterogeneous agents)
- **Modern central bank models** for operational relevance
- **Documentation and replication** to build user base

The dsge-py framework is well-positioned to implement these models given its:
- Flexible model specification interface
- OccBin support for occasionally binding constraints
- Bayesian estimation infrastructure (SMC)
- Data integration (FRED API)

---

**Compiled by:** Claude (Anthropic AI)
**Date:** January 2025
**Repository:** dsge-py (https://github.com/rndubs/dsge-py)
