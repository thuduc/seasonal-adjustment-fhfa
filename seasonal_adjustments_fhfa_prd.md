# FHFA House Price Index Seasonal Adjustment - Product Requirements Document

## Executive Summary

This PRD defines the functional requirements for implementing seasonal adjustments to house price indices (HPIs) based on the Federal Housing Finance Agency (FHFA) methodology described in "Applying Seasonal Adjustments to Housing Markets" by William M. Doerner and Wenzhen Lin (2022).

## Background

House price seasonality has been increasing over the last decade. This system implements seasonal adjustment techniques to remove predictable patterns from house price data, allowing better understanding of housing market conditions.

## Core Methodology

The system uses X-13ARIMA-SEATS, which merges X-12-ARIMA and TRAMO-SEATS programs for seasonal adjustment.

## Functional Requirements

### 1. RegARIMA Model Implementation

#### 1.1 Primary Equation
The system must implement the general regression for a regARIMA model:

```
φ(B)Φ(Bs)(1 - B)^d(yt - Σ βixit) = θ(B)Θ(Bs)εt
```

#### 1.2 Variable Definitions
- `yt`: Dependent variable to be adjusted
- `B`: Lag operator where Byt = yt-1
- `s`: Seasonal period (quarterly = 4)
- `d`: Degree of nonseasonal differencing
- `xit`: Regression variables
- `βi`: Regression coefficients
- `εt`: Independent and identically distributed errors with mean 0 and variance σ²

#### 1.3 Operator Definitions
- `φ(B) = 1 - φ1B - ... - φpBp`: Regular autoregressive operator
- `Φ(Bs) = 1 - Φ1Bs - ... - ΦPBPs`: Seasonal autoregressive operator  
- `θ(B) = 1 - θ1B - ... - θqBq`: Regular moving average operator
- `Θ(Bs) = 1 - Θ1Bs - ... - ΘQBQs`: Seasonal moving average operator

### 2. Seasonality Impact Analysis

#### 2.1 Linear Regression Model
The system must support analysis of seasonality impacts using:

```
yit = α + β1Wit + β2N.salesit + β3Chari,2010 + β4Industryi,2010 + γi + f(year) + εit
```

#### 2.2 Quantile Regression Model
For distributional analysis, implement:

```
Qτ(yit) = ατ + β1τWit + β2τN.salesit + β3τChari,2010 + β4τIndustryi,2010 + γiτ + fτ(year) + εit
```

#### 2.3 Variable Specifications

**Dependent Variable:**
- `yit`: Absolute value of difference between nonseasonally adjusted and seasonally adjusted HPI for ith State/MSA at each quarter

**Weather Variables (Wit):**
- `Range Temp Q1-Q4`: Difference between max and min temperature for each quarter
- `Average Temp`: Average temperature
- `Precipitation`: Precipitation levels

**Control Variables:**
- `N.salesit`: Average number of houses sold in each quarter
- `Chari,2010`: Household characteristics including:
  - Average household income
  - Percentage of white population
  - Percentage of population over 65 years old
  - Percentage with bachelor's degree or higher
  - Percentage of single-family sales
- `Industryi,2010`: Industry concentration shares (top 10 industries)
- `γi`: State/MSA fixed effects
- `f(year)`: Linear splines for years with break points at 1998, 2007, 2011, 2020

### 3. Processing Requirements

#### 3.1 ARIMA Model Selection
The system must:
- Perform automatic ARIMA model identification
- Support three stages: model building, seasonal adjustment, and diagnostic checking
- Apply outlier detection and correction
- Conduct normality tests and Ljung-Box Q tests

#### 3.2 Geographic Coverage
Process seasonal adjustments for:
- United States and Census Divisions
- All 50 states (Hawaii excluded from weather analysis)
- Top 100 Metropolitan Statistical Areas (MSAs)
- Puerto Rico
- Three-digit ZIP codes

#### 3.3 Index Types
Support multiple index types:
- Purchase-Only Indices
- All-Transactions Indices  
- Expanded-Data Indices
- Distress-Free Measures
- Manufactured Homes

### 4. Data Requirements

#### 4.1 Input Data Sources
- FHFA repeat-sales transaction data
- NOAA weather data (temperature, precipitation)
- Census demographic data (2010)
- Industry concentration data

#### 4.2 Time Series Requirements
- Quarterly frequency data
- Historical data from 1991-2020 for calibration
- Support for forecasting future periods

### 5. Output Specifications

#### 5.1 Adjusted Indices
Generate seasonally adjusted HPIs with:
- Monthly appreciation rates
- Annual appreciation rates
- Forecast estimates with 95% confidence intervals

#### 5.2 Diagnostic Outputs
Provide:
- Autocorrelation and partial autocorrelation plots
- Residual diagnostics
- Model fit statistics (R², BIC)
- Comparison metrics between adjusted and non-adjusted indices

### 6. Performance Requirements

#### 6.1 Processing Times (Approximate)
- United States and Census Divisions: 15 minutes
- States: 1.5 hours
- 100 Largest MSAs: 2.5 hours
- All MSAs and Divisions: 24 hours

### 7. Technical Specifications

#### 7.1 Software Dependencies
- X-13ARIMA-SEATS program (Census Bureau)
- Statistical computing environment supporting:
  - Linear regression
  - Quantile regression
  - Time series analysis
  - Fixed effects models

#### 7.2 Model Parameters
Default ARIMA specifications derived from optimal term structures, with support for:
- Nonseasonal AR terms
- Nonseasonal MA terms  
- Seasonal differencing
- Regression variables for outliers and level shifts

### 8. Validation Requirements

#### 8.1 Model Validation
- Compare results between X-12 and X-13 implementations
- Validate forecast accuracy against actual data
- Test sensitivity to parameter changes

#### 8.2 Statistical Tests
- Ljung-Box test for residual autocorrelation (p-value > 0.05)
- Normality tests for residuals
- Outlier detection and adjustment

### 9. Special Considerations

#### 9.1 COVID-19 Adjustments
The system must handle pandemic-era data with options for:
- Concurrent seasonal adjustment with outlier commands
- Structural break detection
- Alternative forecasting methods for extreme events

#### 9.2 Weather Impact Analysis
Implement specialized handling for:
- Extreme temperature ranges affecting seasonality
- Geographic variations in weather impacts
- Quarterly temperature range calculations

### 10. Future Enhancements

Consider implementation of:
- Real-time seasonal adjustment updates
- Alternative seasonal adjustment methods (CAMPLET, CiSSA)
- Machine learning approaches for pattern detection
- Extended geographic coverage beyond top 100 MSAs

## Appendix: Key Coefficients from Analysis

### Temperature Range Impacts (State Level, Model 5)
- Q1: 0.011*** (increases seasonality)
- Q2: 0.014*** (increases seasonality)  
- Q3: 0.027*** (increases seasonality)
- Q4 (baseline): -0.018*** (decreases seasonality)

### Time Trend Coefficients
- 1991-1998: 0.021*** (slow increase)
- 1999-2007: 0.126*** (rapid increase)
- 2008-2011: 0.071*** (continued increase)
- 2012-2020: 0.015*** (slowdown)

Note: *** indicates statistical significance at p < 0.01