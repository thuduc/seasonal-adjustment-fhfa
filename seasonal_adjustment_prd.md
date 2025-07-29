# Product Requirements Document: Housing Price Seasonal Adjustment Model

## 1. Executive Summary

### 1.1 Purpose
This PRD defines the requirements for implementing a seasonal adjustment model for housing price indices based on the methodology developed by Doerner and Lin (2022). The model will provide accurate seasonal adjustments for housing markets across different geographic regions, accounting for weather patterns, demographic characteristics, and market conditions.

### 1.2 Goals
- Implement X-13ARIMA-SEATS seasonal adjustment methodology
- Calculate seasonal adjustments for state and MSA-level housing price indices
- Identify geographic regions requiring more intensive seasonal adjustments
- Provide insights on the relationship between weather patterns and housing market seasonality

### 1.3 Success Metrics
- Model accuracy: R² > 0.4 for state-level adjustments
- Processing time: < 5 minutes for 100 MSAs
- Coverage: Support for 50 states and 400+ MSAs
- Stability: Consistent adjustments across multiple runs

## 2. Background and Context

### 2.1 Problem Statement
Housing price seasonality has increased significantly over the past decade, yet seasonal adjustment methods have remained largely unchanged. Current approaches fail to account for:
- Geographic variations in seasonal patterns
- Weather-related impacts on housing transactions
- Demographic and industry influences on seasonality
- The optimal frequency for recalculation

### 2.2 Current State
- Existing tools use outdated X-12 methodology
- Limited geographic coverage (major markets only)
- Infrequent updates to seasonal factors
- No systematic approach to identifying where adjustments are most needed

### 2.3 Target Users
- Housing market analysts
- Real estate researchers
- Government agencies (FHFA, HUD)
- Financial institutions
- Real estate technology companies

## 3. Functional Requirements

### 3.1 Data Input Module

#### 3.1.1 Housing Price Data
- **Input Format**: Quarterly repeat-sales price indices
- **Required Fields**:
  - Geographic identifier (State/MSA code)
  - Time period (YYYY-Q#)
  - Non-seasonally adjusted HPI value
  - Number of transactions
- **Data Sources**: FHFA, CoreLogic, or custom indices
- **Validation**: Check for missing values, outliers, minimum 5-year history

#### 3.1.2 Weather Data
- **Input Format**: Quarterly weather statistics by geography
- **Required Fields**:
  - Temperature range (max - min for quarter)
  - Average temperature
  - Total precipitation
- **Data Sources**: NOAA or equivalent
- **Validation**: Reasonable ranges, completeness checks

#### 3.1.3 Demographic Data
- **Input Format**: Census data by geography
- **Required Fields**:
  - Percentage over 65 years old
  - Percentage of white population
  - Percentage with bachelor's degree or higher
  - Percentage of single-family home sales
  - Average household income
- **Data Sources**: US Census Bureau, ACS
- **Validation**: Percentages sum appropriately, data currency

#### 3.1.4 Industry Composition Data
- **Input Format**: Employment share by industry
- **Required Industries** (Top 10):
  - Healthcare and Social Assistance
  - Manufacturing
  - Professional Services
  - Retail
  - Education
  - Finance, Insurance, Real Estate (FIRE)
  - Public Administration
  - Construction
  - Transportation
  - Agriculture
- **Validation**: Shares sum to reasonable total

### 3.2 Seasonal Adjustment Engine

#### 3.2.1 X-13ARIMA-SEATS Implementation
- **Core Algorithm**: RegARIMA model with seasonal components
- **Model Specification**:
  ```
  yt = x'it*β + D't*δ + εit
  where:
  - yt = HPI at time t
  - x'it = regression variables
  - D't = time dummies
  - εit = error term following ARIMA process
  ```
- **ARIMA Components**:
  - Autoregressive (AR): p terms
  - Integrated (I): d terms  
  - Moving Average (MA): q terms
  - Seasonal components: (P,D,Q)s

#### 3.2.2 Model Stages
1. **Pre-adjustment Stage**:
   - Trading day effects removal
   - Moving holiday effects adjustment
   - Outlier detection and treatment
   
2. **Seasonal Adjustment Stage**:
   - Parameter estimation via maximum likelihood
   - Seasonal factor extraction
   - Trend-cycle estimation
   
3. **Diagnostic Stage**:
   - Ljung-Box Q test for residuals
   - Normality tests
   - Stability diagnostics

#### 3.2.3 Optimal Parameter Selection
- **Automatic Model Selection**: 
  - Test multiple ARIMA specifications
  - Use AIC/BIC criteria for selection
  - Validate with out-of-sample forecasts
- **Constraints**:
  - Maximum AR order: 4
  - Maximum MA order: 4
  - Maximum seasonal order: 2

### 3.3 Causation Analysis Module

#### 3.3.1 Linear Regression Model
- **Dependent Variable**: |NSA_HPI - SA_HPI|
- **Independent Variables**:
  - Weather controls (by quarter)
  - Demographic characteristics
  - Industry shares
  - Geographic fixed effects
  - Time trends (spline functions)
- **Model Specification**:
  ```
  yit = α + β1*Wit + β2*N.salesit + β3*Chari + β4*Industryi + γi + f(year) + εit
  ```

#### 3.3.2 Quantile Regression
- **Quantiles**: 0.1, 0.25, 0.5, 0.75, 0.9
- **Purpose**: Identify heterogeneous effects across seasonality distribution
- **Output**: Coefficient estimates by quantile

### 3.4 Output Generation

#### 3.4.1 Adjusted Index Series
- **Format**: Time series with SA and NSA values
- **Frequency**: Quarterly
- **Components**:
  - Original NSA index
  - Seasonally adjusted index
  - Seasonal factors
  - Trend-cycle component
  - Irregular component

#### 3.4.2 Diagnostic Reports
- **Model Fit Statistics**:
  - R-squared values
  - AIC/BIC scores
  - Forecast accuracy metrics
- **Seasonal Patterns**:
  - Seasonal factors by quarter
  - Stability tests over time
  - Geographic heat maps

#### 3.4.3 Causation Analysis Results
- **Weather Impact Summary**:
  - Temperature range effects by quarter
  - Precipitation impacts
  - Geographic variations
- **Demographic Influences**:
  - Key demographic drivers
  - Industry composition effects
  - Market thickness impacts

## 4. Non-Functional Requirements

### 4.1 Performance
- **Processing Speed**: 
  - Single geography: < 3 seconds
  - 100 geographies: < 5 minutes
  - Full dataset (400+ MSAs): < 30 minutes
- **Memory Usage**: < 8GB RAM for full dataset
- **Scalability**: Support parallel processing for multiple geographies

### 4.2 Reliability
- **Uptime**: 99.9% availability
- **Error Handling**: Graceful degradation for missing data
- **Consistency**: Reproducible results with same inputs

### 4.3 Usability
- **API Design**: RESTful endpoints for all operations
- **Documentation**: Comprehensive API docs and examples
- **Error Messages**: Clear, actionable error descriptions

### 4.4 Security
- **Data Encryption**: TLS for data in transit
- **Access Control**: API key authentication
- **Audit Logging**: Track all model runs and changes

## 5. Technical Architecture

### 5.1 System Components
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Input    │────▶│ Adjustment Engine│────▶│  Output Layer   │
│   - HPI Data    │     │  - X-13ARIMA    │     │  - API          │
│   - Weather     │     │  - Regression    │     │  - Reports      │
│   - Demographics│     │  - Validation    │     │  - Storage      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### 5.2 Technology Stack
- **Programming Language**: Python 3.9+
- **Key Libraries**:
  - statsmodels (ARIMA implementation)
  - pandas (data manipulation)
  - numpy (numerical operations)
  - scikit-learn (regression models)
  - fastapi (API framework)
- **Database**: PostgreSQL with TimescaleDB extension
- **Cache**: Redis for frequent queries
- **Container**: Docker for deployment

### 5.3 Data Schema

#### Housing Price Index Table
```sql
CREATE TABLE housing_price_index (
    geography_id VARCHAR(10),
    geography_type VARCHAR(10), -- 'STATE' or 'MSA'
    period DATE,
    nsa_index DECIMAL(10,4),
    sa_index DECIMAL(10,4),
    seasonal_factor DECIMAL(10,6),
    num_transactions INTEGER,
    PRIMARY KEY (geography_id, period)
);
```

#### Weather Data Table
```sql
CREATE TABLE weather_data (
    geography_id VARCHAR(10),
    period DATE,
    temp_range DECIMAL(5,2),
    avg_temp DECIMAL(5,2),
    precipitation DECIMAL(6,2),
    PRIMARY KEY (geography_id, period)
);
```

## 6. API Specifications

### 6.1 Endpoints

#### Calculate Seasonal Adjustment
```
POST /api/v1/seasonal-adjustment
{
    "geography_id": "string",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "include_diagnostics": boolean
}

Response:
{
    "geography_id": "string",
    "adjustments": [
        {
            "period": "YYYY-Q#",
            "nsa_index": number,
            "sa_index": number,
            "seasonal_factor": number
        }
    ],
    "diagnostics": {...} // if requested
}
```

#### Batch Processing
```
POST /api/v1/seasonal-adjustment/batch
{
    "geography_ids": ["string"],
    "parameters": {
        "arima_spec": "auto" | "custom",
        "outlier_detection": boolean
    }
}

Response:
{
    "job_id": "string",
    "status": "processing",
    "estimated_completion": "timestamp"
}
```

#### Get Causation Analysis
```
GET /api/v1/causation-analysis/{geography_type}
Parameters:
- include_weather: boolean
- include_demographics: boolean
- quantiles: array[float]

Response:
{
    "model_type": "linear" | "quantile",
    "coefficients": {...},
    "r_squared": number,
    "significant_factors": [...]
}
```

## 7. Implementation Phases

### Phase 1: Core Engine (Weeks 1-4)
- Implement X-13ARIMA-SEATS algorithm
- Basic data input/output functionality
- Single geography processing

### Phase 2: Scalability (Weeks 5-6)
- Batch processing capabilities
- Parallel computation
- Performance optimization

### Phase 3: Causation Analysis (Weeks 7-8)
- Linear regression implementation
- Quantile regression
- Weather impact analysis

### Phase 4: API & Integration (Weeks 9-10)
- RESTful API development
- Documentation
- Client libraries

### Phase 5: Testing & Validation (Weeks 11-12)
- Comprehensive testing suite
- Validation against paper results
- Performance benchmarking

## 8. Testing Requirements

### 8.1 Unit Tests
- Algorithm correctness
- Data validation
- Error handling

### 8.2 Integration Tests
- End-to-end workflows
- API functionality
- Database operations

### 8.3 Performance Tests
- Load testing (100+ concurrent requests)
- Memory usage under stress
- Processing time benchmarks

### 8.4 Validation Tests
- Reproduce paper results for key markets
- Compare with existing FHFA adjustments
- Out-of-sample forecast accuracy

## 9. Success Criteria

### 9.1 Accuracy Metrics
- Model R² ≥ 0.4 for states, ≥ 0.3 for MSAs
- Forecast RMSE < 2% for one-year ahead
- Seasonal factor stability (CV < 0.1)

### 9.2 Performance Metrics
- API response time < 500ms (p95)
- Batch processing: 100 MSAs/5 minutes
- System uptime > 99.9%

### 9.3 Coverage Metrics
- Support all 50 states
- Support 400+ MSAs
- Handle 30+ years of historical data

## 10. Risks and Mitigations

### 10.1 Technical Risks
- **Risk**: X-13ARIMA convergence issues
- **Mitigation**: Implement fallback models, parameter bounds

### 10.2 Data Risks
- **Risk**: Missing or incomplete weather data
- **Mitigation**: Data imputation strategies, quality checks

### 10.3 Performance Risks
- **Risk**: Slow processing for large datasets
- **Mitigation**: Caching, parallel processing, GPU acceleration

## 11. Future Enhancements

### 11.1 Version 2.0 Features
- Machine learning augmentation
- Real-time adjustment updates
- Climate change scenario modeling
- International market support

### 11.2 Integration Opportunities
- Direct FHFA data feeds
- Commercial real estate indices
- Rental market adjustments
- Economic indicator correlations

## 12. Appendix

### 12.1 Glossary
- **NSA**: Non-Seasonally Adjusted
- **SA**: Seasonally Adjusted
- **HPI**: House Price Index
- **MSA**: Metropolitan Statistical Area
- **ARIMA**: AutoRegressive Integrated Moving Average
- **RegARIMA**: Regression with ARIMA errors

### 12.2 References
- Doerner, W.M. & Lin, W. (2022). "Applying Seasonal Adjustments to Housing Markets"
- X-13ARIMA-SEATS Reference Manual (Census Bureau)
- FHFA House Price Index Methodology