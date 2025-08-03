"""Constants for FHFA seasonal adjustment models"""

# Model coefficients from PRD
TEMPERATURE_COEFFICIENTS = {
    "state": {
        "q1": 0.011,
        "q2": 0.014,
        "q3": 0.027,
        "q4": -0.018
    },
    "msa": {
        "q1": 0.018,
        "q2": 0.006,
        "q3": 0.006,
        "q4": -0.005
    }
}

TIME_TREND_COEFFICIENTS = {
    "1991-1998": 0.021,
    "1999-2007": 0.126,
    "2008-2011": 0.071,
    "2012-2020": 0.015
}

# Processing time benchmarks (minutes)
PROCESSING_TIME_TARGETS = {
    "national": 15,
    "state": 90,
    "msa_top100": 150,
    "all_msa": 1440  # 24 hours
}

# Statistical test thresholds
LJUNG_BOX_PVALUE_THRESHOLD = 0.05
NORMALITY_PVALUE_THRESHOLD = 0.05

# Data quality thresholds
MIN_SERIES_LENGTH = 20
MAX_MISSING_RATE = 0.2
OUTLIER_STD_THRESHOLD = 3.0