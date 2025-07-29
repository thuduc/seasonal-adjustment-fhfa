from setuptools import setup, find_packages

setup(
    name="seasonal-adjustment-fhfa",
    version="1.0.0",
    description="Housing Price Seasonal Adjustment Model using X-13ARIMA-SEATS",
    author="FHFA Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "statsmodels>=0.12.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
        "joblib>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0"
        ]
    },
    python_requires=">=3.8",
)