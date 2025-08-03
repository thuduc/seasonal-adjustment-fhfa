from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fhfa-seasonal-adjustment",
    version="0.1.0",
    author="FHFA Seasonal Adjustment Team",
    description="FHFA House Price Index Seasonal Adjustment Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fhfa/seasonal-adjustment",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "statsmodels>=0.14.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "linearmodels>=5.3",
        "pydantic>=2.3.0",
        "click>=8.1.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "hypothesis>=6.82.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.17.0",
        ],
        "performance": [
            "numba>=0.58.0",
            "dask>=2023.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fhfa-seasonal-adjust=scripts.run_seasonal_adjustment:main",
        ],
    },
)