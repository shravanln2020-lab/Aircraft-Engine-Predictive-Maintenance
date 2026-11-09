"""
setup.py — AEPM Package Installation Configuration

Aircraft Engine Predictive Maintenance using Machine Learning
NASA C-MAPSS FD001 · Random Forest Regressor · Streamlit Dashboard

Usage:
    pip install -e .            # editable install (development)
    pip install .               # standard install
"""

from pathlib import Path
from setuptools import find_packages, setup

# Read long description from project README
README_PATH = Path(__file__).resolve().parent / "README.md"
long_description = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""

setup(
    # ── Package Identity ───────────────────────────────────────────────────────
    name="aepm",
    version="1.0.0",
    description=(
        "Aircraft Engine Predictive Maintenance using Machine Learning — "
        "RUL Prediction via Random Forest on NASA C-MAPSS FD001"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AEPM Project Team",
    python_requires=">=3.11",

    # ── Package Discovery ──────────────────────────────────────────────────────
    packages=find_packages(exclude=["tests*", "notebooks*", "reports*"]),

    # ── Runtime Dependencies ───────────────────────────────────────────────────
    install_requires=[
        "numpy==2.5.1",
        "pandas==3.0.5",
        "scikit-learn==1.9.0",
        "scipy==1.18.0",
        "joblib==1.5.3",
        "streamlit==1.61.0",
        "plotly==6.9.0",
    ],

    # ── Optional / Development Dependencies ───────────────────────────────────
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=6.0.0",
            "flake8>=7.0.0",
        ],
    },

    # ── Classifiers ───────────────────────────────────────────────────────────
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.14",
        "Operating System :: OS Independent",
    ],

    # ── Package Data ──────────────────────────────────────────────────────────
    include_package_data=True,
    package_data={
        "": ["*.toml", "*.json", "*.md"],
    },

    # ── Entry Points ──────────────────────────────────────────────────────────
    # Dashboard launch command:
    #     python -m streamlit run app/dashboard.py
)
