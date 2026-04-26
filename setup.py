from setuptools import setup, find_packages

setup(
    name="wildlife_trainer",
    version="0.6",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",z
        "joblib",
        "matplotlib",
        "gcsfs"
    ],
)