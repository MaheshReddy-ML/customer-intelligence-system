from setuptools import find_namespace_packages, setup


setup(
    name="customer-intelligence-system",
    version="0.1.0",
    description=(
        "Customer segmentation and recommendation system using clustering, "
        "embeddings, and neural models."
    ),
    author="Mahesh Reddy",
    packages=find_namespace_packages(include=["src", "src.*"]),
    python_requires=">=3.11",
    install_requires=[
        "joblib>=1.4",
        "matplotlib>=3.8",
        "numpy>=1.26",
        "pandas>=2.2",
        "pyyaml>=6.0",
        "scikit-learn>=1.4",
        "seaborn>=0.13",
        "tensorflow>=2.16",
        "ucimlrepo>=0.0.7",
    ],
)
