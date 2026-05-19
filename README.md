# Customer Intelligence System

> End-to-end customer segmentation, embedding analysis, and recommendation intelligence for retail behavior data.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

Customer Intelligence System is a machine learning project that transforms raw retail transactions into meaningful customer segments, learned embeddings, evaluation metrics, visualizations, and business recommendations. It combines traditional clustering, supervised baselines, neural representation learning, and rule-based recommendation outputs in one clean pipeline.

## What It Does

- Downloads or loads the Online Retail dataset from the UCI Machine Learning Repository.
- Cleans transaction data and builds customer-level behavioral features.
- Creates RFM-style customer signals such as recency, frequency, monetary value, quantity, and average unit price.
- Evaluates K-Means clustering with inertia and silhouette scores.
- Trains a baseline Random Forest classifier for customer segment prediction.
- Trains a neural network with a dedicated customer embedding layer.
- Clusters learned embeddings for deeper behavioral grouping.
- Generates t-SNE and PCA-based visual outputs.
- Maps customer clusters to readable business labels and recommendations.

## Tech Stack

| Area | Tools |
| --- | --- |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn |
| Deep learning | TensorFlow, Keras |
| Visualization | Matplotlib, Seaborn |
| Data source | UCI ML Repository |
| Packaging | pyproject.toml, setuptools |

## Project Structure

```text
customer-intelligence-system/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── metrics/
│   ├── models/
│   └── plots/
├── src/
│   ├── clustering/
│   ├── data/
│   ├── hierarchy/
│   ├── models/
│   ├── pipeline/
│   ├── recommendation/
│   └── utils/
├── tests/
├── requirements.txt
├── pyproject.toml
└── setup.py
```

## Quick Start

Clone the repository:

```bash
git clone https://github.com/MaheshReddy-ML/customer-intelligence-system
cd customer-intelligence-system
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the project in editable mode:

```bash
pip install -e .
```

## Run The Pipeline

Run the complete customer intelligence workflow:

```bash
python3 -m src.pipeline.main_pipeline
```

The pipeline handles data loading, preprocessing, feature engineering, clustering, model training, embedding extraction, and visualization based on the execution flags in `src/pipeline/main_pipeline.py`.

## Run Prediction

After training has created model artifacts in `outputs/models/`, run:

```bash
python3 -m src.models.predict
```

Prediction returns a structured output with:

- Predicted cluster ID
- Human-readable segment name
- Segment summary
- Recommended business action
- Model probability scores

## Generated Outputs

| Output | Description |
| --- | --- |
| `outputs/metrics/kmeans_metrics.csv` | K-Means inertia and silhouette scores |
| `outputs/metrics/cluster_summary.csv` | Average customer features by cluster |
| `outputs/plots/embedding_tsne.png` | t-SNE visualization of learned embeddings |
| `outputs/plots/customer_clusters_pca.png` | PCA visualization of customer clusters |
| `outputs/models/customer_segment_nn.keras` | Trained neural segmentation model |
| `outputs/models/scaler.pkl` | Fitted feature scaler |

Generated data, models, plots, metrics, virtual environments, and cache files are ignored by Git to keep the repository clean and lightweight.

## Typical Workflow

```text
Raw transactions
        |
        v
Preprocessing
        |
        v
Customer feature engineering
        |
        v
K-Means segmentation
        |
        v
Baseline and neural model training
        |
        v
Customer embedding extraction
        |
        v
Embedding clustering and visualization
        |
        v
Segment labels and business recommendations
```

## Configuration

Project-level settings live in:

```text
configs/config.yaml
```

The config includes dataset IDs, clustering parameters, training settings, and output locations.

## Notes

- Use Python 3.11 or newer.
- Run commands from the project root.
- Large generated artifacts are intentionally excluded from Git.
- If TensorFlow installation varies by machine, install the TensorFlow package recommended for your operating system and hardware.

## Author

**Mahesh Reddy**  
Machine Learning and Customer Intelligence
