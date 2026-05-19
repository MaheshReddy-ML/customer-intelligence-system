from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("outputs/metrics")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def analyze_clusters(customer_df: pd.DataFrame):
    """Summarize feature averages for each customer cluster.

    Args:
        customer_df: Customer-level feature table containing a Cluster column.

    Returns:
        DataFrame with mean feature values by cluster.
    """
    cluster_summary = customer_df.groupby("Cluster").mean()
    cluster_summary.to_csv(
        OUTPUT_DIR / "cluster_summary.csv"
    )

    return cluster_summary


def evaluate_kmeans(
    customer_df: pd.DataFrame,
    max_k: int = 10
):
    """Evaluate K-Means cluster counts with inertia and silhouette scores.

    Args:
        customer_df: Customer-level feature table.
        max_k: Largest number of clusters to evaluate.

    Returns:
        Tuple containing inertia values and silhouette scores.
    """
    features = customer_df.drop(
        columns=["CustomerID", "Cluster"],
        errors="ignore"
    )
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        features
    )
    inertias = []
    silhouette_scores = []

    print("\nEvaluating K-Means Clustering...\n")

    for k in range(2, max_k + 1):

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(
            scaled_features
        )
        inertias.append(
            kmeans.inertia_
        )
        score = silhouette_score(
            scaled_features,
            labels
        )
        silhouette_scores.append(
            score
        )
        print(
            f"K={k} | "
            f"Inertia={kmeans.inertia_:.2f} | "
            f"Silhouette Score={score:.4f}"
        )

    metrics_df = pd.DataFrame({
        "K": range(2, max_k + 1),
        "Inertia": inertias,
        "SilhouetteScore": silhouette_scores
    })
    metrics_df.to_csv(
        OUTPUT_DIR / "kmeans_metrics.csv",
        index=False
    )

    print(
        "\nK-Means metrics saved "
        "successfully."
    )

    return inertias, silhouette_scores
