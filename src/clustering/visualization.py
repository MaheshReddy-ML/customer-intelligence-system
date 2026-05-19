from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_customer_clusters(customer_df):
    """Create and save a PCA scatter plot of customer clusters.

    Args:
        customer_df: Customer-level feature table containing Cluster labels.

    Returns:
        None.
    """
    features = customer_df.drop(
        columns=["CustomerID", "Cluster"],
        errors="ignore"
    )
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=customer_df["Cluster"]
    )
    plt.title("Customer Segmentation using K-Means")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster")

    plot_dir = Path(
        "outputs/plots"
    )
    plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    plt.savefig(
        plot_dir / "customer_clusters_pca.png"
    )
    plt.close()
