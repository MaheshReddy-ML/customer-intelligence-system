from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def perform_kmeans_clustering(customer_df, n_clusters=4):
    """Cluster customer features with K-Means.

    Args:
        customer_df: Customer-level feature table.
        n_clusters: Number of clusters to create.

    Returns:
        Tuple containing the input DataFrame with Cluster labels and the model.
    """
    features = customer_df.drop(columns=["CustomerID"])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    clusters = kmeans.fit_predict(scaled_features)
    customer_df["Cluster"] = clusters

    return customer_df, kmeans
