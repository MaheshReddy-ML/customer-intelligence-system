import pandas as pd

from src.clustering.kmeans import perform_kmeans_clustering


def test_perform_kmeans_clustering_adds_cluster_column():
    """Validate that K-Means assigns one cluster label per customer row."""
    customer_df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4],
            "Recency": [1, 2, 90, 95],
            "Frequency": [5.0, 4.8, 1.0, 1.2],
            "MonetaryValue": [8.0, 7.5, 2.0, 2.2],
            "TotalQuantity": [6.0, 5.8, 1.0, 1.1],
            "AvgUnitPrice": [3.0, 3.2, 5.0, 5.1],
        }
    )

    clustered_df, model = perform_kmeans_clustering(customer_df, n_clusters=2)

    assert "Cluster" in clustered_df.columns
    assert len(clustered_df["Cluster"]) == len(customer_df)
    assert model.n_clusters == 2
