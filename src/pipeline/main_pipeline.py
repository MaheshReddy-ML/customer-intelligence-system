from src.data.loader import fetch_online_retail_data
from src.data.preprocessing import preprocess_data
from src.data.feature_engineering import create_customer_features

from src.models.baseline_classifier import (
    train_baseline_classifier
)

from src.models.train import train_neural_network

from src.models.embeddings import (
    extract_embedding_model,
    generate_customer_embeddings
)

from src.clustering.kmeans import (
    perform_kmeans_clustering
)

from src.clustering.embedding_clustering import (
    cluster_embeddings
)

from src.clustering.embedding_visualization import (
    visualize_embedding_clusters
)

from src.clustering.evaluation import (
    analyze_clusters,
    evaluate_kmeans
)

from src.clustering.visualization import (
    plot_customer_clusters
)


RUN_CLUSTERING = True
RUN_VISUALIZATION = False
RUN_BASELINE = False
RUN_NN = False
RUN_EMBEDDINGS = True
RUN_EMBEDDING_CLUSTERING = True
RUN_EMBEDDING_VISUALIZATION = True


def main():
    """Run the customer intelligence training and analysis pipeline.

    Args:
        None.

    Returns:
        None.
    """
    df = fetch_online_retail_data()

    print("Raw Shape:", df.shape)

    df = preprocess_data(df)

    print("Processed Shape:", df.shape)

    customer_df = create_customer_features(df)

    print("\nCustomer Features Shape:")
    print(customer_df.shape)

    if RUN_CLUSTERING:
        evaluate_kmeans(customer_df)
        customer_df, kmeans_model = (
            perform_kmeans_clustering(customer_df)
        )
        cluster_summary = analyze_clusters(
            customer_df
        )

        print("\nCluster Summary:")
        print(cluster_summary)

    if RUN_VISUALIZATION:
        plot_customer_clusters(customer_df)

    if RUN_BASELINE:
        train_baseline_classifier(customer_df)

    if RUN_NN or RUN_EMBEDDINGS:
        nn_model, history, scaled_features = (
            train_neural_network(customer_df)
        )

    if RUN_EMBEDDINGS:
        embedding_model = extract_embedding_model(
            nn_model
        )
        customer_embeddings = (
            generate_customer_embeddings(
                embedding_model,
                scaled_features
            )
        )

        print("\nCustomer Embeddings Shape:")
        print(customer_embeddings.shape)

    if RUN_EMBEDDING_CLUSTERING:
        embedding_labels = cluster_embeddings(
            customer_embeddings
        )

        print("\nEmbedding Cluster Labels Shape:")
        print(embedding_labels.shape)

    if RUN_EMBEDDING_VISUALIZATION:
        visualize_embedding_clusters(
            customer_embeddings,
            embedding_labels
        )


if __name__ == "__main__":
    main()
