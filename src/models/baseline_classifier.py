from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def train_baseline_classifier(customer_df):
    """Train a Random Forest classifier to predict customer clusters.

    Args:
        customer_df: Customer-level feature table containing Cluster labels.

    Returns:
        Trained RandomForestClassifier model.
    """
    cluster_counts = customer_df["Cluster"].value_counts()
    valid_clusters = cluster_counts[cluster_counts > 1].index
    customer_df = customer_df[
        customer_df["Cluster"].isin(valid_clusters)
    ]
    X = customer_df.drop(
        columns=["CustomerID", "Cluster"],
        errors="ignore"
    )
    y = customer_df["Cluster"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)

    print("\nBaseline Classifier Report:\n")
    print(report)

    return model
