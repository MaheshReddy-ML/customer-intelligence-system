from pathlib import Path

import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path("outputs/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_neural_network(customer_df):
    """Train a neural customer segmentation model and persist artifacts.

    Args:
        customer_df: Customer-level feature table containing Cluster labels.

    Returns:
        Tuple containing the trained model, training history, and scaled features.
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
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),
        tf.keras.layers.Dense(
            16,
            activation="relu",
            name="customer_embedding"
        ),
        tf.keras.layers.Dense(
            len(set(y)),
            activation="softmax"
        )
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=64,
        verbose=1
    )
    model.save(
        MODEL_DIR / "customer_segment_nn.keras"
    )

    print("\nModel saved successfully.")

    return model, history, X
