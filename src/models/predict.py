from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from src.hierarchy.label_generator import (
    generate_cluster_labels
)

from src.recommendation.recommender import (
    generate_customer_recommendation
)


MODEL_DIR = Path("outputs/models")


def load_artifacts():
    """Load trained model and scaler artifacts from disk.

    Args:
        None.

    Returns:
        Tuple containing the TensorFlow model and fitted scaler.
    """
    model = tf.keras.models.load_model(
        MODEL_DIR / "customer_segment_nn.keras"
    )
    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )

    return model, scaler


def predict_customer_segment(customer_features):
    """Predict the customer segment and recommendation for one customer.

    Args:
        customer_features: Ordered feature values for one customer.

    Returns:
        Dictionary containing cluster ID, label, recommendation, and probabilities.
    """
    model, scaler = load_artifacts()
    cluster_labels = generate_cluster_labels()
    features = np.array(customer_features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    predicted_cluster = int(
        np.argmax(predictions)
    )
    cluster_name = cluster_labels.get(
        predicted_cluster,
        "Unknown Segment"
    )
    recommendation = (
        generate_customer_recommendation(
            cluster_name
        )
    )

    return {

        "cluster_id": predicted_cluster,

        "cluster_name": cluster_name,

        "summary": recommendation["summary"],

        "business_action": (
            recommendation["business_action"]
        ),

        "probabilities": predictions.tolist()
    }


if __name__ == "__main__":
    sample_customer = [
        20,
        4.5,
        7.2,
        6.8,
        3.4
    ]
    result = predict_customer_segment(
        sample_customer
    )

    print("\nPrediction Result:\n")
    print(result)
