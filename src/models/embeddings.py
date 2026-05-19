import numpy as np
import tensorflow as tf


def extract_embedding_model(model):
    """Create a model that outputs the learned customer embedding layer.

    Args:
        model: Trained TensorFlow model containing a customer_embedding layer.

    Returns:
        TensorFlow model that emits customer embeddings.
    """
    embedding_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(
            "customer_embedding"
        ).output
    )

    return embedding_model


def generate_customer_embeddings(
    embedding_model,
    scaled_features
):
    """Generate customer embeddings from scaled feature data.

    Args:
        embedding_model: TensorFlow embedding extraction model.
        scaled_features: Scaled customer feature matrix.

    Returns:
        NumPy array containing customer embeddings.
    """
    embeddings = embedding_model.predict(
        scaled_features
    )

    return np.array(embeddings)
