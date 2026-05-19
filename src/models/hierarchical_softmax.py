import tensorflow as tf


class HierarchicalSoftmax(tf.keras.layers.Layer):
    """TensorFlow layer that applies a learned softmax projection."""

    def __init__(self, units):
        """Initialize the hierarchical softmax layer.

        Args:
            units: Number of output units.
        """
        super().__init__()
        self.units = units

    def build(self, input_shape):
        """Create trainable weights for the layer.

        Args:
            input_shape: Shape of incoming tensors.

        Returns:
            None.
        """
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        """Apply softmax projection to input tensors.

        Args:
            inputs: Input tensor.

        Returns:
            Softmax-normalized output tensor.
        """
        logits = tf.matmul(inputs, self.w) + self.b

        return tf.nn.softmax(logits)
