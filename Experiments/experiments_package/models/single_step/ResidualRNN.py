import tensorflow as tf
from .RNN import RNN


class ResidualRNN(tf.keras.Model):
    def __init__(self, window_width, feature_size, label_width, num_labels):
        super().__init__()
        self.model = RNN(
            window_width=window_width,
            feature_size=feature_size,
            label_width=label_width,
            num_labels=num_labels,
        )

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta
