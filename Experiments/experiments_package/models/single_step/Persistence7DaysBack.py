import tensorflow as tf


class Persistence7DaysBack(tf.keras.Model):
    def __init__(self, label_index, label_width):
        super().__init__()
        self.label_index = slice(None) if label_index is None else label_index
        self.forecasting_width = label_width

    def call(self, inputs, training=None, mask=None):
        # Output the output of 6 days ago for tomorrow (eq. same day, last week)
        if inputs.shape[1] < self.forecasting_width + 6:
            raise ValueError("The Input width has to be at least long enough for accessing last week")

        result = inputs[:, - (self.forecasting_width + 6): -6, self.label_index]

        if len(result.shape) == 2:
            return result[:, :, tf.newaxis]

        return result
