import tensorflow as tf


class Baseline(tf.keras.Model):
    def __init__(self, label_index, forecasting_width):
        super().__init__()
        self.label_index = label_index
        self.forecasting_width = forecasting_width

    def call(self, inputs):
        result = inputs[:, -self.forecasting_width :, self.label_index]
        return result[:, :, tf.newaxis]
