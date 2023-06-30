import tensorflow as tf


class Baseline(tf.keras.Model):
    def __init__(self, label_index, label_width, repeat_last=False):
        super().__init__()
        self.label_index = slice(None) if label_index is None else label_index
        self.forecasting_width = label_width
        self.repeat_last = repeat_last

    def call(self, inputs, training=None, mask=None):
        if self.repeat_last:
            result = tf.concat(
                [
                    inputs[:, -1:, self.label_index]
                    for i in range(self.forecasting_width)
                ],
                -2,  # concatenate the time steps
            )
        else:
            result = inputs[:, -self.forecasting_width :, self.label_index]

        if len(result.shape) == 2:
            return result[:, :, tf.newaxis]

        return result
