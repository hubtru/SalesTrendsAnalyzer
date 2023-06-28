import tensorflow as tf


def RNN(window_width, feature_size, label_width):
    return tf.keras.models.Sequential(
        [
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(
                32, return_sequences=True, input_shape=(window_width, feature_size)
            ),
            tf.keras.layers.Flatten(input_shape=(window_width, 32)),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=label_width),
            tf.keras.layers.Reshape(target_shape=(label_width, 1)),
        ]
    )
