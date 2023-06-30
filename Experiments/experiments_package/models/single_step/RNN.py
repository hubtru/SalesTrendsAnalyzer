import tensorflow as tf


def RNN(window_width, feature_size, label_width, num_labels=1):
    return tf.keras.models.Sequential(
        [
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(
                16, return_sequences=True, input_shape=(window_width, feature_size)
            ),
            tf.keras.layers.Flatten(input_shape=(window_width, 16)),
            tf.keras.layers.Dropout(rate=0.9),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=label_width * num_labels),
            tf.keras.layers.Reshape(target_shape=(label_width, num_labels)),
        ]
    )
