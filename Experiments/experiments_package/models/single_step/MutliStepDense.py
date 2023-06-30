import tensorflow as tf


def MultiStepDense(window_width, feature_size, label_width):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(window_width, feature_size)),
            tf.keras.layers.Dense(units=32, activation="relu"),
            tf.keras.layers.Dropout(rate=0.8),
            tf.keras.layers.Dense(units=32, activation="relu"),
            tf.keras.layers.Dropout(rate=0.8),
            tf.keras.layers.Dense(units=label_width),
            tf.keras.layers.Reshape(target_shape=(label_width, 1)),
        ]
    )
