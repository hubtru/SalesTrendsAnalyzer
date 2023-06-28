import tensorflow as tf


def Convolutional(window_width, feature_size, label_width):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(
                input_shape=(window_width, feature_size),
                filters=32,
                kernel_size=(2,),
                activation="relu",
            ),
            tf.keras.layers.Flatten(
                input_shape=(window_width - 2 + 1, 32),  # window - kernel + 1
            ),
            tf.keras.layers.Dense(units=32, activation="relu"),
            tf.keras.layers.Dense(units=label_width),
            tf.keras.layers.Reshape(target_shape=(label_width, 1)),
        ]
    )
