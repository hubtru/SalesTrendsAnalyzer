import tensorflow as tf


def MultiDense(label_width, num_labels, window_width, num_features):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(window_width, num_features)),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(16, activation="relu"),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(
                label_width * num_labels, kernel_initializer=tf.initializers.zeros()
            ),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([label_width, num_labels]),
        ]
    )
