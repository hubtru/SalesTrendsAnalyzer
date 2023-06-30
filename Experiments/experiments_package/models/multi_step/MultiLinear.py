import tensorflow as tf


def MultiLinear(label_width, num_labels):
    return tf.keras.Sequential(
        [
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(
                label_width * num_labels, kernel_initializer=tf.initializers.zeros()
            ),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([label_width, num_labels]),
        ]
    )
