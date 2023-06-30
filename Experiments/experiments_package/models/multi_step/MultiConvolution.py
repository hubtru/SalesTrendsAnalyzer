import tensorflow as tf


def MultiConvolution(conv_width, label_width, num_labels):
    return tf.keras.Sequential(
        [
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation="relu", kernel_size=(conv_width,)),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(
                label_width * num_labels, kernel_initializer=tf.initializers.zeros()
            ),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([label_width, num_labels]),
        ]
    )
