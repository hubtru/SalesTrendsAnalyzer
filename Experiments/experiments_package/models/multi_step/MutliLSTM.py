import tensorflow as tf


def MultiLSTM(label_width, num_labels):
    return tf.keras.Sequential(
        [
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(
                label_width * num_labels, kernel_initializer=tf.initializers.zeros()
            ),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([label_width, num_labels]),
        ]
    )
