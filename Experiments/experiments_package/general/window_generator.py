"""
The wrapper for the dataset that we use. It generates windows
of the timeseries data to learn from.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .config import ProductIds
from .config import denormalize


class WindowGenerator:
    def __init__(
            self,
            input_width,
            label_width,
            shift,
            train_df,
            val_df,
            test_df,
            normalization_params,
            time_stamps,
            label_columns=None,

    ):
        """
        This drawing indicates the meaning of the attributes:

        ------------------------Time Dimension ----------------->
                                               |
        <----------total window size---------->|
        <------input width------><---shift---->|
                    <------label width-------->|
                                               |

        """
        if label_width > shift:
            raise ValueError("unnecessary labels included in data")

        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.normalization_params = normalization_params
        self.time_stamps = time_stamps

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        else:
            self.label_columns_indices = None
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self._example = None

    def __repr__(self):
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features):
        """
        splits the incoming data: (batch_size, total_window_size, all_features)
        into:
        inputs: (batch_size, input_width, all_features)
        labels: (batch_size, label_width, #label_columns)

        while the timesteps are according to the parameter meaning (see description of class)
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(
            self,
            model=None,
            plot_col=ProductIds.BENS_LUNCHTIME.value,
            max_subplots=3,
    ):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]

        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(plot_col)
            plt.plot(
                self.input_indices,
                denormalize(inputs[n, :, plot_col_index], self.normalization_params)[plot_col],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices,
                denormalize(labels[n, :, label_col_index], self.normalization_params)[plot_col],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )

            if model is not None:
                plt.scatter(
                    self.label_indices,
                    denormalize(model(inputs)[n, :, label_col_index], self.normalization_params)[plot_col],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            if n == 0:
                plt.legend()

            plt.xlabel("Time [day]")

    def make_dataset(self, data):
        return tf.keras.utils.timeseries_dataset_from_array(
            data=np.array(data, dtype=np.float32),
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,
        ).map(self.split_window)

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    @staticmethod
    def _count_samples(of):
        return sum([x[0].shape[0] for x in list(of)])

    @property
    def train_samples(self):
        return self._count_samples(self.train)

    @property
    def test_samples(self):
        return self._count_samples(self.test)

    @property
    def val_samples(self):
        return self._count_samples(self.val)

    @property
    def total_samples(self):
        return self.val_samples + self.test_samples + self.train_samples
