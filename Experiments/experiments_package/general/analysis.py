"""
General functions for creating plots and
statistics about experiments and model trainings.
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import denormalize, ProductIds
from .performance import (
    get_model_size_byte,
    get_non_trainable_params,
    get_trainable_params,
)
from .window_generator import WindowGenerator


def save_history_plot(history, where):
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss Development")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(where)
    print("    => Saved history ")
    plt.clf()


def create_results_table(
        performance,
        experiment_name,
        data_origin,
        feature_names,
        train_set_instances,
        valid_set_instances,
        test_set_instances,
        train_settings: Dict[str, str],
        models: Dict[str, Any],
):
    performance_stats = performance.create_performance_data()
    # Experiment Meta - Data
    performance_stats[[("Experiment", "Name")]] = experiment_name

    performance_stats[[("Experiment", "Data Origin")]] = data_origin

    performance_stats[[("Experiment", "Used Data Columns")]] = feature_names

    performance_stats[
        [("Experiment", "Data Instances (train/valid/test)")]
    ] = f"{train_set_instances}/{valid_set_instances}/{test_set_instances}"

    # The Settings for Learning Algorithm

    for setting, description in train_settings.items():
        performance_stats[[("Training Settings", setting)]] = description

    # The Individual Model Settings
    for name, model in models.items():
        performance_stats.at[
            name, ("Timing", "Latency (ms/observation)")
        ] = performance.get_timing(name)

        performance_stats.at[name, ("Model", "Summary")] = _get_summary_as_string(model)
        performance_stats.at[name, ("Model", "Name")] = name
        performance_stats.at[name, ("Model", "Size (KB)")] = get_model_size_byte(
            model
        ) / 1024
        performance_stats.at[
            name, ("Model", "Trainable Params")
        ] = get_trainable_params(model)
        performance_stats.at[
            name, ("Model", "Non-Trainable Params")
        ] = get_non_trainable_params(model)

    # Move Model name to first column
    first_column = performance_stats.pop(("Model", "Name"))
    performance_stats.insert(0, ("Model", "Name"), first_column)

    return performance_stats.reset_index(drop=True)


def save_predictions_plot(model, window_generator: WindowGenerator, where: str):
    samples = list(tf.keras.utils.timeseries_dataset_from_array(
        data=np.array(pd.concat([window_generator.train_df, window_generator.val_df, window_generator.test_df]),
                      dtype=np.float32),
        targets=None,
        sequence_length=window_generator.total_window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=1,
    ).map(window_generator.split_window))
    inputs = [x[0] for x in samples]

    label_columns = window_generator.label_columns

    if label_columns is None:
        label_columns = [ProductIds.BENS_LUNCHTIME.value]

    num_plots = len(label_columns)

    fig = plt.figure(figsize=(16 * num_plots, 4))
    data = pd.concat([window_generator.train_df, window_generator.val_df, window_generator.test_df])

    for i, label in enumerate(label_columns):
        label_index = window_generator.label_columns_indices[
            label] if window_generator.label_columns_indices is not None else window_generator.column_indices[label]

        ax = fig.add_subplot(num_plots, 1, i + 1)
        ax.set_title(f"Predictions and Labels for >>{label}<<")

        ax.scatter(
            window_generator.time_stamps,
            denormalize(data, window_generator.normalization_params)[label],
            edgecolors="k",
            label="Labels",
            c="#2ca02c",
            s=32,
        )
        print(model(inputs[0])[:, 0, label_index])
        ax.scatter(
            window_generator.time_stamps[window_generator.total_window_size - window_generator.label_width:],
            [denormalize(model(single_input)[:, 0, label_index],
                         window_generator.normalization_params,
                         labels=[label]) for single_input in inputs],
            marker="X",
            edgecolors="k",
            label="Predictions",
            c="#ff7f0e",
            s=32,
        )

        ax.axvline(x=window_generator.time_stamps[len(window_generator.train_df)], color='black',
                   label='End of training data')
        ax.axvline(x=window_generator.time_stamps[len(window_generator.val_df) + len(window_generator.train_df)],
                   color='orange',
                   label='end of validation data')
        ax.legend()

    fig.savefig(where)
    print(" => Saved Predictions Plot")


def _get_summary_as_string(model) -> str:
    summary_parts = []
    model.summary(print_fn=summary_parts.append)
    summary = "\n".join(summary_parts)
    return summary
