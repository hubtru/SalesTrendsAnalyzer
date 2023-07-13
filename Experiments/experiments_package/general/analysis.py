"""
General functions for creating plots and
statistics about experiments and model trainings.
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ProductIds, denormalize_list
from .data import get_all_product_ids, get_no_sushi_in_product
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
        data_origin: str,
        window_generator: WindowGenerator,
        train_settings: Dict[str, str],
        models: Dict[str, Any],
):
    feature_names = str(list(window_generator.train_df.columns.values)),
    train_set_instances = window_generator.train_samples,
    valid_set_instances = window_generator.val_samples,
    test_set_instances = window_generator.test_samples,

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

        wasted, not_enough = get_sushi_stats(model, window_generator)
        performance_stats.at[name, ("Performance", "Avg. Sushi wasted /day")] = wasted
        performance_stats.at[
            name, ("Performance", "Avg. sushi not enough /day")] = not_enough

    # Move Model name to first column
    first_column = performance_stats.pop(("Model", "Name"))
    performance_stats.insert(0, ("Model", "Name"), first_column)

    return performance_stats.reset_index(drop=True)


def get_sushi_stats(model, window_generator: WindowGenerator):
    product_ids = get_all_product_ids()
    product_ids = [p_id for p_id in product_ids if p_id in window_generator.column_indices.keys()]
    if window_generator.label_columns is not None:
        product_ids = [p_id for p_id in product_ids if p_id in window_generator.label_columns]

    total_days = 0
    wasted = 0
    not_enough = 0
    for p_id in product_ids:
        pred_time, predictions = get_model_predictions_sequentially_with_time(model, window_generator, label=p_id)
        label_time, labels = window_generator.get_feature_sequentially_with_time(p_id)
        combined_times = list(set(pred_time) & set(label_time))
        beginning = min(combined_times)
        end = max(combined_times)
        pred_time = pd.Series(sorted(list(pred_time)))
        label_time = pd.Series(sorted(list(label_time)))

        pred_slice = slice(pred_time[pred_time == beginning].index[0], list(pred_time[pred_time == end].index)[-1])
        label_slice = slice(label_time[label_time == beginning].index[0], list(label_time[label_time == end].index)[-1])

        # round to integer, as we cannot produce "half sets"
        predictions = np.rint(np.array(predictions[pred_slice])).reshape((-1,))
        labels = np.rint(np.array(labels[label_slice])).reshape((-1,))

        sushi_in_product = get_no_sushi_in_product(p_id)

        wasted += np.sum(np.maximum(predictions - labels, 0)) * sushi_in_product
        not_enough += np.sum(np.maximum(labels - predictions, 0)) * sushi_in_product
        total_days += len(labels)

    if total_days == 0:
        return 0, 0

    return int(np.rint(wasted / total_days)), int(np.rint(not_enough / total_days))


def get_model_predictions_sequentially_with_time(model, window_generator: WindowGenerator, label: str):
    inputs = window_generator.get_all_inputs_sequentially()

    label_index = window_generator.label_columns_indices[label]

    # see diagram above on where time starts and ends for the labels.
    # end of the time has to be considered as only the first of the results will get used
    last_time_index = - window_generator.label_width + 1
    time = window_generator.time_stamps[
           window_generator.total_window_size - window_generator.label_width: last_time_index if last_time_index != 0 else None]
    # Only pick the first of the labels predicted
    predictions = denormalize_list(model(inputs)[:, 0, label_index],
                                   window_generator.normalization_params,
                                   label)
    return time, predictions


def save_predictions_plot(model, window_generator: WindowGenerator, where: str):
    label_columns = window_generator.label_columns

    if label_columns is None:
        # If all columns are predicted, only this sushi type is shown
        label_columns = [ProductIds.BENS_LUNCHTIME.value]

    num_plots = len(label_columns)
    fig = plt.figure(figsize=(16 * num_plots, 4))

    for i, label in enumerate(label_columns):
        ax = fig.add_subplot(num_plots, 1, i + 1)
        ax.set_title(f"Predictions and Labels for >>{label}<<")

        x, y = window_generator.get_feature_sequentially_with_time(label)
        ax.scatter(
            x, y,
            edgecolors="k",
            label="Labels",
            c="#2ca02c",
            s=32
        )
        x, y = get_model_predictions_sequentially_with_time(model, window_generator, label)
        ax.scatter(
            x, y,
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
