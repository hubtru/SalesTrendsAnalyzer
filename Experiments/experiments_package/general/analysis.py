"""
General functions for creating plots and
statistics about experiments and model trainings.
"""

from typing import Any, Dict

import matplotlib.pyplot as plt

from .performance import (
    get_model_size_byte,
    get_non_trainable_params,
    get_trainable_params,
)


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
    ## Erperiment Meta - Data
    performance_stats[[("Experiment", "Name")]] = experiment_name

    performance_stats[[("Experiment", "Data Origin")]] = data_origin

    performance_stats[[("Experiment", "Used Data Columns")]] = feature_names

    performance_stats[
        [("Experiment", "Data Instances (train/valid/test)")]
    ] = f"{train_set_instances}/{valid_set_instances}/{test_set_instances}"

    ## The Settings for Learning Algorithm
    for setting, description in train_settings.items():
        performance_stats[[("Training Settings", setting)]] = description

    ## The Individual Model Settings
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


def _get_summary_as_string(model) -> str:
    summary_parts = []
    model.summary(print_fn=summary_parts.append)
    summary = "\n".join(summary_parts)
    return summary
