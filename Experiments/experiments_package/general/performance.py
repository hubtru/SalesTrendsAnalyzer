"""
A helper class that tracks performances for experiments.
"""

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from .data import WindowGenerator


@dataclass
class TrackedPerformances:
    valid = {}
    test = {}
    train = {}
    metrics_names = {}
    timing = {}


class Performances:
    def __init__(self, window_generator: WindowGenerator):
        self.window_generator = window_generator

        self.performances = TrackedPerformances()

        self.model_names = []

    def register_performance(self, name, model):
        self.model_names.append(name)

        print("  ... measuring performance ...")
        before = time.process_time()
        self.performances.train[name] = model.evaluate(
            self.window_generator.train, verbose=0
        )
        self.performances.valid[name] = model.evaluate(
            self.window_generator.val, verbose=0
        )
        self.performances.test[name] = model.evaluate(
            self.window_generator.test, verbose=0
        )
        after = time.process_time()

        self.performances.timing[name] = (
                (after - before) * 1000 / self.window_generator.total_samples
        )

        self.performances.metrics_names[name] = model.metrics_names

    def create_performance_data(self):
        index = pd.MultiIndex.from_product(
            [
                list(
                    set(
                        loss
                        for losses in self.performances.metrics_names.values()
                        for loss in losses
                    )
                ),
                ["Training", "Validation", "Test"],
            ]
        )

        stats = pd.DataFrame(index=index, columns=self.model_names)

        for model_name in self.model_names:
            for index, loss in enumerate(self.performances.metrics_names[model_name]):
                stats.at[(loss, "Training"), model_name] = self.performances.train[
                    model_name
                ][index]

                stats.at[(loss, "Validation"), model_name] = self.performances.valid[
                    model_name
                ][index]

                stats.at[(loss, "Test"), model_name] = self.performances.test[
                    model_name
                ][index]
        return stats.T

    def save(self, where):
        self.create_performance_data().to_csv(where)
        print(" => Statistics Saved.")

    def save_plot(self, where):
        loss_name = "loss"

        x_pos = np.arange(len(self.model_names))
        width = 0.3
        validation_perf = [
            self.performances.valid[model_name][
                self.performances.metrics_names[model_name].index(loss_name)
            ]
            for model_name in self.model_names
        ]
        test_perf = [
            self.performances.test[model_name][
                self.performances.metrics_names[model_name].index(loss_name)
            ]
            for model_name in self.model_names
        ]
        train_perf = [
            self.performances.train[model_name][
                self.performances.metrics_names[model_name].index(loss_name)
            ]
            for model_name in self.model_names
        ]

        plt.ylabel("Loss")
        plt.bar(x_pos - 0.3, train_perf, width, label="Train")
        plt.bar(x_pos, validation_perf, width, label="Validation")
        plt.bar(x_pos + 0.3, test_perf, width, label="Test")
        plt.xticks(ticks=x_pos, labels=self.model_names, rotation=45)
        plt.legend()
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(where)
        print(" => Performance Plot Saved.")
        plt.clf()

    def get_timing(self, model_name):
        return self.performances.timing[model_name]


def get_model_size_byte(model):
    return _get_model_memory_usage(model)


def get_trainable_params(model):
    return np.sum([tf.size(w_matrix).numpy() for w_matrix in model.trainable_variables])


def get_non_trainable_params(model):
    return np.sum(
        [tf.size(w_matrix).numpy() for w_matrix in model.non_trainable_variables]
    )


def _get_model_memory_usage(model):
    """
    Copyright: From comment on stack-overflow:
    https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    """

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == "Model":
            internal_model_mem_count += _get_model_memory_usage(layer)
        single_layer_mem = 1
        out_shape = layer.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    number_size = 4.0
    if tf.keras.backend.floatx() == "float16":
        number_size = 2.0
    if tf.keras.backend.floatx() == "float64":
        number_size = 8.0

    total_memory = number_size * (
            shapes_mem_count + trainable_count + non_trainable_count
    )
    return total_memory + internal_model_mem_count
