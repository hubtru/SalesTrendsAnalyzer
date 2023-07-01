"""
A helper class that tracks performances for experiments.
"""

import os
import tempfile
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

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

        ##Create the datasets before processing for more consistent timing
        validation = self.window_generator.val
        training = self.window_generator.train
        testing = self.window_generator.test

        before = time.process_time()
        self.performances.valid[name] = model.evaluate(validation, verbose=0)
        self.performances.test[name] = model.evaluate(testing, verbose=0)
        self.performances.train[name] = model.evaluate(training, verbose=0)
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
        stats = self.create_performance_data()
        stats.to_csv(where)

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

    def get_timing(self, model_name):
        return self.performances.timing[model_name]


def get_model_size_byte(model):
    _, keras_file = tempfile.mkstemp(".h5")
    try:
        tf.keras.models.save_model(model, keras_file, include_optimizer=False)
    except NotImplementedError:  # IF model cannot be saved (e.g. basemodel)
        return 0

    return os.path.getsize(keras_file)


def get_trainable_params(model):
    return np.sum([tf.size(w_matrix).numpy() for w_matrix in model.trainable_variables])


def get_non_trainable_params(model):
    return np.sum(
        [tf.size(w_matrix).numpy() for w_matrix in model.non_trainable_variables]
    )
