from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .data import WindowGenerator


@dataclass
class TrackedPerformances:
    valid = {}
    test = {}
    train = {}
    metrics_names = {}


class Performances:
    def __init__(self, window_generator: WindowGenerator):
        self.window_generator = window_generator

        self.performances = TrackedPerformances()

        self.model_names = []

    def register_performance(self, name, model):
        self.model_names.append(name)

        self.performances.valid[name] = model.evaluate(
            self.window_generator.val, verbose=0
        )
        self.performances.test[name] = model.evaluate(
            self.window_generator.test, verbose=0
        )
        self.performances.train[name] = model.evaluate(
            self.window_generator.train, verbose=0
        )
        self.performances.metrics_names[name] = model.metrics_names

    def save(self, where):
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

        stats.to_csv(where)

    def save_plot(self, where):
        loss_name = "loss"

        x = np.arange(len(self.model_names))
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
        plt.bar(x - 0.3, train_perf, width, label="Train")
        plt.bar(x, validation_perf, width, label="Validation")
        plt.bar(x + 0.3, test_perf, width, label="Test")
        plt.xticks(ticks=x, labels=self.model_names, rotation=45)
        plt.legend()
        plt.savefig(where)
