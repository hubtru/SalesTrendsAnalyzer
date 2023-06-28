from abc import ABC, abstractmethod
from typing import Any, Dict

from tensorflow import random

from .config import DatasetOptions
from .data import get_window_dataset
from .performance import Performances


class Experiment(ABC):
    """Class that encapsulates one type of experiment with certain learning parameters"""

    def __init__(self, name, dataset_options: DatasetOptions):
        self.data = get_window_dataset(dataset_options)
        self.dataset_options = dataset_options
        self.performance = Performances(self.data)
        self.name = name

    @abstractmethod
    def compile_and_fit(self, model):
        """Return the history of the fit-function"""

    @abstractmethod
    def get_train_settings(self) -> Dict[str, Any]:
        """
        Return a dictionary with <<descrption>> => <<value>> for settings
        of the training algorithm, that are important.

        At Least:
        - batch size
        - learning rate
        - training epochs
        """

    def run(self, models: Dict[str, Any]):
        """Runs the experiment"""
        for name, model in models.items():
            random.set_seed(0)
            print("Run model: ", name)

            self.compile_and_fit(model)
            print("  ... measuring performance:")
            self.performance.register_performance(name, model)
            print("----------------------------------")

        self.save_information(models)

    def save_information(self, models):
        self.performance.save(f"./{self.name}_losses.csv")
        print("Statistics Saved.")
        self.performance.save_plot(f"./{self.name}_plot.jpg")
        print("Image Saved.")

        info = self.create_info(models)
        info.to_csv(f"./{self.name}_results.csv")
        print("Experiment Info Saved.")

    def create_info(self, models: Dict[str, Any]):
        performance_stats = self.performance.create_performance_data()
        performance_stats[[("Experiment", "Name")]] = self.name
        performance_stats[
            [("Experiment", "Data Origin")]
        ] = self.dataset_options.data_origin
        performance_stats[[("Experiment", "Used Data Columns")]] = str(
            list(self.data.train_df.columns.values)
        ).replace(",", ",\n")
        performance_stats[
            [("Experiment", "Data Instances (train/valid/test)")]
        ] = f"{len(self.data.train_df)}/{len(self.data.val_df)}/{len(self.data.test_df)}"
        for setting, description in self.get_train_settings().items():
            performance_stats[[("Training Settings", setting)]] = description

        performance_stats[[("Timing", "Latency (ms/observation)")]] = " "
        performance_stats[[("Model", "Summary")]] = " "
        for name, model in models.items():
            performance_stats.at[
                name, ("Timing", "Latency (ms/observation)")
            ] = "TODO"  # TODO: Add Timing
            summary_parts = []
            model.summary(print_fn=lambda x: summary_parts.append(x))
            summary = "\n".join(summary_parts)
            performance_stats.at[name, ("Model", "Summary")] = summary

        return performance_stats
