"""
The class that abstracts away one experiment.
It has abstract-methods that have to be implemented for a
certain experiment.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import matplotlib.pyplot as plt

from .analysis import create_results_table, save_history_plot, save_predictions_plot
from .config import DatasetOptions, ALL_USED_PRODUCT_IDS
from .data import get_window_dataset
from .decorators import with_random_seed_reset
from .performance import Performances


class Experiment(ABC):
    """Class that encapsulates one type of experiment with certain learning parameters"""

    def __init__(self, name, path_to_output_folder):
        self.dataset_options = self.get_dataset_options()
        self.data = get_window_dataset(self.dataset_options)
        self.performance = Performances(self.data)
        self.name = name
        self.path_to_output_folder = path_to_output_folder

    def __repr__(self):
        return f'Experiment {self.name}'

    @abstractmethod
    def get_dataset_options(self) -> DatasetOptions:
        """Return the options of for the dataset"""

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

    @abstractmethod
    def get_models(self) -> Dict[str, Any]:
        pass

    def _get_model(self, model_name):
        models = self.get_models()
        if model_name not in models.keys():
            raise ValueError(f"Model {model_name} not found")

        return models[model_name]

    @with_random_seed_reset
    def _evaluate_single_model(self, model_name, model, performance: Performances):
        print("Fit model: ", model_name)
        history = self.compile_and_fit(model)
        print(" ... Done")

        performance.register_performance(model_name, model)
        print("----------------------------------")
        return history

    def run(self):
        """Runs the experiment"""
        models = self.get_models()
        print(f"Run Experiment: {self.name}")
        for name, model in models.items():
            self._evaluate_single_model(name, model, self.performance)

        self.save_information(models)

    def save_information(self, models):
        self.performance.save(f"{self.path_to_output_folder}/{self.name}_losses.csv")
        self.performance.save_plot(f"{self.path_to_output_folder}/{self.name}_plot.jpg")

        create_results_table(
            performance=self.performance,
            experiment_name=self.name,
            window_generator=self.data,
            data_origin=self.dataset_options.data_origin,
            train_settings=self.get_train_settings(),
            models=models,
        ).to_csv(f"{self.path_to_output_folder}/{self.name}_results.csv")
        print(" => Experiment Info Saved.")

    def run_model(self, model_name: str, output_all_label_images=False):
        """Runs a single model"""
        model = self._get_model(model_name)
        single_performance = Performances(self.data)
        history = self._evaluate_single_model(model_name, model, single_performance)

        file_location = f"{self.path_to_output_folder}/{self.name}/{model_name}"

        if not os.path.exists(file_location):
            os.makedirs(file_location)

        single_performance.save(f"{file_location}/losses.csv")
        save_history_plot(history, where=f"{file_location}/history.jpg")
        if output_all_label_images:
            save_predictions_plot(model, self.data, where=f"{file_location}/predictions.jpg",
                                  columns=ALL_USED_PRODUCT_IDS)
        else:
            save_predictions_plot(model, self.data, where=f"{file_location}/predictions.jpg",
                                  columns=self.data.label_columns)

        create_results_table(
            performance=single_performance,
            experiment_name=self.name,
            window_generator=self.data,
            data_origin=self.dataset_options.data_origin,
            train_settings=self.get_train_settings(),
            models={model_name: model},
        ).to_csv(f"{file_location}/results.csv")
        print(" => Experiment Info Saved.")
        plt.clf()
