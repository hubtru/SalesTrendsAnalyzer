"""
The class that abstracts away one experiment.
It has abstract-methods that have to be implemented for a
certain experiment.
"""


from abc import ABC, abstractmethod
from typing import Any, Dict

import matplotlib.pyplot as plt


from .analysis import create_results_table, save_history_plot
from .config import DatasetOptions
from .data import get_window_dataset
from .performance import Performances
from .decorators import with_random_seed_reset


class Experiment(ABC):
    """Class that encapsulates one type of experiment with certain learning parameters"""

    def __init__(self, name, path_to_output_folder):
        self.dataset_options = self.get_dataset_options()
        self.data = get_window_dataset(self.dataset_options)
        self.performance = Performances(self.data)
        self.name = name
        self.path_to_output_folder = path_to_output_folder

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
    def _evaluate_single_model(self, model_name, performance: Performances):
        model = self._get_model(model_name)
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
        for name in models.keys():
            self._evaluate_single_model(name, self.performance)

        self.save_information(models)

    def save_information(self, models):
        self.performance.save(f"{self.path_to_output_folder}/{self.name}_losses.csv")
        self.performance.save_plot(f"{self.path_to_output_folder}/{self.name}_plot.jpg")

        create_results_table(
            performance=self.performance,
            experiment_name=self.name,
            data_origin=self.dataset_options.data_origin,
            feature_names=str(list(self.data.train_df.columns.values)),
            train_set_instances=self.data.train_samples,
            valid_set_instances=self.data.val_samples,
            test_set_instances=self.data.test_samples,
            train_settings=self.get_train_settings(),
            models=models,
        ).to_csv(f"{self.path_to_output_folder}/{self.name}_results.csv")
        print(" => Experiment Info Saved.")

    def run_model(self, model_name: str):
        """Runs a single model"""
        single_performance = Performances(self.data)
        history = self._evaluate_single_model(model_name, single_performance)

        single_performance.save(
            f"{self.path_to_output_folder}/{model_name}_{self.name}_losses.csv"
        )
        single_performance.save_plot(
            f"{self.path_to_output_folder}/{model_name}_{self.name}_plot.jpg"
        )
        save_history_plot(
            history,
            where=f"{self.path_to_output_folder}/{model_name}_{self.name}_history.jpg",
        )
