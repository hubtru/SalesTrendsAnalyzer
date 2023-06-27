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
        self.performance = Performances(self.data)
        self.name = name

    @abstractmethod
    def compile_and_fit(self, model):
        """Return the history of the fit-function"""

    def run(self, models: Dict[str, Any]):
        """Runs the experiment"""
        for name, model in models.items():
            random.set_seed(0)
            print("Run model: ", name)

            self.compile_and_fit(model)
            print("  ... measuring performance:")
            self.performance.register_performance(name, model)
            print("----------------------------------")

        self.performance.save(f"./{self.name}_losses.csv")
        print("Statistics Saved.")
        self.performance.save_plot(f"./{self.name}_plot.jpg")
        print("Image Saved.")
