"""
General functions for creating plots and
statistics about experiments and model trainings.
"""

import matplotlib.pyplot as plt


def plot_history(history):
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss Development")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
