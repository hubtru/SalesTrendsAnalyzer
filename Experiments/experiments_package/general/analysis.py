import matplotlib.pyplot as plt


def plot_history(history):
    plt.plot(history.history["loss"], label="Loss (Train)")
    plt.plot(history.history["val_loss"], label="Loss (Validation)")
    plt.legend()
