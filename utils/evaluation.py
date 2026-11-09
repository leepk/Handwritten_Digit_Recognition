import string
import matplotlib.pyplot as plt
import numpy as np


def build_probability_figure(probabilities: np.ndarray):
    fig, ax = plt.subplots(figsize=(16, 5))
    characters = list(
        string.digits +
        string.ascii_uppercase +
        string.ascii_lowercase
    )
    x = np.arange(len(probabilities))

    # Display the prediction probabilities as a bar chart.
    ax.bar(x, probabilities * 100, width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(characters, rotation=0, fontsize=12)
    ax.set_xlabel("Digit and Charactors", fontsize=16)
    ax.set_ylabel("Probability (%)", fontsize=16)
    ax.set_ylim(0, 100)
    #ax.set_title("Prediction Probability")
    fig.tight_layout()
    return fig
