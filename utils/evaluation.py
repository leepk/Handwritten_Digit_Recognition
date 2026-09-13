import matplotlib.pyplot as plt
import numpy as np


def build_probability_figure(probabilities: np.ndarray):
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    digits = np.arange(10)

    # Display the prediction probabilities as a bar chart.
    ax.bar(digits, probabilities * 100)
    ax.set_xticks(digits)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Prediction Probability")
    fig.tight_layout()
    return fig
