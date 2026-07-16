import matplotlib.pyplot as plt
from typing import Iterator, Dict, Iterable
from pathlib import Path

def plot_history(epochs_range: Iterator[int], x: str, history: Dict[str, Iterable[float]],
             title: str, out_file: Path | str):

    plt.plot(epochs_range, history["train_" + x], 
            label="Training " + title, linewidth=2.2, marker='o', markersize=4)
    plt.plot(epochs_range, history["val_" + x], 
            label="Validation " + title, linewidth=2.2, marker='o', markersize=4)

    plt.title(title + " over Epochs", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(title, fontsize=12)

    plt.legend(fontsize=11, frameon=True, framealpha=0.9, loc='upper right')
    plt.xticks(list(epochs_range))

    plt.tight_layout()
    plt.savefig(out_file)
    plt.cla()