import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def show_img(img: np.ndarray, title: str, path: Path, total_frames: int, fontsize: int = 20):
    fig, axs = plt.subplots(4, total_frames // 4, figsize=(40, 20))
    axs = axs.flatten()

    for frame, ax in zip(img, axs):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')

    fig.suptitle(title, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(path / (title + ".png"))
    plt.cla()