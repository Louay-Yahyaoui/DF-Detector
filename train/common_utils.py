from pathlib import Path
import torch
import json
import numpy as np

def load_paths_and_labels(input_path: Path, dtype: torch.dtype, dict_file: str = "paths.json",
                          total_frames: int = 32, shuffle: bool = False, max_data: int = -1):
    with open(input_path / dict_file, "r") as f:
        dict = json.load(f)

    # Loading the frame paths and labels from the preprocessed data.
    frames = list(map(lambda x: Path(input_path / (x)), dict.keys()))
    y = torch.tensor(np.stack(list(dict.values()), axis = 0), dtype = dtype)

    if shuffle:
        perm = (np.random.permutation(len(dict.keys()) // total_frames))  * total_frames
        # Shuffling the test set per video to run inference on full videos.
        frames = sum([frames[i:i + total_frames] for i in perm], start = [])
        y = torch.cat([y[i:i + total_frames] for i in perm], axis = 0)

    if max_data > -1:
        frames = frames[:total_frames * max_data]
        y = y[:total_frames * max_data]

    return frames, y
