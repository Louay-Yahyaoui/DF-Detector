from pathlib import Path
import torch
import json
import numpy as np

def load_paths_and_labels(input_path: Path, dtype: torch.dtype, dict_file: str = "paths.json"):
    with open(input_path / dict_file, "r") as f:
        dict = json.load(f)

    # Loading the frame paths and labels from the preprocessed data.
    frames = list(map(lambda x: Path(input_path / (x)), dict.keys()))
    y = torch.tensor(np.stack(list(dict.values()), axis = 0), dtype = dtype)

    return frames, y
