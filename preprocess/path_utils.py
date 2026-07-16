from pathlib import Path
import json
from typing import List, Tuple

def get_paths_ff(datapath: Path) -> List[Tuple[Path, Path]]:
    """
    Pairs each fake video from the path with its real counterpart.
    Args:
        datapath: Directory containing manipulated videos.

    Returns:
        A list of ``(fake_video, real_video)`` path pairs.
    """
    real_datapath = datapath.parents[3] / "original_sequences" / "youtube" / "c23" / "videos"
    return [(x, real_datapath / (x.stem[:x.stem.find("_")] + ".mp4")) for x in datapath.iterdir()]

def get_paths_dfdc(datapath: Path | str) -> Tuple[List[Path], List[Path]]:
    real_videos, fake_videos = [], []

    with open(datapath / "metadata.json", "r") as f:
        metadata = json.load(f)
    
        # No reason why the test set would be paired (each fake example has a real counterpart).
        # So classes are processed and picked independently.
        real_videos = [datapath / video for video, info in metadata.items() if info["label"] == "REAL"]
        fake_videos = [datapath / video for video, info in metadata.items() if info["label"] == "FAKE"]

    return real_videos, fake_videos