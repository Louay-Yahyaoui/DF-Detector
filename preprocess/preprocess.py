from pathlib import Path
import json
import numpy as np
from collections import OrderedDict
from utils.consts import PROCESSED_DATA_DIR, DEEPFAKE_METHODS, N_DATA,\
     N_TEST, TOTAL_FRAMES, IMG_DIMS, SCALE, PLOT_DIR
import click

from preprocess.img_utils import preprocess_and_write_frames, preprocess_video
from preprocess.plot_utils import show_img
from preprocess.path_utils import get_paths_ff, get_paths_dfdc

from retinaface import RetinaFace
import torch

@click.command()
@click.argument(
    "data-dir",
    type=click.Path(path_type=Path),
    required = True,
)
@click.argument(
    "test-data-dir",
    type=click.Path(path_type=Path),
    required = True,
)
@click.option(
    "--processed-data-dir",
    type=click.Path(path_type=Path),
    default=PROCESSED_DATA_DIR,
    show_default=True,
    help="Directory to write processed data to.",
)
@click.option(
    "--n-data",
    type=int,
    default=N_DATA,
    show_default=True,
    help="Number of training examples per method.",
)
@click.option(
    "--n-test",
    type=int,
    default=N_TEST,
    show_default=True,
    help="Number of test videos per class.",
)
@click.option(
    "--total-frames",
    type=int,
    default=TOTAL_FRAMES,
    show_default=True,
    help="Total number of frames to extract per video.",
)
@click.option(
    "--img-dims",
    type=(int, int),
    nargs=2,
    default=IMG_DIMS,
    show_default=True,
    help="Target image dimensions as two ints, e.g. --img-dims 224 224.",
)
@click.option(
    "--scale",
    type=float,
    default=SCALE,
    show_default=True,
    help="Scale factor for face bounding box cropping.",
)
@click.option(
    "--plot-dir",
    type=click.Path(path_type=Path),
    default=PLOT_DIR,
    show_default=True,
    help="Directory to save preprocessing plots/logs.",
)
def preprocess(
    data_dir: Path,
    test_data_dir: Path,
    processed_data_dir: Path,
    n_data: int,
    n_test: int,
    total_frames: int,
    img_dims: tuple[int, int],
    scale: float,
    plot_dir: Path,
) -> None:
    err1 = "Training data directory doesn't exist."
    err2 = "Test data directory doesn't exist."
    assert data_dir.exists(), err1
    assert test_data_dir.exists(), err2

    # Making output directories if not already made.
    processed_data_dir.mkdir(exist_ok=True, parents=True)
    plot_dir.mkdir(exist_ok=True, parents=True)

    np.random.seed(42)

    data_part = data_dir / "manipulated_sequences" / DEEPFAKE_METHODS[0] / "c23" / "videos"
    videos = {DEEPFAKE_METHODS[0]: get_paths_ff(data_part)}

    n = len(videos[DEEPFAKE_METHODS[0]])
    print(f"There are {n} deepfake videos in the training dataset.")

    # choose N_DATA random indices without replacement from 1 method.
    # We then get all the others methods and the real video for those chosen videos.
    chosen_indices = np.random.permutation(n)[:n_data]
    chosen_pairs = [videos[DEEPFAKE_METHODS[0]][i] for i in chosen_indices]

    # recuperate the paths of the other fake methods from the first one and add in the real video paths.
    videos = [Path(str(x).replace(DEEPFAKE_METHODS[0], method)) for method in DEEPFAKE_METHODS 
            for x, _ in chosen_pairs] + [y for _, y in chosen_pairs]

    # Face detection model retinaface, the highest score face is cropped from each frame.
    retina_model = RetinaFace.build_model()

    methods = sum([[method] * n_data for method in DEEPFAKE_METHODS + ["real"]], start = [])
    perm = np.random.permutation(len(methods))
    labels = [1] * n_data * len(DEEPFAKE_METHODS) + [0] * n_data

    videos, labels, methods = [videos[i] for i in perm], [labels[i] for i in perm], \
                    [methods[i] for i in perm]

    out_train = processed_data_dir / "train"
    out_train.mkdir(parents=True, exist_ok=True)

    frames = preprocess_and_write_frames(retina_model, videos, methods, total_frames, img_dims, out_train, scale)
    # Labels for the frame paths.
    y_train = torch.tensor(labels, dtype = torch.float16).repeat_interleave(total_frames).numpy()

    train_dict = OrderedDict({str(k.name): float(v) for k, v in zip(frames, y_train)})
    with open(out_train / "paths.json", "w") as f:
        json.dump(train_dict, f)
    print(f"Train data is saved to {str(out_train)}.")

    # Fake example
    idx = labels.index(1)
    example_path = videos[idx]
    # Random real video. No risk using index because there's always N_DATA real examples. 
    real_example = videos[labels.index(0)]

    show_img(preprocess_video(retina_model, example_path, total_frames, img_dims, scale).numpy().transpose([0, 2, 3, 1]).astype(np.uint8),
              "Random Fake Video", plot_dir, total_frames, fontsize = 50)
    show_img(preprocess_video(retina_model, real_example, total_frames, img_dims, scale).numpy().transpose([0, 2, 3, 1]).astype(np.uint8),
              "Random Real Video", plot_dir, total_frames, fontsize = 50)

    # Getting the test set video paths.
    fake_videos, real_videos = [], []

    for part in range(1): #10
        data_part = test_data_dir / f"dfdc_train_part_0{part}" / f"dfdc_train_part_{part}"

        real, fake = get_paths_dfdc(data_part)
        real_videos.extend(real)
        fake_videos.extend(fake)

    print(f"Test dataset contains {len(real_videos)} real videos and {len(fake_videos)} fake ones.")

    # Sampling a random subset of the test data containing N_TEST real examples and N_TEST fake ones.
    fake_indices = np.random.permutation(len(fake_videos))[:n_test]
    real_indices = np.random.permutation(len(real_videos))[:n_test]
    test_videos = [fake_videos[i] for i in fake_indices] + [real_videos[j] for j in real_indices]

    out_test = processed_data_dir / "test"
    out_test.mkdir(parents=True, exist_ok=True)
    test_frames = preprocess_and_write_frames(retina_model, test_videos, ["fake"] * n_test + ["real"] * n_test,
                 total_frames, img_dims, out_test, scale)
    y_test = torch.cat([torch.ones(total_frames * n_test), torch.zeros(total_frames * n_test)], axis = 0).numpy()

    test_dict = OrderedDict({str(k.name): float(v) for k, v in zip(test_frames, y_test)})
    with open(out_test / "paths.json", "w") as f:
        json.dump(test_dict, f)
    print(f"Test data is saved to {str(out_test)}.")


if __name__ == "__main__":
    preprocess()