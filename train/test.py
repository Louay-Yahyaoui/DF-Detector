import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path
import click

from train.dataset import ImageDataset
from train.model import CLIPBinaryClassifier, load_clip
from utils.consts import DEVICE, PROCESSED_DATA_DIR, TOTAL_FRAMES, PLOT_DIR, DTYPE
from train.common_utils import load_paths_and_labels

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

def scale_img(img: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Min-max normalizes an image to the range ``[0, 1]``.
    
    Args:
        img: Input image as a NumPy array or PyTorch tensor.
    
    Returns:
        The normalized image with the same type as the input.
    """
    mi, ma = img.min(), img.max()
    img = (img - mi) / (ma - mi)
    return img

def plot_predictions(imgs: torch.Tensor | np.ndarray , preds: torch.Tensor | np.ndarray,
                     labels: torch.Tensor | np.ndarray, fontsize: int = 20, out_file: str = "predictions.png"):
    """
    Displays a batch of images with their predicted fake probabilities and labels.
    
    Args:
        imgs: Batch of images in ``(N, C, H, W)`` format.
        preds: Predicted fake probabilities for each image.
        labels: Ground-truth binary labels (0 = real, 1 = fake).
        fontsize: Font size used for subplot titles.
    """
    imgs, preds, labels = (tens.detach().cpu().numpy().flatten() if not isinstance(tens, np.ndarray) else tens for tens in (imgs, preds, labels) )
    imgs = scale_img(imgs)
    y = 4
    fig, axs = plt.subplots(y, max(imgs.shape[0] // y, 2), figsize=(40, 20))
    axs = axs.flatten()
    
    for ax in axs:
        # Removing all ax ticks including the ones not used.
        ax.axis("off")

    for i in range(min(len(axs), len(labels))):
        img, pred, ax, label = imgs[i], preds[i], axs[i], labels[i]
        ax.imshow(np.transpose(img, [1, 2, 0]).astype(np.float32))
        ax.set_title(f"Predicted {pred*100:0.2f}% fake and is {"fake" if label else "real"}.", 
                    fontsize = fontsize)
        
    plt.savefig(PLOT_DIR / out_file)
    plt.cla()

@torch.no_grad()
def predict_video(model: torch.nn.Module, video: torch.Tensor, device: torch.device | str,
                  return_float = True) -> torch.Tensor | float:
    """
    Predicts the probability that a video is fake from its preprocessed frames.
    
    Args:
        model: Trained binary classification model.
        video: Tensor containing the video's preprocessed frames.
        device: Device on which to run inference.
        return_float: If ``True``, returns a Python ``float``; otherwise, returns
            a scalar tensor.
    
    Returns:
        The mean predicted fake probability across all frames.
    """

    video = video.to(device)
    out, _ = model(video)
    prob = torch.nn.functional.sigmoid(out).mean()

    if return_float:
        return prob.item()
    else:
        return prob

@click.command()
@click.argument(
    "model-ckpt",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--device",
    type=torch.device,
    default=str(DEVICE),
    show_default=True,
    help="Device to train on, e.g. 'cuda' or 'cpu'.",
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=PROCESSED_DATA_DIR,
    show_default=True,
    help="Directory containing processed training data.",
)
@click.option(
    "--total-frames",
    type=int,
    default=TOTAL_FRAMES,
    show_default=True,
    help="Total number of frames per video sample.",
)
@click.option(
    "--plot-dir",
    type=click.Path(path_type=Path),
    default=PLOT_DIR,
    show_default=True,
    help="Directory to save training plots.",
)
@click.option(
    "--max-data",
    type=int,
    default=-1,
    show_default=True,
    help="Maximum test videos.",
)
def test(
    model_ckpt: Path,
    device: torch.device,
    data_dir: Path,
    total_frames: int,
    plot_dir: Path,
    max_data: int,
):
    dtype = DTYPE
    plot_dir.mkdir(exist_ok=True, parents=True)

    test_frames, y_test = load_paths_and_labels(data_dir / "test", dtype, total_frames=total_frames, shuffle=True, max_data=max_data)

    clip_normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),)

    test_data = ImageDataset(test_frames, y_test, dtype = dtype, transform = clip_normalize)

    # Test data from DFDC-10.
    test_loader = DataLoader(
        test_data, batch_size=total_frames, shuffle=False,
        num_workers=4)

    # Loading the pre-trained CLIP visual encoder.
    clip_model = load_clip("ViT-L/14", device = device)
    # Wrapping CLIP and adding the classification layer.
    model = CLIPBinaryClassifier(clip_model, dtype).to(device)

    # Loading the pre-trained checkpoint.
    ckpt = torch.load(model_ckpt)
    model.load_state_dict(ckpt)

    # Visualizing n_plot videos from the test loader.
    preds, images, labels = [], [], []
    n_plot = 32

    for batch, y in test_loader:
        preds.append(predict_video(model, batch, device, return_float = False).cpu().numpy())
        labels.append(y[0].cpu())
        images.append(batch[0].cpu())

    images = scale_img(np.stack(images, axis = 0))
    labels = np.stack(labels, axis = 0)
    preds = np.stack(preds, axis = 0)

    plot_predictions(images[:n_plot], preds[:n_plot], labels[:n_plot], fontsize = 15)

    # Measuring the Area Under the Receiver Operating Characteristic Curve of the entire test set. 
    roc_auc = roc_auc_score(labels, preds)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(labels, preds)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label = f"AUROC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(PLOT_DIR / 'auroc.png')

if __name__ == "__main__":
    test()