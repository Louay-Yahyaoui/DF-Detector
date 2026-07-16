from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))

import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import click
from typing import Tuple, Dict

from utils.consts import (PROCESSED_DATA_DIR, DEVICE, DTYPE, TOTAL_FRAMES, BATCH_SIZE, 
    MODEL_CKPT, PLOT_DIR, EPOCHS, LR, MIN_LR, VAL_SPLIT)
from train.dataset import ImageDataset
from train.model import CLIPBinaryClassifier, load_clip
from train.loss import CLIPLoss, CosineCyclicWarmupLR
from train.plot_utils import plot_history
from train.common_utils import load_paths_and_labels

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_train: int,
    n_val: int, 
    epochs: int = 10,
    lr: float = 3e-4,
    min_lr: float = 1e-5,
    alpha: float = 0.1,
    beta: float  = 0.5,
    start_from_ckpt: Path | str | None = None,
) -> Tuple[torch.nn.Module, Dict]:
    """
    Train a CLIP ViT-L/14 binary classifier.
 
     Args:
        model: Model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        epochs: Number of training epochs.
        lr: Maximum learning rate.
        min_lr: Minimum learning rate for the scheduler.
        alpha: Weight of the alignment loss.
        beta: Weight of the uniformity loss.
        start_from_ckpt: Optional checkpoint to load before training.
        retrain_ckpt: If ``False``, returns the loaded checkpoint without further
            training.
    
    Returns:
        The best-performing model in evaluation mode.
    """
    device = next(model.parameters()).device
    print(f"Using device: {device}")
 
    criterion = CLIPLoss(alpha = alpha, beta = beta)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    steps_per_epoch = len(train_loader)
    scheduler = CosineCyclicWarmupLR(optimizer, steps_per_epoch, cycle_epochs = 10,
                    max_lr = lr, min_lr = min_lr)

    val_losses, val_accs, train_losses, train_accs = ([] for _ in range(4))
    best_val_loss = float("inf")
    best_state    = None

    if start_from_ckpt:
        ckpt = torch.load(start_from_ckpt)
        model.load_state_dict(ckpt)
        best_state  = model.state_dict()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
 
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)
 
            optimizer.zero_grad()
            outputs, l2 = model(images)
            loss    = criterion((outputs, l2), labels)
    
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss    += loss.item() * images.size(0)
            preds          = (outputs >= 0).float()
            train_correct += (preds == labels).sum().item()


 
        train_loss /= n_train
        train_acc   = train_correct / n_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
 
        model.eval()
        val_loss = 0.0
        val_correct = 0
 
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
 
                outputs, l2 = model(images)
                loss        = criterion((outputs, l2), labels)
                
                val_loss   += loss.item() * images.size(0)
                preds       = (outputs >= 0.).float()
                val_correct += (preds == labels).sum().item()
 
        val_loss /= n_val
        val_acc   = val_correct / n_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)
 
        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )
 
        # Save the best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
    
        if best_state is not None:
            torch.save(best_state, 'best_model.pt')
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), f"epoch_{epoch}.pt")

    history = {"train_losses":train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs}

    return model.eval(), history

@click.command()
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
    "--batch-size",
    type=int,
    default=BATCH_SIZE,
    show_default=True,
    help="Training batch size.",
)
@click.option(
    "--model-ckpt",
    type=click.Path(path_type=Path),
    default=MODEL_CKPT,
    show_default=True,
    help="Path to a model checkpoint to resume from (optional).",
)
@click.option(
    "--plot-dir",
    type=click.Path(path_type=Path),
    default=PLOT_DIR,
    show_default=True,
    help="Directory to save training plots.",
)
@click.option(
    "--val-split",
    type=float,
    default=VAL_SPLIT,
    show_default=True,
    help="Fraction of data to use for validation.",
)
@click.option(
    "--epochs",
    type=int,
    default=EPOCHS,
    show_default=True,
    help="Number of training epochs.",
)
@click.option(
    "--lr",
    type=float,
    default=LR,
    show_default=True,
    help="Initial learning rate.",
)
@click.option(
    "--min-lr",
    type=float,
    default=MIN_LR,
    show_default=True,
    help="Minimum learning rate (e.g. for scheduler floor).",
)
def train(
    device: torch.device,
    data_dir: Path,
    total_frames: int,
    batch_size: int,
    model_ckpt: Path | None,
    plot_dir: Path,
    val_split: float,
    epochs: int,
    lr: float,
    min_lr: float,
) -> None:
    err = "Data directory doesn't exist."
    assert data_dir.exists(), err
    
    torch.manual_seed(42)
    dtype = DTYPE
    plot_dir.mkdir(exist_ok=True, parents=True)

    frames, y_train = load_paths_and_labels(data_dir / "train", dtype)

    clip_normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),)

    data = ImageDataset(frames, y_train, dtype = dtype, transform = clip_normalize)

    # We have more than enough training data but not enough compute.
    # Data augmentation remains in the back pocket for now.
    num_workers = 4

    n_val   = int(len(data) * val_split) // total_frames * total_frames
    n_train = len(data) - n_val

    # Train, val split. They originate from the same data (FF++).
    # The videos are pre-shuffled. Splitting train and val by video (not frame) to prevent leakage.
    train_ds = Subset(data, list(range(n_train)))
    val_ds = Subset(data, list(range(n_train + 1, len(data))))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers)

    # Loading the pre-trained CLIP visual encoder.
    clip_model = load_clip("ViT-L/14", device = device)
    # Wrapping CLIP and adding the classification layer.
    model = CLIPBinaryClassifier(clip_model, dtype).to(device)

    model, history = train_loop(model, train_loader, val_loader, n_train, n_val, epochs = epochs, lr = lr,
         start_from_ckpt = model_ckpt, min_lr = min_lr)
    print("Training done.")

    epochs_range = range(1, epochs + 1)

    # Saving history plots to disk.
    plot_history(epochs_range, "losses", history, "Loss", plot_dir / "training_loss.png")
    plot_history(epochs_range, "accs", history, "Accuracy", plot_dir / "training_accuracy.png")

if __name__ == "__main__":
    train()