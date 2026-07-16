import torch
import math
from typing import Tuple

class CLIPLoss(torch.nn.Module):
    """
    Loss combining binary classification with embedding alignment and uniformity.
    Follows the Deepfake Detection paper.

    Args:
        alpha: Weight of the alignment loss.
        beta: Weight of the uniformity loss.
    """
    def __init__(self, alpha: float = 0.1, beta: float = 0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta

    def _align(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Aligns samples with the same labels (1 to 1 and 0 to 0).
        """
        # Embeddings is 2D [B, emb_dim] and labels must be 1D [B, ]
        labels = labels.reshape((-1,))
        assert embeddings.size(0) == labels.size(-1)

        # GenD code. It was unclear what this does without the code.
        # [B, 1] == [1, B]
        # Broadcasts repeating the first labels along the columns and the latter across rows.
        labels_equal_mask = (labels[:, None] == labels[None, :]).triu(diagonal = 1).float()
        # Every (x, y) where x!=y and labels[x]==labels[y]==1.
        positive_indices = torch.nonzero(labels_equal_mask, as_tuple=False)

        if positive_indices.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Sum of squared difference across emb_dim to get the distance, mean across the batch.
        return ((embeddings[positive_indices[:, 0]] - embeddings[positive_indices[:, 1]])**2).sum(axis = 1).mean()

    def _uniform(self, embeddings: torch.Tensor):
        """
        Calculates the average squared distance between any two embeddings
        adjusted by a temperature t.
        """
        t = 2
        min_value = 1e-6
        return torch.pdist(embeddings, p=2).pow(2).mul(-t).exp().mean().clamp(min=min_value).log()

    def forward(self, outputs: Tuple[torch.Tensor, torch.Tensor], labels: torch.Tensor):
        out, norm = outputs
        return self.bce(out, labels) + self.alpha * self._align(norm, labels) + self.beta * self._uniform(norm)

class CosineCyclicWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with cyclic linear warm-up followed by cosine decay.

    Args:
        optimizer: Optimizer whose learning rate is scheduled.
        steps_per_epoch: Number of optimizer steps in one epoch.
        warmup_epochs: Number of warm-up epochs per cycle.
        cycle_epochs: Total number of epochs in each cycle.
        min_lr: Minimum learning rate.
        max_lr: Peak learning rate reached after warm-up.
        last_epoch: Index of the last completed step.

    Returns:
        A learning rate that increases linearly during warm-up and decays
        following a cosine schedule for the remainder of each cycle.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        steps_per_epoch: int,
        warmup_epochs: float = 1.0,
        cycle_epochs: float = 10.0,
        min_lr: float = 1e-5,
        max_lr: float = 3e-4,
        last_epoch: int = -1,
    ):
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.cycle_steps = cycle_epochs * steps_per_epoch
        self.decay_steps = self.cycle_steps - self.warmup_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)
 
    def get_lr(self):
        step = self.last_epoch  # incremented once per .step() call
        cycle_step = step % self.cycle_steps
 
        if cycle_step < self.warmup_steps:
            # linear warm-up: 1e-5 -> 3e-4 over one epoch
            frac = cycle_step / self.warmup_steps
            lr = self.min_lr + (self.max_lr - self.min_lr) * frac
        else:
            # cosine decay: 3e-4 -> 1e-5 over the remaining nine epochs
            decay_step = cycle_step - self.warmup_steps
            frac = decay_step / self.decay_steps
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * frac))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cos_factor
 
        return [lr for _ in self.optimizer.param_groups]