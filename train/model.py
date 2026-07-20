import torch
import clip

def load_clip(model_name: str, device: torch.device | str) -> torch.nn.Module:
    """
    Loads the visual encoder from a pretrained CLIP model.

    Args:
        model_name: Name of the CLIP model to load.
        device: Device to load CLIP onto. 

    Returns:
        The CLIP visual encoder with its final projection layer removed.
    """
    # Taking only the visual model, we won't use their preprocess or the language model.
    clip_model = clip.load(model_name, device=device)[0].visual
    # Removing the final projection and updating output dim to exclude it.
    clip_model.proj = None
    clip_model.output_dim = clip_model.transformer.width
    return clip_model

class L2Normalize(torch.nn.Module):
    """
    nn.Module wrapper for the L2Normalization from the functional API.
    """
    def __init__(self, dim: int = 1, eps: float = 1e-12):
        super(L2Normalize, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=1, eps=self.eps)

class CLIPBinaryClassifier(torch.nn.Module):
    """
    Wraps a CLIP visual encoder with a binary classification head.
    Args:
        visual: Pretrained CLIP visual encoder.
    """
    def __init__(self, clip_model: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        # CLIP model may run in lower precision to save memory.
        self.visual = clip_model
        self.visual_dtype = next(self.visual.conv1.parameters()).dtype

        self.dtype = dtype

        for name, param in self.visual.named_parameters():
            param.requires_grad = "ln" in name
 
        embed_dim = self.visual.output_dim  # 1024 for ViT-L/14
        self.l2 = L2Normalize()
        self.linear = torch.nn.Linear(embed_dim, 1, dtype = self.dtype)
 
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images.to(self.visual_dtype)).to(self.dtype)
        l2 = self.l2(features)
        return self.linear(l2), l2