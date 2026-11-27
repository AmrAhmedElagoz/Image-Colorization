import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from transformers import Dinov2Model

from .common import gray_rgb

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DiscHead(nn.Module):
    """Judge; Discriminator trainable head over DinoV2"""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # spectral_norm stabilize the D 
            spectral_norm(nn.Conv2d(2 * dim, hidden, 1)), nn.LeakyReLU(0.2),    # 1-patch
            spectral_norm(nn.Conv2d(hidden, hidden, 3, padding=1)), nn.LeakyReLU(0.2),  # 3-neighbour patchs
            spectral_norm(nn.Conv2d(hidden, 1, 1))) # scoring

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())          # DINO features are bf16; the head's weights are fp32


class DinoDiscriminator(nn.Module):

    # evenly spread (3, 6, 9, 12) over dinov2 to get the early middle and high blocks
    def __init__(self, dino_id: str = "facebook/dinov2-small", layers: tuple[int, ...] = (3, 6, 9, 12),
                 size: int = 224, hidden: int = 128):
        super().__init__()
        self.dino = Dinov2Model.from_pretrained(dino_id, torch_dtype=torch.bfloat16).eval().requires_grad_(False)
        self.layers, self.size = layers, size
        dim = self.dino.config.hidden_size                      # 384 for ViT-S
        self.heads = nn.ModuleList(DiscHead(dim, hidden) for _ in layers)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()                                        # frozen backbone, always eval
        return self

    def features(self, rgb: torch.Tensor) -> list[torch.Tensor]:
        """rgb [B,3,H,W] in [0,1] not [0, 255] -> one [B, C, 16, 16] map per chosen block."""
        x = F.interpolate(rgb.float(), size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)   # from 256 to dino's 224
        x = (x - self.mean) / self.std  # imagenet norm

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            hs = self.dino(pixel_values=x.to(torch.bfloat16), output_hidden_states=True).hidden_states
        g = self.size // self.dino.config.patch_size            # 16; dino's patch is 14
        out = []
        for i in self.layers:                                   
            t = hs[i][:, 1:]                                    # drop the CLS token
            t = F.layer_norm(t.float(), t.shape[-1:])           # depths have different scales; put them on one
            out.append(t.transpose(1, 2).reshape(len(t), -1, g, g)) # preps for next phase; 3x3 neighbour CNN (dino's raw out is flatten 256)
        return out  # four blocks of tensors

    def forward(self, rgb: torch.Tensor, L: torch.Tensor) -> list[torch.Tensor]:

        with torch.no_grad():
            cond = self.features(gray_rgb(L))                   

        # for each block side by side color and gray let the block judge and score it
        return [h(torch.cat([f, c], 1)) for h, f, c in zip(self.heads, self.features(rgb), cond)]


def diff_augment(rgb: torch.Tensor, L: torch.Tensor, shift: float = 0.125) -> tuple[torch.Tensor, torch.Tensor]:
    """
    for small dataset judge can't see the same image twice; randomly pad and shift by 32 pixels
    """
    x = torch.cat([rgb, L], 1)
    # 50:50 flip
    if torch.rand(()) < 0.5:
        x = x.flip(3)

    H, W = x.shape[-2:]
    s = int(H * shift)  # 32 pixel shift
    dy, dx = torch.randint(-s, s + 1, (2,)).tolist()
    x = F.pad(x, (s, s, s, s))[:, :, s + dy:s + dy + H, s + dx:s + dx + W]
    return x[:, :3], x[:, 3:]
