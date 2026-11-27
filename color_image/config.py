from dataclasses import fields, dataclass
from pydantic import Field
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ZHANG_DIR = ROOT / "baselines" / "zhang2016"
sys.path.insert(0, str(ZHANG_DIR))  # makes `import colorizers` work
MODELS = ("eccv16", "siggraph17", "sd_lora")


@dataclass
class Config:
    
    data: Path = ROOT / "data" / "coco"
    split: str = "test"
    n: int = 12 # 0 entire slipt
    size: int = 256
    batch_size: int = 8
    workers: int = 4
    seed: int = 42
    out: Path = ROOT / "runs" / "baselines"
    grid_rows: int = 6
    save_images: bool = True 
    models: str = "eccv16,siggraph17" 

    zhang_dir: Path = ZHANG_DIR
    pts_in_hull: Path = ROOT / "color_image" / "assets" / "pts_in_hull.npy"

    @property
    def model_list(self) -> list[str]:
        return [m.strip() for m in self.models.split(",") if m.strip()]


@dataclass
class ModelConfig:

    clip_id: str = "openai/clip-vit-base-patch32"
    sd_id: str = "timbrooks/instruct-pix2pix"
    clip_ckpt: Path | None = None 
    prompt: str = "Colorize the photo"
    timestep: int = 999
    unet_rank: int = 16
    vae_rank: int = 8
    n_bins: int = 313
    grad_checkpointing: bool = True

@dataclass
class LossConfig:
    w_l1: float = 1.0          
    w_lpips: float = 1.0       
    w_ce: float = 0.1          # rebalanced 313-bin color CE
    lpips_net: str = "alex"    # defults to alexnet
    lam: float = 0.5           # Zhang hyperparam
    sigma: float = 5.0         # Gaussian hyperparam
    knn: int = 5
    prior: Path = ROOT / "color_image" / "assets" / "ab_prior.npy"

@dataclass
class TrainConfig:
    data: Path = ROOT / "data" / "coco"
    out: Path = ROOT / "runs" / "phase_b"
    clip_ckpt: Path | None = ROOT / "runs" / "clip" / "gray_clip.pt"   # distilled GrayCLIP
    size: int = 256
    epochs: int = 12
    batch_size: int = 2
    accum: int = 4             # effective batch = batch_size * accum
    lr_lora: float = 1e-4      # pretrained-model LoRA layers
    lr_new: float = 1e-3       # skip convs + colour head
    wd: float = 0.01
    warmup: int = 100          # optimiser steps
    clip_grad: float = 1.0
    patience: int = 3          
    min_delta: float = 0.0
    workers: int = 4
    seed: int = 42
    limit: int | None = None       # first N train images
    val_limit: int | None = None   # first N val images
    n_vis: int = 8                 # images in the per-epoch sample grid
    resume: Path | None = None
    # GAN phase
    gan: bool = False
    init: Path | None = None       # start the generator from these weights (runs/phase_b/best.pt)
    lr_d: float = 1e-4             # discriminator heads
    d_warmup: int = 200            # optimiser steps where only D learns its job (no adversarial loss on G yet)
    w_adv: float = 0.05
    diffaug: bool = True
    dino_id: str = "facebook/dinov2-small"
