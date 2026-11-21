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

