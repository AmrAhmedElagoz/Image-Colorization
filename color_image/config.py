from dataclasses import fields, dataclass
from pydantic import Field
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
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

    zhang_dir: Path = ROOT / "baselines" / "zhang2016"
    pts_in_hull: Path = ROOT / "scripts" / "assets" / "pts_in_hull.npy" 
    sys.path.insert(0, str(zhang_dir))

    @property
    def model_list(self) -> list[str]:
        return [m.strip() for m in self.models.split(",") if m.strip()]