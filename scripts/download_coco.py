import argparse
import json
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/train2017/{}"
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def is_colour(path: Path) -> bool:
    """True if the image has real chroma (not L-mode, and not RGB with R=G=B)."""
    img = Image.open(path)
    if img.mode != "RGB":
        return False
    a = np.asarray(img.resize((64, 64)), dtype=np.int16)
    return bool(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() > 2)


def fetch(name: str, img_dir: Path) -> tuple[str, bool]:
    dst = img_dir / name
    try:
        if not dst.exists():
            urllib.request.urlretrieve(IMG_URL.format(name), dst)
        if is_colour(dst):
            return name, True
    except Exception as e: 
        print(f"skip {name}: {e}")
    dst.unlink(missing_ok=True)
    return name, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/coco"))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    out, img_dir = args.out, args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ann_zip = out / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        print("downloading annotations ...")
        urllib.request.urlretrieve(ANN_URL, ann_zip)
    with zipfile.ZipFile(ann_zip) as z:
        captions = json.loads(z.read("annotations/captions_train2017.json"))

    id2name = {im["id"]: im["file_name"] for im in captions["images"]}
    caps: dict[str, list[str]] = {}
    for a in captions["annotations"]:
        caps.setdefault(id2name[a["image_id"]], []).append(a["caption"].strip())

    candidates = sorted(id2name.values())
    random.Random(args.seed).shuffle(candidates)

    kept: list[str] = []
    pos = 0
    with ThreadPoolExecutor(args.workers) as pool:
        while len(kept) < args.n:
            batch = candidates[pos : pos + (args.n - len(kept))]
            pos += len(batch)
            kept += [n for n, ok in pool.map(lambda n: fetch(n, img_dir), batch) if ok]
            print(f"{len(kept)}/{args.n} colour images")

    keep = set(kept)
    for p in img_dir.iterdir():
        if p.name not in keep:
            p.unlink()

    (out / "splits").mkdir(exist_ok=True)
    start = 0
    for split, frac in SPLITS.items():
        k = round(frac * args.n)
        (out / "splits" / f"{split}.txt").write_text("\n".join(kept[start : start + k]) + "\n")
        start += k
    (out / "captions.json").write_text(json.dumps({n: caps[n] for n in kept}, indent=1))
    print(f"done: {len(kept)} images in {img_dir}")


if __name__ == "__main__":
    main()
