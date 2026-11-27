import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.color as kc
from torch.utils.checkpoint import checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, inject_adapter_in_model
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from .config import ModelConfig
from .common import norm_L, gray_rgb

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class GrayCLIP(nn.Module):
    """CLIP ViT-B/32 vision to read both gray images and color images and generates tokens for both.
    """

    def __init__(self, clip_id: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.net = CLIPVisionModel.from_pretrained(clip_id, torch_dtype=dtype)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def tokens(self, rgb: torch.Tensor) -> torch.Tensor:
        """Any RGB image in [0, 1] -> [B, 50, 768] tokens (the teacher in distillation sees colour here)."""
        with torch.autocast("cuda", enabled=False):  # keep the resize / normalisation in fp32
            x = F.interpolate(rgb.float(), size=(224, 224), mode="bicubic", antialias=True).clamp(0, 1)
            x = ((x - self.mean) / self.std).to(self.net.dtype)
        return self.net(x).last_hidden_state

    def forward(self, L: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):  # this disables mixed precision; Lab -> RGB gray needs full precision, bf has 3 significant decimal digits vs fp32 7
            rgb = gray_rgb(L.float())
        return self.tokens(rgb)


class GrayCLIPDistiller:
    """
    A teacher/student dist class for CLIP to read gray images.

    Teacher: frozen CLIP looking at the COLOR photo. 
    Student: GrayCLIP looking at the GRAY version (L only).    
    """

    def __init__(self, clip_id: str, device: str | None = None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # The student keeps fp32 weights (frozen too) and runs under bf16
        self.teacher = GrayCLIP(clip_id, torch.bfloat16).requires_grad_(False).eval().to(self.dev)
        self.student = GrayCLIP(clip_id, torch.float32).to(self.dev)

    def loss(self, rgb: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            target = self.teacher.tokens(rgb).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            pred = self.student(L).float()
        return F.mse_loss(pred, target)

    def _epoch(self, loader, opt=None, sched=None) -> float:
        train = opt is not None
        self.student.train(train)
        total, n = 0.0, 0
        for batch in tqdm(loader, desc="train" if train else "val", leave=False):
            rgb, L = batch["rgb"].to(self.dev), batch["L"].to(self.dev)
            with torch.set_grad_enabled(train):
                loss = self.loss(rgb, L)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                opt.step()
                sched.step()
            total += loss.item() * len(rgb)
            n += len(rgb)
        return total / n

    @torch.no_grad()
    def evaluate(self, loader) -> float:
        return self._epoch(loader)

    def fit(self, train_dl, val_dl, out: Path, epochs: int = 25, lr: float = 2e-5, wd: float = 0.01,
            warmup: int = 100, patience: int | None = 5, min_delta: float = 0.0) -> float:
        """AdamW + linear warm-up + cosine decay + grad-clip 1.0."""

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        opt = torch.optim.AdamW(self.student.parameters(), lr=lr, weight_decay=wd)
        total = epochs * len(train_dl)

        # linear warm-up + cosine decay
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

        best = self.evaluate(val_dl)
        print(f"epoch 0 (OG CLIP on gray): val MSE {best:.5f}")
        best_ep, bad = 0, 0
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr = self._epoch(train_dl, opt, sched)
            va = self.evaluate(val_dl)
            flag = ""
            bad = 0 if va < best - min_delta else bad + 1  
            if va < best:                                    
                best, best_ep, flag = va, ep, "  *saved*"
                self.save(out)
                
            print(f"epoch {ep:3d}  train {tr:.5f}  val {va:.5f}  gap {va - tr:+.5f}  ({time.time() - t0:.0f}s){flag}")
            if patience is not None and bad >= patience:
                print(f"early stop: no improvement for {patience} epochs (best epoch {best_ep})")
                break
        if not out.exists():
            print("warning: validation never improved on the OG CLIP, nothing was saved")
        print(f"best val MSE {best:.5f} at epoch {best_ep} -> {out}")
        return best

    def save(self, path: Path) -> None:
        torch.save(self.student.net.state_dict(), path)


class ColorHead(nn.Module):
    """1x1 conv on 1/4-resolution (64x64 for a 256 image) decoder features -> logits over the 313 ab bins (at training only)."""

    def __init__(self, in_ch: int = 512, n_bins: int = 313):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_bins, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.conv(feat.float())  # decoder features are bf16; the head's weights are fp32


class SDUNet(nn.Module):
    """
    Gray L -> predicted ab, with a single frozen-UNet pass (+ LoRA) at a fixed timestep.
    L --VAE enc--> z_L --UNet(cat[z_L, z_L], t=999, [gray-CLIP | text])--> x0 --VAE dec (+skips)--> rgb -> ab
    """

    def __init__(self, conf: ModelConfig | None = None):
        super().__init__()
        self.conf= conf

        tok = CLIPTokenizer.from_pretrained(self.conf.sd_id, subfolder="tokenizer")
        txt = CLIPTextModel.from_pretrained(self.conf.sd_id, subfolder="text_encoder")
        ids = tok([self.conf.prompt], padding="max_length", truncation=True, return_tensors="pt").input_ids
        with torch.no_grad():
            self.register_buffer("text_emb", txt(ids)[0].to(torch.bfloat16), persistent=False)
        del txt

        sched = DDPMScheduler.from_pretrained(self.conf.sd_id, subfolder="scheduler")
        self.register_buffer("alpha_bar", sched.alphas_cumprod[self.conf.timestep].float(), persistent=False)   # moves with .to_device()

        self.gray_clip = GrayCLIP(self.conf.clip_id, torch.bfloat16)

        if self.conf.clip_ckpt:
            # this will have the checkpoint from distclip
            self.gray_clip.net.load_state_dict(torch.load(self.conf.clip_ckpt, map_location="cpu"))

        self.gray_clip.requires_grad_(False).eval()

        self.vae = AutoencoderKL.from_pretrained(self.conf.sd_id, subfolder="vae", torch_dtype=torch.bfloat16).requires_grad_(False)
        self.sf = self.vae.config.scaling_factor    # 0.18215 SD 1.x; TODO: explore SDXL/SD3

        # I'll test inserting adapter only for decoder (color producer), later will test with encoder too.
        inject_adapter_in_model(
            LoraConfig(r=self.conf.vae_rank, lora_alpha=self.conf.vae_rank, init_lora_weights="gaussian",
                       target_modules=["conv1", "conv2", "to_q", "to_k", "to_v", "to_out.0"]),
            self.vae.decoder)   # target attn layer and conv1-2
        
        self.skip_convs = nn.ModuleList(nn.Conv2d(i, o, 1) for i, o in ((512, 512), (256, 512), (128, 512), (128, 256)))
        for m in self.skip_convs:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        self.unet = UNet2DConditionModel.from_pretrained(self.conf.sd_id, subfolder="unet", torch_dtype=torch.bfloat16).requires_grad_(False)
        self.unet.add_adapter(LoraConfig(r=self.conf.unet_rank, lora_alpha=self.conf.unet_rank, init_lora_weights="gaussian",
                                         target_modules=["to_k", "to_q", "to_v", "to_out.0"]))  # basically target the attn layers only
        if self.conf.grad_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.color_head = ColorHead(512, self.conf.n_bins)

        # set lora params as trainable and cast to fp32
        for n, p in self.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()
                p.requires_grad_(True)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        squeezing gray image, returns mean and sharp notes for skips
        """
        enc = self.vae.encoder
        h, acts = enc.conv_in(x), []
        for blk in enc.down_blocks: # 4 blocks
            acts.append(h)  # for skip conn
            h = blk(h)

        h = enc.conv_out(enc.conv_act(enc.conv_norm_out(enc.mid_block(h))))
        mean, _ = self.vae.quant_conv(h).chunk(2, dim=1)
        return mean, acts

    def decode(self, z: torch.Tensor, acts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decoder + skips -> rebuild the OG image from Unet preds
        """
        dec = self.vae.decoder
        h = dec.mid_block(dec.conv_in(self.vae.post_quant_conv(z)), None)

        skips, feat = acts[::-1], None
        for i, blk in enumerate(dec.up_blocks):
            h = h + self.skip_convs[i](skips[i])
            h = checkpoint(blk, h, None, use_reentrant=False) if (self.training and self.conf.grad_checkpointing) \
                else blk(h, None)
            if i == 0:
                feat = h 

        return dec.conv_out(dec.conv_act(dec.conv_norm_out(h))), feat

    def forward(self, L: torch.Tensor) -> dict[str, torch.Tensor]:

        B = L.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=L.is_cuda):
            # prep for VAE; VAE expects 3 channels (RGB) thus the repeat
            x = norm_L(L).repeat(1, 3, 1, 1).to(torch.bfloat16)
            z, acts = self.encode(x)
            z = z * self.sf # scaling latent to what UNet expects

            ctx = torch.cat([self.gray_clip(L).to(z.dtype), self.text_emb.expand(B, -1, -1)], dim=1)
            t = torch.full((B,), self.conf.timestep, device=L.device, dtype=torch.long)
            eps = self.unet(torch.cat([z, z], dim=1), t, ctx, return_dict=False)[0]

            a = self.alpha_bar
            x0 = (z.float() - (1 - a).sqrt() * eps.float()) / a.sqrt()  # one DDPM step to x0
            img, feat = self.decode((x0 / self.sf).to(z.dtype), acts)

        rgb = ((img.float() + 1) / 2).clamp(0, 1)
        ab = kc.rgb_to_lab(rgb)[:, 1:3]
        return {"ab": ab, "rgb": rgb, "feat": feat}

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in names}

    def save(self, path) -> None:
        torch.save(self.trainable_state_dict(), path)

    def load(self, path) -> None:
        missing = self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys


