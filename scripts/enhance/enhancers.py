# scripts/enhance/enhancers.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB [0..255] -> torch float BCHW in [0,1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 4:  # RGBA -> RGB
        arr = arr[..., :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def _to_image(t: torch.Tensor) -> Image.Image:
    """torch float BCHW [0,1] -> PIL RGB [0..255]."""
    t = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    t = (t * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(t, mode="RGB")


class _ZeroDCE(nn.Module):
    """DCE-Net per Zero-DCE: 7 convs (3×3, stride 1, 32 ch), symmetric concats, Tanh → 24 maps."""

    def __init__(self, ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(ch * 2, ch, 3, 1, 1)  # cat(x3, x4)
        self.conv6 = nn.Conv2d(ch * 2, ch, 3, 1, 1)  # cat(x2, x5)
        self.conv7 = nn.Conv2d(ch * 2, 24, 3, 1, 1)  # cat(x1, x6) -> 24 (8*3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        r = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))  # (B,24,H,W)
        return r


def _apply_curve(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Iterative LE-curve (n=8):
      x_{k+1} = x_k + r_k * (x_k * (1 - x_k)), with per-step, per-channel maps r_k.
    x: BCHW in [0,1]; r: B(8*3)HW.
    """
    B, C, H, W = x.shape
    assert C == 3
    r = r.view(B, 8, 3, H, W)
    x_k = x
    for k in range(8):
        r_k = r[:, k, :, :, :]
        x_k = x_k + r_k * (x_k * (1.0 - x_k))
    return x_k.clamp(0.0, 1.0)


class ZeroDCEEnhancer:
    """
    Wrapper that loads a Zero-DCE checkpoint and exposes a simple enhance() method.
    """

    def __init__(self, weights: Path, device: str | None = None, half: bool = False):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = _ZeroDCE().to(self.device)

        # Load checkpoint (try safer weights_only if your torch supports it)
        try:
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(weights, map_location="cpu")

        state = ckpt.get("state_dict", ckpt)

        # Normalize keys: drop 'module.' and map official 'e_conv*' -> our 'conv*'
        remapped = {}
        for k, v in state.items():
            k2 = k
            if k2.startswith("module."):
                k2 = k2[len("module.") :]
            if k2.startswith("e_conv"):
                k2 = k2.replace("e_conv", "conv", 1)
            remapped[k2] = v

        self.model.load_state_dict(remapped, strict=True)

        self.model.eval()
        self.half = half and self.device.type == "cuda"
        if self.half:
            self.model.half()

    @torch.inference_mode()
    def enhance_pil(
        self, img: Image.Image, tile: Tuple[int, int] | None = None
    ) -> Image.Image:
        """
        Enhance a PIL RGB image. Optional tiling (h, w) for very large images (keeps geometry).
        """
        if tile is None:
            t = _to_tensor(img).to(self.device)
            if self.half:
                t = t.half()
            r = self.model(t)
            if self.half:
                r = r.float()
            out = _apply_curve(t.float(), r.float())
            return _to_image(out)
        else:
            th, tw = tile
            W, H = img.size
            out = Image.new("RGB", (W, H))
            for y in range(0, H, th):
                for x in range(0, W, tw):
                    crop = img.crop((x, y, min(x + tw, W), min(y + th, H)))
                    out_crop = self.enhance_pil(crop, tile=None)
                    out.paste(out_crop, (x, y))
            return out
