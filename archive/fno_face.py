"""Fourier Neural Operator (FNO) face-embedding model + pickle helpers."""
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.out_ch, self.m1, self.m2 = out_ch, modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))

    @staticmethod
    def _mul(x, w):
        return torch.einsum("bixy,ioxy->boxy", x, torch.view_as_complex(w))

    def forward(self, x):
        B, _, H, W = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):  # FFT must stay float32
            x = x.float()
            xf = torch.fft.rfft2(x)
            out = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            out[:, :, : self.m1, : self.m2] = self._mul(xf[:, :, : self.m1, : self.m2], self.w1)
            out[:, :, -self.m1:, : self.m2] = self._mul(xf[:, :, -self.m1:, : self.m2], self.w2)
            return torch.fft.irfft2(out, s=(H, W))


class FNOBlock(nn.Module):
    def __init__(self, width, m1, m2):
        super().__init__()
        self.spec = SpectralConv2d(width, width, m1, m2)
        self.skip = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x):
        return F.gelu(self.norm(self.spec(x) + self.skip(x)))


class FNOFace(nn.Module):
    """Image (3xHxW) -> L2-normalised embedding. FNO layers operate on a
    downsampled grid; a strided conv stem keeps CPU time well under 1s."""

    def __init__(self, emb_dim=512, width=64, modes=12, depth=4, in_size=112):
        super().__init__()
        self.cfg = dict(emb_dim=emb_dim, width=width, modes=modes, depth=depth, in_size=in_size)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, width - 2, 3, 2, 1), nn.GroupNorm(2, width - 2), nn.GELU(),
        )  # 112 -> 28; +2 coordinate channels appended in forward
        self.blocks = nn.ModuleList([FNOBlock(width, modes, modes) for _ in range(depth)])
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width * 7 * 7, emb_dim), nn.BatchNorm1d(emb_dim))

    def forward(self, x):
        x = self.stem(x)
        B, _, H, W = x.shape
        gy = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        gx = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x = torch.cat([x, gx, gy], 1)
        for b in self.blocks:
            x = b(x)
        x = F.adaptive_avg_pool2d(x, 7)
        return F.normalize(self.head(x), dim=1)


def save_pkl(model, path):
    """Pickle config + weights (numpy) so the .pkl loads without custom classes."""
    state = {k: (v.cpu().numpy().astype(np.float16) if v.is_floating_point() else v.cpu().numpy())
             for k, v in model.state_dict().items()}  # fp16 halves file size
    with open(path, "wb") as f:
        pickle.dump({"arch": "FNOFace", "cfg": model.cfg, "state_dict": state}, f)


def load_pkl(path):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    m = FNOFace(**blob["cfg"])
    m.load_state_dict({k: (torch.from_numpy(v).float() if v.dtype == np.float16 else torch.from_numpy(v))
                       for k, v in blob["state_dict"].items()})
    return m.eval()


def save_pth(model, path):
    """Standard PyTorch checkpoint (.pth): config + full-precision float32 weights."""
    torch.save({"arch": "FNOFace", "cfg": model.cfg,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)


def load_pth(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    m = FNOFace(**blob["cfg"])
    m.load_state_dict(blob["state_dict"])
    return m.eval()