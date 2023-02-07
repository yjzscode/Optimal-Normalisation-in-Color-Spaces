from torch import nn
import torch
from kornia.color import lab_to_rgb, rgb_to_lab
from einops import repeat
import random


class LabRandNorm(nn.Module):
    def __init__(
        self,
        model=None,
        device=None,
        idx=None,
        epsilon=1e-7,
        ep1_scale=1,
        ep2_scale=1,
    ):
        super(LabRandNorm, self).__init__()
        self.idx = idx
        self.device = device
        self.epsilon = epsilon
        self.model = model if model is not None else nn.Identity()
        self.ep1_scale = ep1_scale
        self.ep2_scale = ep2_scale

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), f"image should be scaled to [0,1] rather than [0,256], current scale {x.min()}-{x.max()}"
        
        if not self.training:
            return self.model(x)
        
        B, _, H, W = x.shape

        if self.idx == None:
            idx = random.randint(0, B - 1)
        else:
            idx = self.idx

        x = rgb_to_lab(x)

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        ep1 = torch.randn(1).to(self.device) * self.ep1_scale
        ep2 = torch.randn(1).to(self.device) * self.ep2_scale

        mu_t = repeat(mu[idx, ...] + ep1, "c -> b c h w", b=B, h=H, w=W)
        sigma_t = repeat(
            sigma[idx, ...] + self.epsilon + ep2, "c -> b c h w", b=B, h=H, w=W
        )

        mu_s = repeat(mu, "b c -> b c h w", h=H, w=W)
        sigma_s = repeat(sigma, "b c -> b c h w", h=H, w=W)

        x = (x - mu_s) / sigma_s * sigma_t + mu_t
        x = lab_to_rgb(x)

        return self.model(x)


if __name__ == "__main__":
    import torch

    lab_norm = LabRandNorm()
    x = torch.randn((10, 3, 224, 224)).clip(min=0, max=1)
    y = lab_norm(x)
    print((y - x).max())
