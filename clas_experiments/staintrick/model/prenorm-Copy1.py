from torch import nn
import torch
from kornia.color import lab_to_rgb, rgb_to_lab
from einops import repeat


class LabPreNorm(nn.Module):
    def __init__(
        self,
        model=None,
        device=None,
        epsilon=1e-7,
        requires_grad=True,
    ):
        super(LabPreNorm, self).__init__()
        self.epsilon = epsilon

        self.mu0 = torch.tensor([64, 20, -10]).to(device)
        self.sigma0 = torch.tensor([12, 7, 5]).to(device)

        self.mu = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.sigma = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.model = model if model is not None else nn.Identity()

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), f"image should be scaled to [0,1] rather than [0,256], current scale {x.min()}-{x.max()}"

        x = rgb_to_lab(x)

        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W)
        sigma = repeat(sigma + self.epsilon, "b c -> b c h w", h=H, w=W)
        
        mu_prime = repeat(self.mu + self.mu0, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime = repeat(self.sigma + self.sigma0, "c -> b c h w", b=B, h=H, w=W)

        x = (x - mu) / sigma * sigma_prime + mu_prime
        x_norm = lab_to_rgb(x)

        return self.model(x_norm),x_norm


class LabEMAPreNorm(nn.Module):
    def __init__(
        self,
        model=None,
        device=None,
        lmbd=0,
        epsilon=1e-7,
    ):
        super(LabEMAPreNorm, self).__init__()
        self.epsilon = epsilon
        self.lmbd = lmbd

        self.mu = torch.tensor([64, 20, -10]).to(device)
        self.sigma = torch.tensor([12, 7, 5]).to(device)
        self.model = model if model is not None else nn.Identity()

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), f"image should be scaled to [0,1] rather than [0,256], current scale {x.min()}-{x.max()}"

        x = rgb_to_lab(x)

        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        self.mu = (1-self.lmbd) * self.mu + self.lmbd * mu.mean(axis=0)
        self.sigma = (1-self.lmbd) * self.sigma + self.lmbd * sigma.mean(axis=0)
        
        mu = repeat(mu, "b c -> b c h w", h=H, w=W)
        sigma = repeat(sigma + self.epsilon, "b c -> b c h w", h=H, w=W)
        
        mu_prime = repeat(self.mu, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime = repeat(self.sigma, "c -> b c h w", b=B, h=H, w=W)

        x = (x - mu) / sigma * sigma_prime + mu_prime
        x = lab_to_rgb(x)

        return self.model(x)

if __name__ == "__main__":
    import torch

    lab_norm = LabEMAPreNorm()
    x = torch.randn((1, 3, 224, 224)).clip(min=0, max=1)
    y = lab_norm(x)
    print((y - x).max())
