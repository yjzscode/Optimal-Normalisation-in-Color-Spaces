from torch import nn
import torch
from kornia.color import lab_to_rgb, rgb_to_lab, hsv_to_rgb, rgb_to_hsv
from einops import repeat

import numpy as np
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class SelfAttention(nn.Module):
    
    def __init__(self, in_dim=3, activation=F.relu,device=None):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//3 , kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//3 , kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)

        n=8
        self.f_pool = nn.MaxPool2d(n, stride=n)
        self.g_pool = nn.MaxPool2d(n, stride=n)
        self.upsample = nn.Upsample(scale_factor=n, mode='bilinear')

        self.softmax  = nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        n=8
        m_batchsize, C, width, height = x.size()
        
        f = self.f(x) # B * (C//3) * W * H
        f = self.f_pool(f).view(m_batchsize, -1, width//n * height//n)  # B * (C//3) * (W/8 * H/8)
        g = self.g(x) # B * (C//3) * W * H
        g = self.g_pool(g).view(m_batchsize, -1, width//n * height//n) # B * (C//3) * (W/8 * H/8)
        
        h = self.h(x)
        h = self.g_pool(h).view(m_batchsize, -1, width//n * height//n) # B * C * (W * H)
        
        attention = torch.bmm(f.permute(0, 2, 1), g).view(m_batchsize, width//n * height//n, width//n * height//n) # B * (W/8 * H/8) * (W/8 * H/8)
        attention = self.softmax(attention) # 归一化
        
        self_attetion = torch.bmm(h, attention).view(m_batchsize, C, width//n, height//n) # B * C * (W//8 * H//8)
        out = self.upsample(self_attetion) 
#         print(out.size())
        
        return out
    
# Stain normalization using templates directly
class TemplateNorm(nn.Module):
    def __init__(
        self,
        model=None,
        device=None,
        epsilon=1e-7,
        requires_grad=False,
        
        mu0 = [64, 20, -10],
        sigma0 = [12, 7, 5],
        
    ):
        super(TemplateNorm, self).__init__()
        self.epsilon = epsilon
        self.mu0 = torch.tensor(mu0).to(device)
        self.sigma0 = torch.tensor(sigma0).to(device)
        self.model = model if model is not None else nn.Identity()
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

        self.mu_lab = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.sigma_lab = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.mu_hsv = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.sigma_hsv = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )

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
        
        mu_prime = repeat(self.mu0, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime = repeat(self.sigma0, "c -> b c h w", b=B, h=H, w=W)

        x = (x - mu) / sigma * sigma_prime + mu_prime
        x_norm = lab_to_rgb(x)

        return self.model(x_norm),x_norm

# An example of LStainNorm in sole color space
class LabPreNorm(nn.Module):
    def __init__(
        self,
        model=None,
        device=None,
        epsilon=1e-7,
        requires_grad=True,
        
#         mu0 = None,
#         sigma0 = None,
        
        mu0 = [64, 20, -10],
        sigma0 = [12, 7, 5],
        
    ):
        super(LabPreNorm, self).__init__()
        self.epsilon = epsilon
        self.mu0 = torch.tensor(mu0).to(device)
        self.sigma0 = torch.tensor(sigma0).to(device)

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

        self.mu_lab = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.sigma_lab = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.mu_hsv = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.sigma_hsv = nn.Parameter(
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

# The final model: color space conversion and fusion   
class SA3(nn.Module):
    def __init__(
        self,
        model=None,
        device=None,
        epsilon=1e-7,
        requires_grad=True,
        
        mu0_lab = [64, 20, -10],
        sigma0_lab = [12, 7, 5],
        mu0_hsv = [64, 20, -10],
        sigma0_hsv = [12, 7, 5],
        mu0_rgb = [64, 20, -10],
        sigma0_rgb = [12, 7, 5],

        gamma = [1,1,1],
        
    ):
        super(SA3, self).__init__()
        self.epsilon = epsilon
        
        self.mu0_lab = torch.tensor(mu0_lab).to(device)
        self.sigma0_lab = torch.tensor(sigma0_lab).to(device)
        self.mu0_hsv = torch.tensor(mu0_hsv).to(device)
        self.sigma0_hsv = torch.tensor(sigma0_hsv).to(device)
        self.mu0_rgb = torch.tensor(mu0_rgb).to(device)
        self.sigma0_rgb = torch.tensor(sigma0_rgb).to(device)
        
        self.gamma = torch.tensor(gamma).to(device)

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
        self.mu_lab = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.sigma_lab = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.mu_hsv = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.sigma_hsv = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        
        self.mu_rgb = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        self.sigma_rgb = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        
        self.model = model if model is not None else nn.Identity()
        self.SelfAttention = SelfAttention()
        self.gamma_lab = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        
        self.gamma_hsv = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        
        self.gamma_rgb = nn.Parameter(
            torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        
        self.a0 = nn.Parameter(
            torch.tensor(
                [1.0, 1.0, 1.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )
        
        self.b0 = nn.Parameter(
            torch.tensor(
                [1.0, 1.0, 1.0], dtype=torch.float32, requires_grad=requires_grad
            )
        )

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), f"image should be scaled to [0,1] rather than [0,256], current scale {x.min()}-{x.max()}"
        
        x_lab = rgb_to_lab(x)
        B, _, H, W = x_lab.shape
        mu_lab = x_lab.mean(axis=(2, 3))
        sigma_lab = x_lab.std(axis=(2, 3))
        mu_lab = repeat(mu_lab, "b c -> b c h w", h=H, w=W)
        sigma_lab = repeat(sigma_lab + self.epsilon, "b c -> b c h w", h=H, w=W)       
        mu_prime_lab = repeat(self.mu_lab + self.mu0_lab, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime_lab = repeat(self.sigma_lab + self.sigma0_lab, "c -> b c h w", b=B, h=H, w=W)
        x_lab = (x_lab - mu_lab) / sigma_lab * sigma_prime_lab + mu_prime_lab
        x_lab = torch.abs(lab_to_rgb(x_lab))
        
        x_hsv = torch.abs(rgb_to_hsv(x))
        B, _, H, W = x_hsv.shape
        mu_hsv = x_hsv.mean(axis=(2, 3))
        sigma_hsv = x_hsv.std(axis=(2, 3))
        mu_hsv = repeat(mu_hsv, "b c -> b c h w", h=H, w=W)
        sigma_hsv = repeat(sigma_hsv + self.epsilon, "b c -> b c h w", h=H, w=W)        
        mu_prime_hsv = repeat(self.mu_hsv + self.mu0_hsv, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime_hsv = repeat(self.sigma_hsv + self.sigma0_hsv, "c -> b c h w", b=B, h=H, w=W)
        x_hsv = (x_hsv - mu_hsv) / sigma_hsv * sigma_prime_hsv + mu_prime_hsv
        x_hsv = torch.abs(hsv_to_rgb(x_hsv))
        
        x_rgb = torch.abs(x)
        B, _, H, W = x_rgb.shape
        mu_rgb = x_rgb.mean(axis=(2, 3))
        sigma_rgb = x_rgb.std(axis=(2, 3))
        mu_rgb = repeat(mu_rgb, "b c -> b c h w", h=H, w=W)
        sigma_rgb = repeat(sigma_rgb + self.epsilon, "b c -> b c h w", h=H, w=W)        
        mu_prime_rgb = repeat(self.mu_rgb + self.mu0_rgb, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime_rgb = repeat(self.sigma_rgb + self.sigma0_rgb, "c -> b c h w", b=B, h=H, w=W)
        x_rgb = (x_rgb - mu_rgb) / sigma_rgb * sigma_prime_rgb + mu_prime_rgb
        x_rgb = torch.abs(x_rgb)
        

        
        lab_attention = self.SelfAttention(x_lab)
        hsv_attention = self.SelfAttention(x_hsv)
        rgb_attention = self.SelfAttention(x_rgb)
        
        gamma_lab = repeat(self.gamma+self.gamma_lab, "c -> b c h w", b=B, h=H, w=W)
        gamma_hsv = repeat(self.gamma+self.gamma_hsv, "c -> b c h w", b=B, h=H, w=W)
        gamma_rgb = repeat(self.gamma+self.gamma_rgb, "c -> b c h w", b=B, h=H, w=W)
        
        
        lab_attention = (gamma_lab * torch.abs(lab_attention) + x_lab)/(x_lab+self.epsilon)
        hsv_attention = (gamma_hsv * torch.abs(hsv_attention) + x_hsv)/(x_hsv+self.epsilon)
        rgb_attention = (gamma_rgb * torch.abs(rgb_attention) + x_rgb)/(x_rgb+self.epsilon)
        
        lab_attention_norm = lab_attention/(lab_attention + hsv_attention + rgb_attention + self.epsilon)
        hsv_attention_norm = hsv_attention/(lab_attention + hsv_attention + rgb_attention + self.epsilon)
        rgb_attention_norm = 1 - lab_attention_norm - hsv_attention_norm
        
        lab_weight = self.a0[0] * lab_attention_norm + self.b0[0]
        hsv_weight = self.a0[1] * hsv_attention_norm + self.b0[1]
        rgb_weight = self.a0[2] * rgb_attention_norm + self.b0[2]
        
        x_norm = (x_lab * lab_weight + x_hsv * hsv_weight + x_rgb * rgb_weight)/ (lab_weight + hsv_weight + rgb_weight)
        
        return self.model(x_norm),x_norm

if __name__ == "__main__":
    import torch

    lab_norm = LabPreNorm()
    x = torch.randn((1, 3, 224, 224)).clip(min=0, max=1)
    y = lab_norm(x)
    print((y - x).max())

    
    
