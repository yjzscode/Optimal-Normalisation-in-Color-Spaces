import torch
import torch.nn as nn


def rgb_to_hed(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    h: torch.Tensor = 1.88 * r -0.07 * g -0.60 * b
    e: torch.Tensor = -1.02 * r + 1.13 * g -0.48 * b
    d: torch.Tensor = -0.55 * r -0.13 * g + 1.57 * b

    out: torch.Tensor = torch.stack([h, e, d], -3)

    return out


def hed_to_rgb(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :]
    e: torch.Tensor = image[..., 1, :, :]
    d: torch.Tensor = image[..., 2, :, :]

    r: torch.Tensor = 0.65*h+0.07*e+0.27*d
    g: torch.Tensor = 0.71*h+0.99*e+0.57*d
    b: torch.Tensor = 0.29*h+0.11*e+0.78*d

    out: torch.Tensor = torch.stack([r, g, b], dim=-3)

    return out


class RgbToHed(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_hed(image)


class HedToRgb(nn.Module):

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return hed_to_rgb(image)