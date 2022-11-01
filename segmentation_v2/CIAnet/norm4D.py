import torch
import torch.nn as nn

# mean_ = (0.485, 0.456, 0.406)
# std_ = (0.229, 0.224, 0.225)

def norm4D(image: torch.Tensor,mean_,std_) -> torch.Tensor:

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    x1: torch.Tensor = image[..., 0, :, :]
    x2: torch.Tensor = image[..., 1, :, :]
    x3: torch.Tensor = image[..., 2, :, :]

    y1: torch.Tensor = std_[0] * x1 + mean_[0]
    y2: torch.Tensor = std_[1] * x2 + mean_[1]
    y3: torch.Tensor = std_[2] * x3 + mean_[2]

    out: torch.Tensor = torch.stack([y1,y2,y3], -3)

    return out


def unorm4D(image: torch.Tensor,mean_,std_) -> torch.Tensor:

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    x1: torch.Tensor = image[..., 0, :, :]
    x2: torch.Tensor = image[..., 1, :, :]
    x3: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = (1/std_[0]) * (x1 - mean_[0])
    y: torch.Tensor = (1/std_[1]) * (x2 - mean_[1])
    z: torch.Tensor = (1/std_[2]) * (x3 - mean_[2])

    out: torch.Tensor = torch.stack([x, y, z], -3)

    return out


class Norm4D(nn.Module):

    def forward(self, image: torch.Tensor,mean_,std_) -> torch.Tensor:
        return norm4D(image,mean_,std_)
class Unorm4D(nn.Module):

    def forward(self, image: torch.Tensor,mean_,std_) -> torch.Tensor:
        return unorm4D(image,mean_,std_)


