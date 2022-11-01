import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader,TensorDataset
from os import listdir
import os
from PIL import Image
from torchvision import transforms
from CIAnet import *
import scipy.io
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import random
# from Dataset import ciaData
import cv2 as cv #8.3添加
from kornia.color import lab_to_rgb,rgb_to_lab,hsv_to_rgb,rgb_to_hsv#8.28添加
from hed import hed_to_rgb,rgb_to_hed#9.11
from norm4D import norm4D,unorm4D
from einops import repeat

class bw_1(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(bw_1, self).__init__()
        self.epsilon = epsilon
    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_lab(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)
        
        mean = repeat(mean_1, "c -> b c h w", b=B, h=H, w=W).to(device)
        std = repeat(std_1, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = (x - mu) / sigma * std + mean
        x = lab_to_rgb(x)
        x = unorm4D(x,mean_,std_)
        return x

    
class bw_2(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(bw_2, self).__init__()
        self.epsilon = epsilon
    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_lab(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)
        
        mean = repeat(mean_2, "c -> b c h w", b=B, h=H, w=W).to(device)
        std = repeat(std_2, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = (x - mu) / sigma * std + mean
        x = lab_to_rgb(x)
        x = unorm4D(x,mean_,std_)

        return x

    
class bw_3(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(bw_3, self).__init__()
        self.epsilon = epsilon
    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_lab(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)
        
        mean = repeat(mean_3, "c -> b c h w", b=B, h=H, w=W).to(device)
        std = repeat(std_3, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = (x - mu) / sigma * std + mean
        x = lab_to_rgb(x)
        x = unorm4D(x,mean_,std_)

        return x
    
class bw_4(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(bw_4, self).__init__()
        self.epsilon = epsilon
    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_lab(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)
        
        mean = repeat(mean_4, "c -> b c h w", b=B, h=H, w=W).to(device)
        std = repeat(std_4, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = (x - mu) / sigma * std + mean
        x = lab_to_rgb(x)
        x = unorm4D(x,mean_,std_)

        return x

class bw_5(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(bw_5, self).__init__()
        self.epsilon = epsilon
    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_lab(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)
        
        mean = repeat(mean_5, "c -> b c h w", b=B, h=H, w=W).to(device)
        std = repeat(std_5, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = (x - mu) / sigma * std + mean
        x = lab_to_rgb(x)
        x = unorm4D(x,mean_,std_)

        return x
    
class bw_mean(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(bw_mean, self).__init__()
        self.epsilon = epsilon
    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_lab(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)
        
        mean = repeat(mean_lab, "c -> b c h w", b=B, h=H, w=W).to(device)
        std = repeat(std_lab, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = (x - mu) / sigma * std + mean
        x = lab_to_rgb(x)
        x = unorm4D(x,mean_,std_)

        return x

    
class bw_lab_norigin(nn.Module): #without origin
    def __init__(self, epsilon=1e-7):
        super(bw_lab_norigin, self).__init__()
        self.epsilon = epsilon
        self.sigma = nn.Parameter(
            torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)
        )
        self.mu = nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)
        )

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_lab(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime = repeat(self.mu, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime = repeat(self.sigma, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = (x - mu) / sigma * sigma_prime + mu_prime
        x = lab_to_rgb(x)
        x = unorm4D(x,mean_,std_)

        return x

class bw_lab_2(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(bw_lab_2, self).__init__()
        self.epsilon = epsilon
        self.sigma = nn.Parameter(
            torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)
        )
        self.mu = nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)
        )

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_lab(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime = repeat(self.mu, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime = repeat(self.sigma, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean = repeat(mean_2, "c -> b c h w", b=B, h=H, w=W).to(device)
        std = repeat(std_2, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = ((x - mu) / sigma * sigma_prime + mu_prime)*std+mean
        x = lab_to_rgb(x)
        x = unorm4D(x,mean_,std_)
        return x    

class bw_lab_3(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(bw_lab_3, self).__init__()
        self.epsilon = epsilon
        self.sigma = nn.Parameter(
            torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)
        )
        self.mu = nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)
        )

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_lab(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime = repeat(self.mu, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime = repeat(self.sigma, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean = repeat(mean_3, "c -> b c h w", b=B, h=H, w=W).to(device)
        std = repeat(std_3, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = ((x - mu) / sigma * sigma_prime + mu_prime)*std+mean
        x = lab_to_rgb(x)
        x = unorm4D(x,mean_,std_)

        return x    

    
class bw_lab(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(bw_lab, self).__init__()
        self.epsilon = epsilon
        self.sigma = nn.Parameter(
            torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)
        )
        self.mu = nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)
        )

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_lab(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime = repeat(self.mu, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime = repeat(self.sigma, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean = repeat(mean_lab, "c -> b c h w", b=B, h=H, w=W).to(device)
        std = repeat(std_lab, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = ((x - mu) / sigma * sigma_prime + mu_prime)*std+mean
        x = lab_to_rgb(x)
        x = unorm4D(x,mean_,std_)

        return x    
    
class bw_hsv(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(bw_hsv, self).__init__()
        self.epsilon = epsilon
        self.sigma = nn.Parameter(
            torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)
        )
        self.mu = nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)
        )

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_hsv(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime = repeat(self.mu, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime = repeat(self.sigma, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean = repeat(mean_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)
        std = repeat(std_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = ((x - mu) / sigma * sigma_prime + mu_prime)*std+mean
        x = hsv_to_rgb(x)
        x = unorm4D(x,mean_,std_)

        return x 
    
class bw_hed(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(bw_hed, self).__init__()
        self.epsilon = epsilon
        self.sigma = nn.Parameter(
            torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad=True)
        )
        self.mu = nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)
        )

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        
        x = rgb_to_hed(x)
        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W).to(device)
        sigma = repeat(sigma+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime = repeat(self.mu, "c -> b c h w", b=B, h=H, w=W).to(device).to(device)
        sigma_prime = repeat(self.sigma, "c -> b c h w", b=B, h=H, w=W).to(device).to(device)
        
        mean = repeat(mean_hed, "c -> b c h w", b=B, h=H, w=W).to(device)
        std = repeat(std_hed, "c -> b c h w", b=B, h=H, w=W).to(device)

        x = ((x - mu) / sigma * sigma_prime + mu_prime)*std+mean
        x = hed_to_rgb(x)
        x = unorm4D(x,mean_,std_)

        return x    
    
    
#9.22    
class bw_concat(nn.Module):  #8.25
    def __init__(self):            
        super(bw_concat, self).__init__()
        self.epsilon = epsilon
        self.weight_lab=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_lab=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hsv=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hsv=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hed=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hed=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        #lab
        x_lab = rgb_to_lab(x)
        B, _, H, W = x_lab.shape

        mu_lab = x_lab.mean(axis=(2, 3))
        sigma_lab = x_lab.std(axis=(2, 3))

        mu_lab = repeat(mu_lab, "b c -> b c h w", h=H, w=W).to(device)
        sigma_lab = repeat(sigma_lab+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime_lab = repeat(self.bias_lab, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime_lab = repeat(self.weight_lab, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean_lab = repeat(mean_lab, "c -> b c h w", b=B, h=H, w=W).to(device)
        std_lab = repeat(std_lab, "c -> b c h w", b=B, h=H, w=W).to(device)

        x_lab = ((x_lab - mu_lab) / sigma_lab * sigma_prime_lab + mu_prime_lab)*std_lab+mean_lab
        x_lab = lab_to_rgb(x_lab)
        
        #hsv
        x_hsv = rgb_to_hsv(x)
        B, _, H, W = x_hsv.shape

        mu_hsv = x_hsv.mean(axis=(2, 3))
        sigma_hsv = x_hsv.std(axis=(2, 3))

        mu_hsv = repeat(mu_hsv, "b c -> b c h w", h=H, w=W).to(device)
        sigma_hsv = repeat(sigma_hsv+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime_hsv = repeat(self.bias_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime_hsv = repeat(self.weight_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean_hsv = repeat(mean_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)
        std_hsv = repeat(std_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)

        x_hsv = ((x_hsv - mu_hsv) / sigma_hsv * sigma_prime_hsv + mu_prime_hsv)*std_hsv+mean_hsv
        x_hsv = hsv_to_rgb(x_hsv)
        
        #hed
        x_hed = rgb_to_hed(x)
        B, _, H, W = x_hed.shape

        mu_hed = x_hed.mean(axis=(2, 3))
        sigma_hed = x_hed.std(axis=(2, 3))

        mu_hed = repeat(mu_hed, "b c -> b c h w", h=H, w=W).to(device)
        sigma_hed = repeat(sigma_hed+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime_hed = repeat(self.bias_hed, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime_hed = repeat(self.weight_hed, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean_hed = repeat(mean_hed, "c -> b c h w", b=B, h=H, w=W).to(device)
        std_hed = repeat(std_hed, "c -> b c h w", b=B, h=H, w=W).to(device)

        x_hed = ((x_hed - mu_hed) / sigma_hed * sigma_prime_hed + mu_prime_hed)*std_hed+mean_hed
        x_hed = hed_to_rgb(x_hed)
        x = torch.cat([x_lab,x_hsv,x_hed],1)
        x = unorm4D(x,mean_,std_)
        
        return x
    
class bw_avg(nn.Module):  #8.25
    def __init__(self):            
        super(bw_avg, self).__init__()
        self.epsilon = epsilon
        self.weight_lab=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_lab=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hsv=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hsv=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hed=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hed=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
        
    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        #lab
        x_lab = rgb_to_lab(x)
        B, _, H, W = x_lab.shape

        mu_lab = x_lab.mean(axis=(2, 3))
        sigma_lab = x_lab.std(axis=(2, 3))

        mu_lab = repeat(mu_lab, "b c -> b c h w", h=H, w=W).to(device)
        sigma_lab = repeat(sigma_lab+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime_lab = repeat(self.bias_lab, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime_lab = repeat(self.weight_lab, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean_lab = repeat(mean_lab, "c -> b c h w", b=B, h=H, w=W).to(device)
        std_lab = repeat(std_lab, "c -> b c h w", b=B, h=H, w=W).to(device)

        x_lab = ((x_lab - mu_lab) / sigma_lab * sigma_prime_lab + mu_prime_lab)*std_lab+mean_lab
        x_lab = lab_to_rgb(x_lab)
        
        #hsv
        x_hsv = rgb_to_hsv(x)
        B, _, H, W = x_hsv.shape

        mu_hsv = x_hsv.mean(axis=(2, 3))
        sigma_hsv = x_hsv.std(axis=(2, 3))

        mu_hsv = repeat(mu_hsv, "b c -> b c h w", h=H, w=W).to(device)
        sigma_hsv = repeat(sigma_hsv+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime_hsv = repeat(self.bias_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime_hsv = repeat(self.weight_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean_hsv = repeat(mean_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)
        std_hsv = repeat(std_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)

        x_hsv = ((x_hsv - mu_hsv) / sigma_hsv * sigma_prime_hsv + mu_prime_hsv)*std_hsv+mean_hsv
        x_hsv = hsv_to_rgb(x_hsv)
        
        #hed
        x_hed = rgb_to_hed(x)
        B, _, H, W = x_hed.shape

        mu_hed = x_hed.mean(axis=(2, 3))
        sigma_hed = x_hed.std(axis=(2, 3))

        mu_hed = repeat(mu_hed, "b c -> b c h w", h=H, w=W).to(device)
        sigma_hed = repeat(sigma_hed+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime_hed = repeat(self.bias_hed, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime_hed = repeat(self.weight_hed, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean_hed = repeat(mean_hed, "c -> b c h w", b=B, h=H, w=W).to(device)
        std_hed = repeat(std_hed, "c -> b c h w", b=B, h=H, w=W).to(device)

        x_hed = ((x_hed - mu_hed) / sigma_hed * sigma_prime_hed + mu_prime_hed)*std_hed+mean_hed
        x_hed = hed_to_rgb(x_hed)
        x = (x_lab + x_hsv + x_hed)/3
        x = unorm4D(x,mean_,std_)
        return x
    
    
class bw_weighted_avg(nn.Module):  #8.25
    def __init__(self):            
        super(bw_weighted_avg, self).__init__()
        self.epsilon = epsilon
        self.weight_lab=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_lab=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hsv=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hsv=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        self.weight_hed=nn.Parameter(torch.tensor([1,1,1], dtype=torch.float32, requires_grad=True,device=device))
        self.bias_hed=nn.Parameter(torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True,device=device))
        
        self.w1=nn.Parameter(torch.tensor([0.33,0.33,0.33], dtype=torch.float32, requires_grad=True,device=device))
        self.w2=nn.Parameter(torch.tensor([0.33,0.33,0.33], dtype=torch.float32, requires_grad=True,device=device))              
        
    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), "image should be scaled to [0,1] rather than [0,256]"
        #lab
        x_lab = rgb_to_lab(x)
        B, _, H, W = x_lab.shape

        mu_lab = x_lab.mean(axis=(2, 3))
        sigma_lab = x_lab.std(axis=(2, 3))

        mu_lab = repeat(mu_lab, "b c -> b c h w", h=H, w=W).to(device)
        sigma_lab = repeat(sigma_lab+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime_lab = repeat(self.bias_lab, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime_lab = repeat(self.weight_lab, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean_lab = repeat(mean_lab, "c -> b c h w", b=B, h=H, w=W).to(device)
        std_lab = repeat(std_lab, "c -> b c h w", b=B, h=H, w=W).to(device)

        x_lab = ((x_lab - mu_lab) / sigma_lab * sigma_prime_lab + mu_prime_lab)*std_lab+mean_lab
        x_lab = lab_to_rgb(x_lab)
        
        #hsv
        x_hsv = rgb_to_hsv(x)
        B, _, H, W = x_hsv.shape

        mu_hsv = x_hsv.mean(axis=(2, 3))
        sigma_hsv = x_hsv.std(axis=(2, 3))

        mu_hsv = repeat(mu_hsv, "b c -> b c h w", h=H, w=W).to(device)
        sigma_hsv = repeat(sigma_hsv+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime_hsv = repeat(self.bias_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime_hsv = repeat(self.weight_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean_hsv = repeat(mean_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)
        std_hsv = repeat(std_hsv, "c -> b c h w", b=B, h=H, w=W).to(device)

        x_hsv = ((x_hsv - mu_hsv) / sigma_hsv * sigma_prime_hsv + mu_prime_hsv)*std_hsv+mean_hsv
        x_hsv = hsv_to_rgb(x_hsv)
        
        #hed
        x_hed = rgb_to_hed(x)
        B, _, H, W = x_hed.shape

        mu_hed = x_hed.mean(axis=(2, 3))
        sigma_hed = x_hed.std(axis=(2, 3))

        mu_hed = repeat(mu_hed, "b c -> b c h w", h=H, w=W).to(device)
        sigma_hed = repeat(sigma_hed+self.epsilon, "b c -> b c h w", h=H, w=W).to(device)

        mu_prime_hed = repeat(self.bias_hed, "c -> b c h w", b=B, h=H, w=W).to(device)
        sigma_prime_hed = repeat(self.weight_hed, "c -> b c h w", b=B, h=H, w=W).to(device)
        
        mean_hed = repeat(mean_hed, "c -> b c h w", b=B, h=H, w=W).to(device)
        std_hed = repeat(std_hed, "c -> b c h w", b=B, h=H, w=W).to(device)

        x_hed = ((x_hed - mu_hed) / sigma_hed * sigma_prime_hed + mu_prime_hed)*std_hed+mean_hed
        x_hed = hed_to_rgb(x_hed)
        
        w1 = repeat(self.w1, "c -> b c h w", b=B, h=H, w=W).to(device)
        w2 = repeat(self.w2, "c -> b c h w", b=B, h=H, w=W).to(device)
        I = repeat(1, "c -> b c h w", b=B, h=H, w=W).to(device)
               
        x = x_lab*w1 + x_hsv*w2 + x_hed*(I-w1-w2)
        x = unorm4D(x,mean_,std_)
        return x