from torch import nn
import torch
from kornia.color import lab_to_rgb, rgb_to_lab, hsv_to_rgb, rgb_to_hsv
from einops import repeat
from model.hed import hed_to_rgb, rgb_to_hed


class TemplateNorm(nn.Module):
    def __init__(
        self,
        model=None,
        device=None,
        epsilon=1e-7,
        
#         mu0 = None,
#         sigma0 = None,
        
        mu0 = [64, 20, -10],
        sigma0 = [12, 7, 5],
        
    ):
        super(TemplateNorm, self).__init__()
        self.epsilon = epsilon
        self.mu0 = torch.tensor(mu0).to(device)
        self.sigma0 = torch.tensor(sigma0).to(device)
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
        
        mu_prime = repeat(self.mu0, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime = repeat(self.sigma0, "c -> b c h w", b=B, h=H, w=W)

        x = (x - mu) / sigma * sigma_prime + mu_prime
        x_norm = lab_to_rgb(x)

        return self.model(x_norm),x_norm

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
    
    
    
class HsvPreNorm(nn.Module):
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
        super(HsvPreNorm, self).__init__()
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
        self.model = model if model is not None else nn.Identity()

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), f"image should be scaled to [0,1] rather than [0,256], current scale {x.min()}-{x.max()}"

        x = rgb_to_hsv(x)

        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W)
        sigma = repeat(sigma + self.epsilon, "b c -> b c h w", h=H, w=W)
        
        mu_prime = repeat(self.mu + self.mu0, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime = repeat(self.sigma + self.sigma0, "c -> b c h w", b=B, h=H, w=W)

        x = (x - mu) / sigma * sigma_prime + mu_prime
        x_norm = hsv_to_rgb(x)

        return self.model(x_norm),x_norm
    
    
class HedPreNorm(nn.Module):
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
        super(HedPreNorm, self).__init__()
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
        self.model = model if model is not None else nn.Identity()

    def forward(self, x):
        assert (
            x.max() <= 1 and x.min() >= 0
        ), f"image should be scaled to [0,1] rather than [0,256], current scale {x.min()}-{x.max()}"

        x = rgb_to_hed(x)

        B, _, H, W = x.shape

        mu = x.mean(axis=(2, 3))
        sigma = x.std(axis=(2, 3))

        mu = repeat(mu, "b c -> b c h w", h=H, w=W)
        sigma = repeat(sigma + self.epsilon, "b c -> b c h w", h=H, w=W)
        
        mu_prime = repeat(self.mu + self.mu0, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime = repeat(self.sigma + self.sigma0, "c -> b c h w", b=B, h=H, w=W)

        x = (x - mu) / sigma * sigma_prime + mu_prime
        x_norm = hed_to_rgb(x)

        return self.model(x_norm),x_norm
    
class LabHsvAvg(nn.Module):
    def __init__(
        self,
        model=None,
        device=None,
        epsilon=1e-7,
        requires_grad=True,
        
#         mu0 = None,
#         sigma0 = None,
        
        mu0_lab = [64, 20, -10],
        sigma0_lab = [12, 7, 5],
        mu0_hsv = [64, 20, -10],
        sigma0_hsv = [12, 7, 5],
        
    ):
        super(LabHsvAvg, self).__init__()
        self.epsilon = epsilon
        self.mu0_lab = torch.tensor(mu0_lab).to(device)
        self.sigma0_lab = torch.tensor(sigma0_lab).to(device)
        self.mu0_hsv = torch.tensor(mu0_hsv).to(device)
        self.sigma0_hsv = torch.tensor(sigma0_hsv).to(device)

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
        x_lab = rgb_to_lab(x)

        B, _, H, W = x_lab.shape

        mu_lab = x_lab.mean(axis=(2, 3))
        sigma_lab = x_lab.std(axis=(2, 3))

        mu_lab = repeat(mu_lab, "b c -> b c h w", h=H, w=W)
        sigma_lab = repeat(sigma_lab + self.epsilon, "b c -> b c h w", h=H, w=W)
        
        mu_prime_lab = repeat(self.mu_lab + self.mu0_lab, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime_lab = repeat(self.sigma_lab + self.sigma0_lab, "c -> b c h w", b=B, h=H, w=W)

        x_lab = (x_lab - mu_lab) / sigma_lab * sigma_prime_lab + mu_prime_lab
        x_lab = lab_to_rgb(x_lab)
        
        x_hsv = rgb_to_hsv(x)

        B, _, H, W = x_hsv.shape

        mu_hsv = x_hsv.mean(axis=(2, 3))
        sigma_hsv = x_hsv.std(axis=(2, 3))

        mu_hsv = repeat(mu_hsv, "b c -> b c h w", h=H, w=W)
        sigma_hsv = repeat(sigma_hsv + self.epsilon, "b c -> b c h w", h=H, w=W)
        
        mu_prime_hsv = repeat(self.mu_hsv + self.mu0_hsv, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime_hsv = repeat(self.sigma_hsv + self.sigma0_hsv, "c -> b c h w", b=B, h=H, w=W)

        x_hsv = (x_hsv - mu_hsv) / sigma_hsv * sigma_prime_hsv + mu_prime_hsv
        x_hsv = hsv_to_rgb(x_hsv)
        
        x_norm = (x_lab + x_hsv)/2

        return self.model(x_norm),x_norm    

class LabHsvConcat(nn.Module):
    def __init__(
        self,
        model=None,
        device=None,
        epsilon=1e-7,
        requires_grad=True,
        
#         mu0 = None,
#         sigma0 = None,
        
        mu0_lab = [64, 20, -10],
        sigma0_lab = [12, 7, 5],
        mu0_hsv = [64, 20, -10],
        sigma0_hsv = [12, 7, 5],
        
    ):
        super(LabHsvConcat, self).__init__()
        self.epsilon = epsilon
        self.mu0_lab = torch.tensor(mu0_lab).to(device)
        self.sigma0_lab = torch.tensor(sigma0_lab).to(device)
        self.mu0_hsv = torch.tensor(mu0_hsv).to(device)
        self.sigma0_hsv = torch.tensor(sigma0_hsv).to(device)

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
        x_lab = rgb_to_lab(x)

        B, _, H, W = x_lab.shape

        mu_lab = x_lab.mean(axis=(2, 3))
        sigma_lab = x_lab.std(axis=(2, 3))

        mu_lab = repeat(mu_lab, "b c -> b c h w", h=H, w=W)
        sigma_lab = repeat(sigma_lab + self.epsilon, "b c -> b c h w", h=H, w=W)
        
        mu_prime_lab = repeat(self.mu_lab + self.mu0_lab, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime_lab = repeat(self.sigma_lab + self.sigma0_lab, "c -> b c h w", b=B, h=H, w=W)

        x_lab = (x_lab - mu_lab) / sigma_lab * sigma_prime_lab + mu_prime_lab
        x_lab = lab_to_rgb(x_lab)
        
        x_hsv = rgb_to_hsv(x)

        B, _, H, W = x_hsv.shape

        mu_hsv = x_hsv.mean(axis=(2, 3))
        sigma_hsv = x_hsv.std(axis=(2, 3))

        mu_hsv = repeat(mu_hsv, "b c -> b c h w", h=H, w=W)
        sigma_hsv = repeat(sigma_hsv + self.epsilon, "b c -> b c h w", h=H, w=W)
        
        mu_prime_hsv = repeat(self.mu_hsv + self.mu0_hsv, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime_hsv = repeat(self.sigma_hsv + self.sigma0_hsv, "c -> b c h w", b=B, h=H, w=W)

        x_hsv = (x_hsv - mu_hsv) / sigma_hsv * sigma_prime_hsv + mu_prime_hsv
        x_hsv = hsv_to_rgb(x_hsv)
        
        x_norm = torch.cat([x_lab,x_hsv],1)

        return self.model(x_norm),x_norm   
    
    
class Lab_keep_white(nn.Module):#白色用原图作mask
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
        super(Lab_keep_white, self).__init__()
        self.epsilon = epsilon
        self.mu0 = torch.tensor(mu0).to(device)
        self.sigma0 = torch.tensor(sigma0).to(device)
        self.zero = torch.zeros(3).to(device)
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
        zeros = repeat(self.zero, "c -> b c h w", b=B, h=H, w=W)
        
        mu_prime = repeat(self.mu + self.mu0, "c -> b c h w", b=B, h=H, w=W)
        sigma_prime = repeat(self.sigma + self.sigma0, "c -> b c h w", b=B, h=H, w=W)
        
        x_white_L = x[:,0,:,:].long()    
        x_white_A = x[:,1,:,:].long()  
        x_white_B = x[:,2,:,:].long()  
        
        x_white_L = torch.where(x_white_L < 82, 0, x_white_L).float()
        x_white_A = torch.where(x_white_L < 82, 0, x_white_A).float()
        x_white_B = torch.where(x_white_L < 82, 0, x_white_B).float()
        
        x_white_id_L = torch.where(x_white_L < 82, 0, (mu[:,0,:,:] / sigma[:,0,:,:] * sigma_prime[:,0,:,:] - mu_prime[:,0,:,:]).long()).float()
        x_nowhite_L = x[:,0,:,:] - x_white_L
        x_white_id_A = torch.where(x_white_L < 82, 0, (mu[:,1,:,:] / sigma[:,1,:,:] * sigma_prime[:,1,:,:] - mu_prime[:,1,:,:]).long()).float()
        x_nowhite_A = x[:,1,:,:] - x_white_A
        x_white_id_B = torch.where(x_white_L < 82, 0, (mu[:,2,:,:] / sigma[:,2,:,:] * sigma_prime[:,2,:,:] - mu_prime[:,2,:,:]).long()).float()
        x_nowhite_B = x[:,2,:,:] - x_white_B
        
        x_N = torch.stack((x_nowhite_L, x_nowhite_A, x_nowhite_B), 1)
        x_id = torch.stack((x_white_id_L, x_white_id_A, x_white_id_B),1)#problem白色部分全白，没有变化
        x_W = torch.stack((x_white_L, x_white_A, x_white_B),1)
        
#         print(x_nowhite.shape)
        

        x_N = (x_N - mu) / sigma * sigma_prime + mu_prime
        
        x = x_N + x_id + x_W
        
        x_norm = lab_to_rgb(x)

        return self.model(x_norm),x_norm
    
    
class Lab_gamma(nn.Module):
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
        super(Lab_gamma, self).__init__()
        self.epsilon = epsilon
        self.mu0 = torch.tensor(mu0).to(device)
        self.sigma0 = torch.tensor(sigma0).to(device)
        self.ln2 = torch.log(torch.tensor([2])).to(device)
#         self._56 = torch.tensor([-5/6]).to(device)
#         self._100 = torch.tensor([100]).to(device)

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
        
#         print(sigma_prime)
        
        gamma = (self.sigma + self.sigma0).long() / 100 #gamma增强
        
        mu_l = (self.sigma + self.sigma0).long() / 100
#         print(gamma[0],mu_l)
        gamma[0] = torch.where(gamma[0] < 1/12, 
                            (-5/6 * torch.log(gamma[0].float()) / self.ln2).long(), 
                            torch.exp(1-(0.5*(mu_l[0].float() + gamma[0].float()))).long())
#         gamma=2
#         print(gamma)
#         print(x[:,0,:,:])
        
        x[:,0,:,:] = (((x[:,0,:,:].long()/100)**gamma[0]) / ((x[:,0,:,:].long()/100)**gamma[0] + (1 - (x[:,0,:,:].long()/100)**gamma[0])*(mu_l[0]**gamma[0])) * 100).float()
#         print(x[:,0,:,:])
        
        x_norm = lab_to_rgb(x)
        
        return self.model(x_norm),x_norm
    
    
class Lab_keep_white_v2(nn.Module):
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
        super(Lab_keep_white_v2, self).__init__()
        self.epsilon = epsilon
        self.mu0 = torch.tensor(mu0).to(device)
        self.sigma0 = torch.tensor(sigma0).to(device)
        self.device = device

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
        
        x_norm = (x - mu) / sigma * sigma_prime + mu_prime
        threshold =(82 - mu[:,0,:,:]) / sigma[:,0,:,:] * sigma_prime[:,0,:,:] + mu_prime[:,0,:,:]#82
        x_norm_L = x_norm[:,0,:,:].clone()
#         print(threshold)
        #1.三次样条,效果不明显
#         a = -1 / (self.epsilon + (threshold - 100)**2)  
#         b = -2 / (torch.abs(threshold - 100) + self.epsilon)
#         x_norm_white = a*(x_norm[:,0,:,:]-100)**3 + b*(x_norm[:,0,:,:]-100)**2 + 100
        
        #2.n次多项式，n偶数，在x=threshold不可导，可考虑molifier磨光或者调整系数
        #实验acc降低，图像在阈值处不够光滑
        
#         x_norm[:,0,:,:] = torch.where(x_norm[:,0,:,:].long() > 110, 110*torch.ones((B,H,W)).to(self.device), x_norm[:,0,:,:])
        
#         x_norm[:,1,:,:] = torch.where(x_norm[:,1,:,:].long() < -90, -90*torch.ones((B,H,W)).to(self.device), x_norm[:,1,:,:])
#         x_norm[:,2,:,:] = torch.where(x_norm[:,2,:,:].long() < -90, -90*torch.ones((B,H,W)).to(self.device), x_norm[:,2,:,:])
        maximal = 100
#         print(maximal)
        n = 4
        a = 1 / ((threshold - maximal)**(n-1) - self.epsilon)
        x_norm_white = a * (x_norm[:,0,:,:] - maximal)**n + maximal
        
        x_norm[:,0,:,:] = torch.where(x_norm[:,0,:,:].long() > threshold, x_norm_white, x_norm[:,0,:,:])
        x_norm[:,0,:,:] = torch.where(x_norm_L.long() > 100, x_norm_L, x_norm[:,0,:,:])#染色归一化颜色风格有显著差距的patch时存在白色过色问题解决
        
#         #解决一些阈值附近较深背景表现差的问题:再做一次归一化 效果不好，白色部分变色
#         mu_norm = x_norm[:,0,:,:].clone().mean(axis=(1,2))
#         sigma_norm = x_norm[:,0,:,:].clone().std(axis=(1,2))
#         mu_norm = repeat(mu_norm, "b  -> b h w", h=H, w=W)
#         sigma_norm = repeat(sigma_norm + self.epsilon, "b -> b h w", h=H, w=W)
        
#         mu_prime_norm = x_norm[:,0,:,:].clone().mean(axis = (0,1,2))
#         sigma_prime_norm = x_norm[:,0,:,:].clone().std(axis = (0,1,2))
# #         mu_prime_norm = repeat(mu_prime_norm, "c -> b c h w", b=B, h=H, w=W)
# #         sigma_prime_norm = repeat(sigma_prime_norm, "c -> b c h w", b=B, h=H, w=W)
#         x_norm[:,0,:,:] = (x_norm[:,0,:,:] - mu_norm) / sigma_norm * sigma_prime_norm + mu_prime_norm
        


        #3.x + molifier
#         C = 50
#         x_norm_white = x_norm[:,0,:,:] + C*torch.exp(x_norm[:,0,:,:]-100)#1/(((x_norm[:,0,:,:]-100)/(100-threshold))**2-1-self.epsilon)).to(self.device)
#         x_norm_L = x_norm[:,0,:,:].clone()
#         x_norm[:,0,:,:] = torch.where(x_norm_L.long() > threshold, x_norm_white, x_norm[:,0,:,:])
#         x_norm[:,0,:,:] = torch.where(x_norm_L.long() > 100, x_norm_L, x_norm[:,0,:,:])
# #         print(x_norm[:,0,:,:])


        x_norm = lab_to_rgb(x_norm)
        
        return self.model(x_norm),x_norm
    
    

    

if __name__ == "__main__":
    import torch

    lab_norm = LabEMAPreNorm()
    x = torch.randn((1, 3, 224, 224)).clip(min=0, max=1)
    y = lab_norm(x)
    print((y - x).max())

    
    
