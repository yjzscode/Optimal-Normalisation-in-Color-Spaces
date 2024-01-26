import os
import torch
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime
import time
from omegaconf import OmegaConf
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim import lr_scheduler
import pandas as pd
import torchvision

######
from model import (
    TIMM,
    LabPreNorm,
    TemplateNorm,
    SA3
)
from set import HistoDataset
from utils import (
    AverageMeter,
    accuracy,
    save_log,
    LOGITS,
)

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

class Test:
    def __init__(
            self,
            config_path: str,
    ):
        config = OmegaConf.load(config_path)
        ##### Create Dataloaders.
        testset = HistoDataset(
            root=config.test_root,
            transform=test_transform,
        )
        test_loader = DataLoader(
            dataset=testset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        mu0_lab = config.mu0_lab
        sigma0_lab = config.sigma0_lab
        mu0_hsv = config.mu0_hsv
        sigma0_hsv = config.sigma0_hsv
        mu0_rgb = config.mu0_rgb
        sigma0_rgb = config.sigma0_rgb

        model = TIMM(
            model_name=config.model,
            num_classes=8,
        )

        model = SA3(model, self.device, mu0_lab=mu0_lab, sigma0_lab=sigma0_lab, mu0_hsv=mu0_hsv,sigma0_hsv=sigma0_hsv, mu0_rgb=mu0_rgb,sigma0_rgb=sigma0_rgb)
        state_dict = model.load_state_dict(torch.load(
            "./staintrick/results/20230206_01:44_HsvTem_1_vgg11/weights/best_model.pth",
            map_location='cpu', strict=False))
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model}
        model.load_state_dict(pretrained_dict)
        self.model = model.to(self.device)

    def test(
            self,
    ):
        test_loss, test_acc = self.test_per_epoch(model=self.model)
        print(test_loss, test_acc)

    def test_per_epoch(self, model, ep):
        test_loss_recorder = AverageMeter()
        test_acc_recorder = AverageMeter()
        #         learnable_para = []
        with torch.no_grad():
            model.eval()
            #             for batch_idx, (img, label) in enumerate(self.test_loader):
            for img, label in tqdm(self.test_loader):
                img = img.to(self.device)
                label = label.to(self.device)

                out = self.model(img)[0][LOGITS]
                loss = F.cross_entropy(out, label)
                output_norm = self.model(img)[1]  # 10.29 染色归一化后图片输出

                acc = accuracy(out, label)[0]

                test_loss_recorder.update(loss.item(), out.size(0))
                test_acc_recorder.update(acc.item(), out.size(0))

        test_loss = test_loss_recorder.avg
        test_acc = test_acc_recorder.avg

        return test_loss, test_acc 

if __name__ == "__main__":

    config_path = './staintrick/SA1.yaml'
    test = Test(config_path)
    test.test()