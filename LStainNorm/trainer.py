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


train_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class Trainer:
    def __init__(
        self,
        config_path: str,
    ):
        config = OmegaConf.load(config_path)

        if hasattr(config, "seed"):
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)

        ##### Create Dataloaders.
        trainset = HistoDataset(
            root=config.train_root,
            transform=train_transform,
        )
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
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

        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.config = config

        num_classes = len(os.listdir(config.train_root))

        ##### Create folders for the outputs.
        postfix = time.strftime("%Y%m%d_%H:%M")
        if hasattr(config, "postfix") and config.postfix != "":
            postfix += "_" + config.postfix

        self.output_path = os.path.join(config.output_path, postfix)

        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "weights"), exist_ok=True)
        self.logging = open(os.path.join(self.output_path, "logging.txt"), "w+")

        OmegaConf.save(config=config, f=os.path.join(self.output_path, "config.yaml"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TIMM(
            model_name=config.model,
            num_classes=num_classes,
        )
        
        
        
        mu0 = config.mu0
        sigma0 = config.sigma0
        mu0_lab = config.mu0_lab
        sigma0_lab = config.sigma0_lab
        mu0_hsv = config.mu0_hsv
        sigma0_hsv = config.sigma0_hsv
        mu0_rgb = config.mu0_rgb
        sigma0_rgb = config.sigma0_rgb
        

        if hasattr(config, "prenorm") and config.prenorm:
            print("Using PreNorm.")
            model = LabPreNorm(model, self.device, mu0=mu0,sigma0=sigma0)
            
        elif hasattr(config, "temnorm") and config.temnorm:
            print("Using TemplateNorm.")
            model = TemplateNorm(model, self.device, mu0=mu0,sigma0=sigma0)
            
            
        elif hasattr(config, "SA3") and config.SA3:
            print("Using LStainNorm.")
            model = SA3(model, self.device, mu0_lab=mu0_lab,sigma0_lab=sigma0_lab,mu0_hsv=mu0_hsv,sigma0_hsv=sigma0_hsv, mu0_rgb=mu0_rgb,sigma0_rgb=sigma0_rgb)
        
        else:
            print('Error! No such model in prenorm!')
            
        
        self.model = model.to(self.device)


        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=config.learning_rate,#5e-4
            weight_decay=config.weight_decay,
        )
            
        if config.scheduler_norm.lower() == "step" and config.scheduler.lower() == "cosine":
            self.scheduler_norm = lr_scheduler.StepLR(
                    optimizer=self.optimizer, 
                    step_size=2, 
                    gamma=0.9,
                )

            self.scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=config.T_max,
                    eta_min=config.min_learning_rate,
                )

        elif config.scheduler.lower() == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=config.T_max,
                    eta_min=config.min_learning_rate,
                )

        else:
            raise ValueError("Unkown scheduler {}".format(config.scheduler.lower()))

        self.epochs = config.epochs
        self.patience = config.patience

    def train(
        self,
    ):
        best_epoch = 0.0
        best_test_acc = 0.0

        time_start = time.time()

        msg = "[{}] Total training epochs : {}".format(
            datetime.now().strftime("%A %H:%M"), self.epochs
        )
        save_log(self.logging, msg)
     
        
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch()

            test_loss, test_acc = self.test_per_epoch(model=self.model,ep=epoch)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.output_path, "weights", "model_epoch{}.pth".format(epoch)
                    ),
                )
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_path, "weights", "best_model.pth"),
                )

            msg = "[{}] Epoch {:03d} \
                \n Train loss: {:.5f},   Train acc: {:.3f}%;\
                \n Test loss: {:.5f},   Test acc: {:.3f}%;  \
                \n Best test acc: {:.3f}%;\
                \n mu: {},  sigma: {}%;\
                \n mu_lab: {},  sigma_lab: {}%;\
                \n mu_rgb: {},  sigma_rgb: {}%;\
                \n mu_hsv: {},  sigma_hsv: {}% \n".format(                
                datetime.now().strftime("%A %H:%M"),
                epoch,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                best_test_acc,
                self.model.mu,
                self.model.sigma,
                self.model.mu_lab,
                self.model.sigma_lab,
                self.model.mu_rgb,
                self.model.sigma_rgb,
                self.model.mu_hsv,
                self.model.sigma_hsv,
            )
            save_log(self.logging, msg)

            if (epoch - best_epoch) > self.patience: 
                break


        msg = "[{}] Best test acc:{:.3f}% @ epoch {} \n".format(
            datetime.now().strftime("%A %H:%M"), best_test_acc, best_epoch
        )
        save_log(self.logging, msg)

        time_end = time.time()
        msg = "[{}] run time: {:.1f}s, {:.2f}h\n".format(
            datetime.now().strftime("%A %H:%M"),
            time_end - time_start,
            (time_end - time_start) / 3600,
        )
        save_log(self.logging, msg)
    
    

    
    def train_one_epoch(self):
        train_loss_recorder = AverageMeter()
        train_acc_recorder = AverageMeter()

        self.model.train()

        for img, label in tqdm(self.train_loader):
            self.optimizer.zero_grad()

            img = img.to(self.device)
            label = label.to(self.device)

            out = self.model(img)[0][LOGITS]
            loss = F.cross_entropy(out, label)

            loss.backward()
            self.optimizer.step()
            acc = accuracy(out, label)[0]            
            
            train_loss_recorder.update(loss.item(), out.size(0))
            train_acc_recorder.update(acc.item(), out.size(0))

        if self.config.scheduler_norm.lower() == "step" and self.config.scheduler.lower() == "cosine":
            for name, param in self.model.named_parameters():
                if "model" in name:
                    param.requires_grad = False
                    self.scheduler_norm.step()
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    self.scheduler.step()
                    param.requires_grad = True
        elif self.config.scheduler.lower() == "cosine":
            self.scheduler.step()
            

        train_loss = train_loss_recorder.avg
        train_acc = train_acc_recorder.avg

        return train_loss, train_acc

    def test_per_epoch(self, model,ep):
        test_loss_recorder = AverageMeter()
        test_acc_recorder = AverageMeter()
        with torch.no_grad():
            model.eval()
            for img, label in tqdm(self.test_loader):

                img = img.to(self.device)
                label = label.to(self.device)

                out = self.model(img)[0][LOGITS]
                loss = F.cross_entropy(out, label)
                output_norm = self.model(img)[1] #10.29 染色归一化后图片输出
                
                acc = accuracy(out, label)[0]

                test_loss_recorder.update(loss.item(), out.size(0))
                test_acc_recorder.update(acc.item(), out.size(0))
                

        test_loss = test_loss_recorder.avg
        test_acc = test_acc_recorder.avg

        return test_loss, test_acc
