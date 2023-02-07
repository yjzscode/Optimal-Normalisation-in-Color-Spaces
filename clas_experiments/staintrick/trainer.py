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
    LabEMAPreNorm,
    LabRandNorm,
    TemplateNorm, 
    HsvPreNorm, 
    HedPreNorm, 
    LabHsvAvg,
    LabHsvConcat,
    Lab_keep_white,
    Lab_gamma,
    Lab_keep_white_v2,
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
# To fix the EOFError,discribed in https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0

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
        ##### Create models.
#         if hasattr(config, "gpu_id"):
#             self.device = torch.device(
#                 "cuda:{}".format(config.gpu_id) if torch.cuda.is_available() else "cpu"
#             )
#         else:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TIMM(
            model_name=config.model,
            num_classes=num_classes,
        )
        
        model_concat = TIMM(
            model_name=config.model,
            num_classes=num_classes,
            chans=6,
        )
        
        mu0 = config.mu0
        sigma0 = config.sigma0
#         mu0_lab = config.mu0_lab
#         sigma0_lab = config.sigma0_lab
#         mu0_hsv = config.mu0_hsv
#         sigma0_hsv = config.sigma0_hsv
        

        if hasattr(config, "prenorm") and config.prenorm:
            print("Using PreNorm.")
            prenorm = True
            model = LabPreNorm(model, self.device, mu0=mu0,sigma0=sigma0)
        else:
            prenorm = False
            
        if hasattr(config, "avg") and config.avg:
            avg = True
            print("Using LabHsvAvg.")
            model = LabHsvAvg(model, self.device, mu0_lab=mu0_lab,sigma0_lab=sigma0_lab,mu0_hsv=mu0_hsv,sigma0_hsv=sigma0_hsv)
        else:
            avg = False
        if hasattr(config, "concat") and config.concat:
            concat = True
            print("Using LabHsvConcat.")
            model = LabHsvConcat(model_concat, self.device, mu0_lab=mu0_lab,sigma0_lab=sigma0_lab,mu0_hsv=mu0_hsv,sigma0_hsv=sigma0_hsv)
        else:
            concat = False
            
        if hasattr(config, "lab_keep_white") and config.lab_keep_white:
            print("Using Lab_keep_white.")
            model = Lab_keep_white(model, self.device, mu0=mu0,sigma0=sigma0)
            lab_keep_white = True
        else:
            lab_keep_white = False
            
        if hasattr(config, "lab_gamma") and config.lab_gamma:
            print("Using Lab_gamma.")
            model = Lab_gamma(model, self.device, mu0=mu0,sigma0=sigma0)
            lab_gamma = True
        else:
            lab_gamma = False
            
        if hasattr(config, "lab_keep_white_v2") and config.lab_keep_white_v2:
            print("Using Lab_keep_white_v2.")
            model = Lab_keep_white_v2(model, self.device, mu0=mu0,sigma0=sigma0)
            lab_keep_white_v2 = True
        else:
            lab_keep_white_v2 = False
            
            
        if hasattr(config, "emaprenorm") and config.emaprenorm:
            print("Using EMAPreNorm.")
            model = LabEMAPreNorm(
                model=model, 
                device=self.device,
                lmbd=config.emaprenorm_lambda if hasattr(config, "emaprenorm_lambda") else 0,
            )

        elif hasattr(config, "randnorm") and config.randnorm:
            print("Using RandNorm.")
            model = LabRandNorm(model, self.device)
            
        elif hasattr(config, "temnorm") and config.temnorm:
            print("Using TemplateNorm.")
            model = TemplateNorm(model, self.device, mu0=mu0,sigma0=sigma0)
            
        elif hasattr(config, "hsvnorm") and config.hsvnorm:
            print("Using HsvPreNorm.")
            model = HsvPreNorm(model, self.device, mu0=mu0,sigma0=sigma0)
        
        elif hasattr(config, "hednorm") and config.hednorm:
            print("Using HedPreNorm.")
            model = HedPreNorm(model, self.device, mu0=mu0,sigma0=sigma0)
        
            
        
        else:
            a = False
            
        
        self.model = model.to(self.device)

#         if prenorm or lab_keep_white or lab_gamma or lab_keep_white_v2:
#             self.optimizer = AdamW(
#                 params=[
#                     {"params": model.mu, "lr": 5.0e-2},#12.11 config.learning_rate },#10
#                     {
#                         "params": model.sigma,
#                         "lr": 5.0e-2,
#                     },
#                     {"params": self.model.model.parameters()},
#                 ],
#                 lr=5.0e-4, #config.learning_rate,
#                 weight_decay=config.weight_decay,
#             )

#             self.optimizer1 = AdamW(
#                 params=[
#                     {"params": model.mu, "lr": config.learning_rate},#10
#                     {
#                         "params": model.sigma,
#                         "lr": config.learning_rate,#50
#                     },
#                 ],
# #                 lr=config.learning_rate,
#                 weight_decay=config.weight_decay,
#             )
#             self.optimizer2 = AdamW(
#                 params=[
#                     {"params": self.model.model.parameters()},
#                 ],
#                 lr=config.learning_rate,
#                 weight_decay=config.weight_decay,
#             )
#         elif concat or avg:
#             self.optimizer = AdamW(
#                 params=[
#                     {"params": model.mu_lab, "lr": config.learning_rate / 50},#10
#                     {
#                         "params": model.sigma_lab,
#                         "lr": config.learning_rate / 10,#50
#                     },
                    
#                     {"params": model.mu_hsv, "lr": config.learning_rate},
#                     {
#                         "params": model.sigma_hsv,
#                         "lr": config.learning_rate,
#                     },
#                     {"params": model.model.parameters()},
#                 ],
#                 lr=config.learning_rate,
#                 weight_decay=config.weight_decay,
#             )
            
        
#         else:
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=5.0e-4,
            weight_decay=config.weight_decay,
        )
            

        #prenorm与cnn不一样的scheduler 11.10    
#         if config.scheduler_norm.lower() == "linear":#两个optimizer两个scheduler很慢
#             self.scheduler = lr_scheduler.LinearLR(
#                 optimizer=self.optimizer1,
#             )
#         if config.scheduler.lower() == "cosine":
#             self.scheduler = lr_scheduler.CosineAnnealingLR(
#                 optimizer=self.optimizer,
#                 T_max=config.T_max,
#                 eta_min=config.min_learning_rate,
#             )
        if config.scheduler_norm.lower() == "step" and config.scheduler.lower() == "cosine":
#             self.scheduler_norm = lr_scheduler.StepLR(
#                     optimizer=self.optimizer, 
#                     step_size=2, 
#                     gamma=0.9,
#                 )
#             self.scheduler_norm = lr_scheduler.CosineAnnealingLR(
#                 optimizer=self.optimizer,
#                 T_max=5,
#                 eta_min=config.min_learning_rate,
#             )
            self.scheduler_norm=lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=self.optimizer,
                T_0=5,
                T_mult=1,
                eta_min=config.min_learning_rate,
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
#         if config.scheduler.lower() == "expoential":
#             self.scheduler = lr_scheduler.ExponentialLR(
#                 optimizer=self.optimizer, gamma=config.gamma
#             )
#         elif config.scheduler.lower() == "cosine":
#             self.scheduler = lr_scheduler.CosineAnnealingLR(
#                 optimizer=self.optimizer,
#                 T_max=config.T_max,
#                 eta_min=config.min_learning_rate,
#             )
#         elif config.scheduler.lower() == "constant":
#             self.scheduler = lr_scheduler.ConstantLR(
#                 optimizer=self.optimizer,
#             )
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
#         list = []
        
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
#             test_loss, test_acc, learnable_para = self.test_per_epoch(model=self.model,ep=epoch)
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
                \n Best test acc: {:.3f} \n".format(
                datetime.now().strftime("%A %H:%M"),
                epoch,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                best_test_acc,
            )
            save_log(self.logging, msg)

            if (epoch - best_epoch) > self.patience:
                break
        
#         list.append(learnable_para)
        #可学习参数的值
#         column=['EL','EA','EB','VL','VA','VB'] #列表头名称columns=column,
#         test=pd.DataFrame(data=list)#将数据放进表格
#         test.to_csv(r'/root/autodl-tmp/staintrick/results/test.csv') #数据存入csv,存储位置及文件名称


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
            
#             self.optimizer1.zero_grad()
#             self.optimizer2.zero_grad()

            img = img.to(self.device)
            label = label.to(self.device)

            out = self.model(img)[0][LOGITS]
            loss = F.cross_entropy(out, label)

            loss.backward()
            self.optimizer.step()
#             self.optimizer1.step()
#             self.optimizer2.step()

            acc = accuracy(out, label)[0]
            
            
            
            train_loss_recorder.update(loss.item(), out.size(0))
            train_acc_recorder.update(acc.item(), out.size(0))

#         self.scheduler_norm.step()
#         print(self.model.named_parameters())
#         self.scheduler.step()
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
#         learnable_para = []
        with torch.no_grad():
            model.eval()
#             for batch_idx, (img, label) in enumerate(self.test_loader):
            for img, label in tqdm(self.test_loader):

                img = img.to(self.device)
                label = label.to(self.device)

                out = self.model(img)[0][LOGITS]
                loss = F.cross_entropy(out, label)
                output_norm = self.model(img)[1] #10.29 染色归一化后图片输出
                
                acc = accuracy(out, label)[0]

                test_loss_recorder.update(loss.item(), out.size(0))
                test_acc_recorder.update(acc.item(), out.size(0))
                
#                 for i in range(label.shape[0]):#11.8
#                     print(out[i])
#                     print(label[i])
                    
#                     if label[i] != out[i].argmax(dim=0):
#                         torchvision.utils.save_image(
#                     img[i],
#                     os.path.join(r'/root/autodl-tmp/staintrick/results/wrong_origin', 'ep%d-test-batch%d-%s.jpg' % (ep,i,datetime.now().strftime("%A %H:%M"))),
#                     padding=0,
#                     normalize=False)
#                         torchvision.utils.save_image(
#                     output_norm[i],
#                     os.path.join(r'/root/autodl-tmp/staintrick/results/wrong_norm', 'ep%d-test-batch%d-%s.jpg' % (ep, i,  datetime.now().strftime("%A %H:%M"))),
#                     padding=0,
#                     normalize=False)
                        
                        
                
#                 learnable_para.append((self.model.mu + self.model.mu0).detach().cpu().numpy().tolist()
#                             +(self.model.sigma + self.model.sigma0).detach().cpu().numpy().tolist())
                
                
#                 path1='/root/autodl-tmp/staintrick/results/origin'
#                 path2='/root/autodl-tmp/staintrick/results/LabPreNorm'
#                 path3='/root/autodl-tmp/staintrick/results/template/%s' % (self.config.postfix)
#                 path4='/root/autodl-tmp/staintrick/results/lab_keep_white'
#                 path5='/root/autodl-tmp/staintrick/results/lab_gamma'
#                 path6='/root/autodl-tmp/staintrick/results/lab_keep_white_v2/%s' % (self.config.postfix)
                
                               
#                 if self.config.prenorm:
#                     if not os.path.exists(path2):
#                         os.mkdir(path2)
#                     torchvision.utils.save_image(
#                         output_norm,
#                         os.path.join(path2, 'ep%d-test-batch-%s.jpg' % (ep,datetime.now().strftime("%A %H:%M:%S"))),
#                         padding=0,
#                         normalize=False)
#                 elif self.config.temnorm:
#                     if not os.path.exists(path3):
#                         os.mkdir(path3)
#                     torchvision.utils.save_image(
#                         output_norm,
#                         os.path.join(path3, 'ep%d-test-batch-%s.jpg' % (ep,datetime.now().strftime("%A %H:%M:%S"))),
#                         padding=0,
#                         normalize=False)
#                 elif self.config.lab_keep_white:
#                     if not os.path.exists(path4):
#                         os.mkdir(path4)
#                     torchvision.utils.save_image(
#                         output_norm,
#                         os.path.join(path4, 'ep%d-test-batch-%s.jpg' % (ep,datetime.now().strftime("%A %H:%M:%S"))),
#                         padding=0,
#                         normalize=False)
                    
#                 elif self.config.lab_gamma:
#                     if not os.path.exists(path5):
#                         os.mkdir(path5)
#                     torchvision.utils.save_image(
#                         output_norm,
#                         os.path.join(path5, 'ep%d-test-batch-%s.jpg' % (ep,datetime.now().strftime("%A %H:%M:%S"))),
#                         padding=0,
#                         normalize=False)
                    
#                 elif self.config.lab_keep_white_v2:
#                     if not os.path.exists(path6):
#                         os.mkdir(path6)
#                     torchvision.utils.save_image(
#                         output_norm,
#                         os.path.join(path6, 'ep%d-test-batch-%s.jpg' % (ep,datetime.now().strftime("%A %H:%M:%S"))),
#                         padding=0,
#                         normalize=False)
#                 if not os.path.exists(path1):
#                     os.mkdir(path1)
#                 torchvision.utils.save_image(
#                     img,
#                     os.path.join(path1,'ep%d-test-batch-%s.jpg' % (ep,datetime.now().strftime("%A %H:%M:%S"))),
#                     padding=0,
#                     normalize=False)

        test_loss = test_loss_recorder.avg
        test_acc = test_acc_recorder.avg

        return test_loss, test_acc#, learnable_para
