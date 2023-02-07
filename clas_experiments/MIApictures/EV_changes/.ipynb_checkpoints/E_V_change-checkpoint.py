import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import xlwt
import xlrd
from xlutils.copy import copy
import torch



img=cv.imread(r'/root/autodl-tmp/MIApictures/NORM-TCGA-DAKDQWRM.png')
img32 = img * (1. / 255)
# print(img32.shape)
img_lab = cv.cvtColor(np.float32(img32), cv.COLOR_BGR2LAB)
# img_lab = torch.tensor(img_lab)
# img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
print(img_lab)
img0 = img_lab[..., 0]
img1 = img_lab[..., 1]
img2 = img_lab[..., 2]
L = np.array(img0)
print(np.shape(L))
EL = np.mean(L)
DL = np.std(L)
A = np.array(img1)
EA = np.mean(A)
DA = np.std(A)
B = np.array(img2)
EB = np.mean(B)
DB = np.std(B)
# img_lab = img_lab* (1. )
# img_rgb = cv.cvtColor(np.float32(img_lab), cv.COLOR_HSV2RGB)
# print(img_rgb)
print(EL, EA, EB, DL, DA, DB)

# img0 = img0-EL*np.ones((224,224))+80*np.ones((224,224))
# img_lab0 = np.dstack((img0, img1, img2))

# img_BGR = cv.cvtColor(np.float32(img_lab0), cv.COLOR_LAB2BGR)*255
# print(img0)
# cv.imwrite('/root/autodl-tmp/MIApictures/80.png', img_BGR)

img0 = img0/DL*25
img_lab0 = np.dstack((img0, img1, img2))

img_BGR = cv.cvtColor(np.float32(img_lab0), cv.COLOR_LAB2BGR)*255
print(img0)
cv.imwrite('/root/autodl-tmp/MIApictures/25.png', img_BGR)



