import os
import time
import cv2
import numpy as np
import PIL.Image as Image
from torchvision import transforms as transforms
from skimage import color
#from color_norm.Reinhard_quick import reinhard_cn, reinhard_cn_temp
#from Reinhard_quick import reinhard_cn, reinhard_cn_temp
def quick_loop(image, image_avg, image_std, temp_avg, temp_std, isHed=False):
    image = (image - np.array(image_avg)) * (np.array(temp_std) / np.array(image_std)) + np.array(temp_avg)
    if isHed: # HED in range[0,1]
        pass
    else: # LAB/HSV in range[0,255]
        image = np.clip(image, 0, 255).astype(np.uint8)  
    return image

def getavgstd(image):
    avg = []
    std = []
    image_avg_l = np.mean(image[:, :, 0])
    image_std_l = np.std(image[:, :, 0])
    image_avg_a = np.mean(image[:, :, 1])
    image_std_a = np.std(image[:, :, 1])
    image_avg_b = np.mean(image[:, :, 2])
    image_std_b = np.std(image[:, :, 2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg, std)

def reinhard_cn(image_path, temp_path, save_path, isDebug=False, color_space=None):
    isHed = False
    image = cv2.imread(image_path)
    if isDebug:
        cv2.imwrite('source.png', image)
    template = cv2.imread(temp_path)  ### template images
    if isDebug:
        cv2.imwrite('template.png', template)

    if color_space == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # LAB range[0,255]
        template = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
    elif color_space == 'HED':
        isHed = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color.rgb2hed needs RGB as input
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

        image = color.rgb2hed(image)  # HED range[0,1]
        template = color.rgb2hed(template)
    elif color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    elif color_space == 'GRAY': 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(save_path, image)
        return

    image_avg, image_std = getavgstd(image)
    template_avg, template_std = getavgstd(template)
    if isDebug: 
        print("isDebug!!!")
        print('source_avg: ', image_avg)
        print('source_std: ', image_std)
        print('target_avg: ', template_avg)
        print('target_std: ', template_std)

    # Reinhard's Method to Stain Normalization
    image = quick_loop(image, image_avg, image_std, template_avg, template_std, isHed=isHed)

    if color_space == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        cv2.imwrite(save_path, image)
    elif color_space == 'HED': # HED[0,1]->RGB[0,255]
        image = color.hed2rgb(image) 
        imin = image.min()
        imax = image.max()
        image = (255 * (image - imin) / (imax - imin)).astype('uint8')
        image = Image.fromarray(image)
        image.save(save_path)
    elif color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        cv2.imwrite(save_path, image)

    if isDebug:
        cv2.imwrite('results.png', image)


# # template_path = '/root//autodl-tmp/pycharm_project_colorNorm/color_norm/demo/other/TUM-AIQIMVKD_template.png'
# template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/1.png'
# #template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-AR-A1AS-01Z-00-DX1_0_512.png'
# #template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-B0-5698-01Z-00-DX1_512_0.png'
# #template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-F9-A8NY-01Z-00-DX1_512_512.png'
# #template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-FG-A87N-01Z-00-DX1_0_512.png'
# #template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-HE-7130-01Z-00-DX1_0_0.png'
# #template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-HT-8564-01Z-00-DX1_0_512.png'
# #template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-KB-A93J-01A-01-TS1_512_0.png'
# #template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-UZ-A9PJ-01Z-00-DX1_0_512.png'
# #template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-XS-A8TJ-01Z-00-DX1_512_512.png'

# # path_dataset = '/autodl-tmp/nine_class/standard/train'
# # path_dataset = '/autodl-tmp/nine_class/standard/val'
# # path_dataset = '/autodl-tmp/nine_class/standard/test'

# path_dataset_list = [
#     '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
#     '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
#     '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
# ]

# # target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/train'
# # target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/val' #/val_colorNorm
# # target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/test' #/test_colorNorm

# target_path_dataset_list = [
#     '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/1/train',
#     '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/1/test',
#     '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/1/val'
# ]

# for target_path_dataset in target_path_dataset_list:
#     if not os.path.isdir(target_path_dataset):
#         os.makedirs(target_path_dataset)

# i = 0

# for idx in range(len(path_dataset_list)):
#     path_dataset = path_dataset_list[idx]
#     target_path_dataset = target_path_dataset_list[idx]
#     print(path_dataset)
#     print(target_path_dataset)
#     for class_dir in os.listdir(path_dataset):
#         print(os.listdir(path_dataset))
#         if class_dir in ['label','bound','.ipynb_checkpoints']:
#             continue
#         path_class = os.path.join(path_dataset,class_dir)
#         target_path_class = os.path.join(target_path_dataset,class_dir)
#         # print(target_path_class)
#         if not os.path.isdir(target_path_class):
#             os.makedirs(target_path_class)
#         print(path_class)
#         t1 = time.time()
#         for image in os.listdir(path_class):
#             i += 1
#             path_img = os.path.join(path_class,image)
#             #print(path_img)
#             save_path = os.path.join(target_path_class,image)
#             # print(save_path)
#             # img = cv2.imread(path_img)
#             # 以后一定要记得.ipynb_checkpoint的存在+try很好用
#             # try:
#             img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
#     #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
#             # except:
#             #     print('path_img:',path_img)
#             #     print('save_path:',save_path)
#             #     continue

#             if i % 200 == 0:
#                 t3 = time.time()
#                 print('i:',i, 'time:',t3-t1)
#             # break
#         t2 = time.time()
#         print(t2-t1)
#         # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/10.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]

# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/train'
# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/val' #/val_colorNorm
# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/test' #/test_colorNorm

target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/10/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/10/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/10/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/11.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-AR-A1AS-01Z-00-DX1_0_512.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-B0-5698-01Z-00-DX1_512_0.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-F9-A8NY-01Z-00-DX1_512_512.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-FG-A87N-01Z-00-DX1_0_512.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-HE-7130-01Z-00-DX1_0_0.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-HT-8564-01Z-00-DX1_0_512.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-KB-A93J-01A-01-TS1_512_0.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-UZ-A9PJ-01Z-00-DX1_0_512.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-XS-A8TJ-01Z-00-DX1_512_512.png'

# path_dataset = '/autodl-tmp/nine_class/standard/train'
# path_dataset = '/autodl-tmp/nine_class/standard/val'
# path_dataset = '/autodl-tmp/nine_class/standard/test'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]

# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/train'
# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/val' #/val_colorNorm
# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/test' #/test_colorNorm

target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/11/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/11/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/11/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/12.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-AR-A1AS-01Z-00-DX1_0_512.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-B0-5698-01Z-00-DX1_512_0.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-F9-A8NY-01Z-00-DX1_512_512.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-FG-A87N-01Z-00-DX1_0_512.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-HE-7130-01Z-00-DX1_0_0.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-HT-8564-01Z-00-DX1_0_512.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-KB-A93J-01A-01-TS1_512_0.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-UZ-A9PJ-01Z-00-DX1_0_512.png'
#template_path = '/root/autodl-tmp/RandStainNA-master/segmentation/colornorm/template/TCGA-XS-A8TJ-01Z-00-DX1_512_512.png'

# path_dataset = '/autodl-tmp/nine_class/standard/train'
# path_dataset = '/autodl-tmp/nine_class/standard/val'
# path_dataset = '/autodl-tmp/nine_class/standard/test'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]

# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/train'
# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/val' #/val_colorNorm
# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/test' #/test_colorNorm

target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/12/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/12/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/12/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/13.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/13/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/13/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/13/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/14.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/14/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/14/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/14/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/15.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/15/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/15/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/15/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/16.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/16/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/16/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/16/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/2.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/2/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/2/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/2/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/3.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/3/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/3/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/3/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/4.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/4/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/4/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/4/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/5.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/5/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/5/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/5/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/6.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/6/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/6/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/6/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/7.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/7/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/7/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/7/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/8.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/8/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/8/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/8/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
template_path = '/root/autodl-tmp/RandStainNA-master/classification/NCT-CRC-HE-100K-NONORM/templates/9.png'

path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/val'
]


target_path_dataset_list = [
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/9/train',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/9/test',
    '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/9/val'
]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        print(os.listdir(path_dataset))
        if class_dir in ['label','bound','.ipynb_checkpoints']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            #print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break
#################################################################################################################################################
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       