##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil


def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.2  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
    return


if __name__ == '__main__':
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/ADI/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/test/ADI/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/ADI/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/val/ADI/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/DEB/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/test/DEB/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/DEB/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/val/DEB/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/BACK/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/test/BACK/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/BACK/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/val/BACK/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/LYM/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/test/LYM/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/LYM/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/val/LYM/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/MUC/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/test/MUC/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/MUC/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/val/MUC/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/MUS/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/test/MUS/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/MUS/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/val/MUS/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/STR/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/test/STR/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/STR/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/val/STR/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/TUM/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/test/TUM/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/train/TUM/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/val/TUM/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    
    