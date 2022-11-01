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
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/ADI/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/test/ADI/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/BACK/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/test/BACK/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/DEB/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/test/DEB/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/LYM/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/test/LYM/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/MUC/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/test/MUC/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/MUS/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/test/MUS/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/NORM/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/test/NORM/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/STR/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/test/STR/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    fileDir = "/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/TUM/"  # 源图片文件夹路径 
    tarDir = '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png/test/TUM/'  # 移动到新的文件夹路径
    moveFile(fileDir)
    
    
    