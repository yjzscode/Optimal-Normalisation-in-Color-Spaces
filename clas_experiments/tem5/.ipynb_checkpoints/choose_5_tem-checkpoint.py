# author by LYS 2017/5/24
# for Deep Learning course
'''
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
'''
import os, random, shutil
def copyFile(fileDir1,fileDir2,fileDir3,fileDir4,fileDir5,fileDir6,fileDir7,fileDir8,tarDir):
    # 1
    pathDir1 = os.listdir(fileDir1)
    pathDir2 = os.listdir(fileDir2)
    pathDir3 = os.listdir(fileDir3)
    pathDir4 = os.listdir(fileDir4)
    pathDir5 = os.listdir(fileDir5)
    pathDir6 = os.listdir(fileDir6)
    pathDir7 = os.listdir(fileDir7)
    pathDir8 = os.listdir(fileDir8)
    pathDir = pathDir1+pathDir2+pathDir3+pathDir4+pathDir5+pathDir6+pathDir7+pathDir8
    
 
    # 2
    sample = random.sample(pathDir, 5)
    print (sample)
    
    # 3
# 	for name in sample:
# 		shutil.copyfile(fileDir+name, tarDir+name)
if __name__ == '__main__':
    fileDir1 = "/root/autodl-tmp/clas_experiments/RandStainNA-master/classification/100K-NONORM-png/train0/ADI"
    fileDir2 = '/root/autodl-tmp/clas_experiments/RandStainNA-master/classification/100K-NONORM-png/train0/DEB'
    fileDir3 = '/root/autodl-tmp/clas_experiments/RandStainNA-master/classification/100K-NONORM-png/train0/LYM'
    fileDir4 = '/root/autodl-tmp/clas_experiments/RandStainNA-master/classification/100K-NONORM-png/train0/MUC'
    fileDir5 = '/root/autodl-tmp/clas_experiments/RandStainNA-master/classification/100K-NONORM-png/train0/MUS'
    fileDir6 = '/root/autodl-tmp/clas_experiments/RandStainNA-master/classification/100K-NONORM-png/train0/NORM'
    fileDir7 = '/root/autodl-tmp/clas_experiments/RandStainNA-master/classification/100K-NONORM-png/train0/STR'
    fileDir8 = '/root/autodl-tmp/clas_experiments/RandStainNA-master/classification/100K-NONORM-png/train0/TUM'
    tarDir = '/root/autodl-tmp/clas_experiments/tem5/tem5_2'
    
    copyFile(fileDir1,fileDir2,fileDir3,fileDir4,fileDir5,fileDir6,fileDir7,fileDir8,tarDir)