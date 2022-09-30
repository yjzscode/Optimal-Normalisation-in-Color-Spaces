# model_name='resnet18_lab_norm'
# #dataset_path='/root/autodl-tmp/RandStainNA-master/classification/BACHdata/b062'
# dataset_gray_path='/root/autodl-tmp/BACH/gray'
# train_path='/root/autodl-tmp/RandStainNA-master/classification/train.py'
# train_path_lab_norm='/root/autodl-tmp/RandStainNA-master/classification/train_lab_norm.py'
# b=256
# workers=3
# lr=0.1
# num_classes=8
# # # ####### baseline ###############
# python $train_path_lab_norm '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/origin7k' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline7k_lab_norm_AddOrigin$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

model_name='resnet18_hsv_norm'
#dataset_path='/root/autodl-tmp/RandStainNA-master/classification/BACHdata/b062'
dataset_gray_path='/root/autodl-tmp/BACH/gray'
train_path='/root/autodl-tmp/RandStainNA-master/classification/train.py'
train_path_hsv_norm='/root/autodl-tmp/RandStainNA-master/classification/train_hsv_norm.py'
train_path_hed_norm='/root/autodl-tmp/RandStainNA-master/classification/train_hed_norm.py'
b=256
workers=3
lr=0.1
num_classes=8
# # ####### baseline ###############
python $train_path_hsv_norm '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/origin7k' \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
--epochs 70 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment baseline7k_hsv_norm_AddOrigin_70ep$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

# python $train_path_hed_norm '/root/autodl-tmp/RandStainNA-master/classification/CRC-VAL-HE-7K/PNG/origin7k' \
# --model 'resnet18_hed_norm' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline7k_hed_norm_AddOrigin$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp


model_name='resnet18_hsv_norm'
#dataset_path='/root/autodl-tmp/RandStainNA-master/classification/BACHdata/b062'
dataset_gray_path='/root/autodl-tmp/BACH/gray'
train_path='/root/autodl-tmp/RandStainNA-master/classification/train.py'
train_path_hsv_norm='/root/autodl-tmp/RandStainNA-master/classification/train_hsv_norm.py'
train_path_hed_norm='/root/autodl-tmp/RandStainNA-master/classification/train_hed_norm.py'
b=128
workers=3
lr=0.1
num_classes=4
# # ####### baseline ###############
python $train_path_hsv_norm '/root/autodl-tmp/RandStainNA-master/classification/BACHnorm' \
--model $model_name \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
--epochs 70 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment baselineBACH_hsv_norm_AddOrigin_70ep$model_name \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

# python $train_path_hed_norm '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/origin' \
# --model 'resnet18_hed_norm' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baselineBACH_hed_norm_AddOrigin$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/b062' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_b062_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/b081' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_b081_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/is039' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_is039_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/is055' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_is055_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/is05612' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_is05612_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/is05615' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_is05615_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/iv002' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_iv002_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/iv011' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_iv011_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/iv0521' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_iv0521_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/iv0525' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_iv0525_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/n051' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_n051_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/n056' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_n056_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp


# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/BACH/BACHdata/origin' \
# --model $model_name \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment baseline_origin_50$model_name \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp


