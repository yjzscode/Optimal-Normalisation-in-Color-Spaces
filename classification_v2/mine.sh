train_path='/root/autodl-tmp/RandStainNA-master/classification/train_all.py'
b=256
workers=15
lr=0.1
num_classes=8
# # ####### baseline ###############
# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_8' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_TEM8 \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_6' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_TEM6 \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_9' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_TEM9 \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_2' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_TEM2 \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_13' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_TEM13 \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_mean' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_TEMmean \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_lab_norigin' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_LAB_norigin \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_lab_norm_6' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_LAB_6 \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_lab_norm_8' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_LAB_8 \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
--model 'resnet18_lab_norm_mean' \
--num-classes $num_classes --dataset 'torch/image_folder' \
--color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
--epochs 2 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
--opt sgd --weight-decay 1e-4 --momentum 0.9 \
--sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
--experiment 100K_LAB_mean_ep2 \
-j $workers --no-prefetcher --pin-mem \
--seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_hsv_norm' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_hsv \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_hed_norm' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_hed \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_concat' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_concat \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_avg' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_avg \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp

# python $train_path '/root/autodl-tmp/RandStainNA-master/classification/100K-NONORM-png' \
# --model 'resnet18_weighted_avg' \
# --num-classes $num_classes --dataset 'torch/image_folder' \
# --color-jitter 0 0 0 0 --hflip 0.5 --vflip 0.5 \
# --epochs 50 --batch-size $b --validation-batch-size $b --warmup-epochs 3 --warmup-lr 0.001 \
# --opt sgd --weight-decay 1e-4 --momentum 0.9 \
# --sched cosine --lr $lr --min-lr 1e-6 --lr-k-decay 1.0 --lr-cycle-limit 1 --lr-cycle-decay 0.01 \
# --experiment 100K_weighted_avg \
# -j $workers --no-prefetcher --pin-mem \
# --seed 97 --native-amp





