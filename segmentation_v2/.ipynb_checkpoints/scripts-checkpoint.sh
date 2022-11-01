ep=30
b=2
lr=2e-5
workers=15
# dataset_path='/root/autodl-tmp/RandStainNA-master/segmentation/standard'
dataset_path='/root/autodl-tmp/RandStainNA-master/segmentation/train_split'

# baseline
# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_1' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_tem1 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_2' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_tem2 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_3' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_tem3 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_4' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_tem4 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_5' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_tem5 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_mean' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_tem_mean \
# --seed 97 \
# --workers $workers\

#lab

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_lab_norigin' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_lab_norigin \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_lab_2' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_lab_2 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_lab_3' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_lab_3 \
# --seed 97 \
# --workers $workers\

python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
--model 'CIAnet_lab' \
--dataset $dataset_path \
--epochs 2 --batch-size $b \
--lr $lr \
--output seg_lab_mean \
--seed 97 \
--workers $workers\

#hsv

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_hsv' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_hsv \
# --seed 97 \
# --workers $workers\

#hed

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_hed' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_hed \
# --seed 97 \
# --workers $workers\

#integration

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_concat' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_concat \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_avg' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_avg \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_weighted_avg' \
# --dataset $dataset_path \
# --epochs $ep --batch-size $b \
# --lr $lr \
# --output seg_weighted_avg \
# --seed 97 \
# --workers $workers\


####################################
#60ep
# baseline
# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_1' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_tem1 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_2' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_tem2 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_3' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_tem3 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_4' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_tem4 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_5' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_tem5 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_mean' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_tem_mean \
# --seed 97 \
# --workers $workers\

# #lab

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_lab_2' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_lab_2 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_lab_3' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_lab_3 \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_lab' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_lab_mean \
# --seed 97 \
# --workers $workers\

# #hsv

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_hsv' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_hsv \
# --seed 97 \
# --workers $workers\

# #hed

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_hed' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_hed \
# --seed 97 \
# --workers $workers\

# #integration

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_concat' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_concat \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_avg' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_avg \
# --seed 97 \
# --workers $workers\

# python /root/autodl-tmp/RandStainNA-master/segmentation/CIAtrain/CIA_all.py \
# --model 'CIAnet_weighted_avg' \
# --dataset $dataset_path \
# --epochs 60 --batch-size $b \
# --lr $lr \
# --output 60seg_weighted_avg \
# --seed 97 \
# --workers $workers\





