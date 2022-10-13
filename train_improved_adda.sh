#!/bin/bash
# setting training config here
batch_size=128
epochs=50
lr=0.00002 # 1e-5
lr_d=0.0005
step=50
## mnist-svhn
save_dir="checkpoint/improved/mnistm-svhn"
train_dataset="mnistm"
test_dataset="svhn"
checkpoint="checkpoint/mnistm-mnistm/last_checkpoint.pkl"

phase="train_adda"

config="--batch_size ${batch_size} --epochs ${epochs} --lr ${lr} --lr_d ${lr_d} --step ${step} --save_dir ${save_dir} --phase ${phase} --train_dataset ${train_dataset} --test_dataset ${test_dataset}"
config="${config} --checkpoint ${checkpoint}" # uncomment this line to load checkpoint

run="python3 improved.py ${config}"

echo "${run}"
${run}