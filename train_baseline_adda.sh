#!/bin/bash
# setting training config here
batch_size=128
epochs=90
lr=0.001
lr_d=0.001
step=30
#lamda=0.2 # usps-mnistm
lamda=0.3
save_dir="checkpoint/baseline/mnistm-svhn"
train_dataset="mnistm"
test_dataset="svhn"
phase="train_adda"
checkpoint=""

config="--batch_size ${batch_size} --epochs ${epochs} --lr ${lr} --lr_d ${lr_d} --step ${step} --lamda ${lamda} --save_dir ${save_dir} --phase ${phase} --train_dataset ${train_dataset} --test_dataset ${test_dataset}"
#config="${config} --checkpoint ${checkpoint}" # uncomment this line to load checkpoint

run="python3 baseline.py ${config}"

echo "${run}"
${run}
