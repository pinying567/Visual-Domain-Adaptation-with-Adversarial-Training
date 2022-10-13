#!/bin/bash
# setting training config here
batch_size=128
epochs=90
lr=0.001
step=30
save_dir="checkpoint/baseline/mnistm-mnistm"
train_dataset="mnistm"
test_dataset="mnistm"
phase="train"
checkpoint=""

config="--batch_size ${batch_size} --epochs ${epochs} --lr ${lr} --step ${step} --save_dir ${save_dir} --phase ${phase} --train_dataset ${train_dataset} --test_dataset ${test_dataset}"
#config="${config} --checkpoint ${checkpoint}" # uncomment this line to load checkpoint

run="python3 baseline.py ${config}"

echo "${run}"
${run}
