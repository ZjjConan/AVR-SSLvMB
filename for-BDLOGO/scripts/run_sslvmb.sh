#!/usr/bin/env bash
cd ..

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 train_meta_sslvmb.py --config configs/train_sslvmb_shapebd.yaml --gpu 0,1,2,3 --seed 123 --fp16
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 train_meta_sslvmb.py --config configs/train_sslvmb_shapebd.yaml --gpu 0,1,2,3 --seed 124 --fp16
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 train_meta_sslvmb.py --config configs/train_sslvmb_shapebd.yaml --gpu 0,1,2,3 --seed 125 --fp16