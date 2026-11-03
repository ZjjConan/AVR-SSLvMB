export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH="adding your model path here"/model_best.pth.tar
DATA_PATH="your_dataset_root_dir"/Datasets/

python main.py --dataset-name I-RAVEN --dataset-dir ${DATA_PATH} --gpu 0,1,2,3 --fp16 \
            -a sslvmb --batch-size 128 --num-extra-stages 3 --ckpt ckpts/ -p 50 \
            -e --resume ${MODEL_PATH}