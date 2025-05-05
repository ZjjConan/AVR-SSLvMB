export CUDA_VISIBLE_DEVICES=0,1,2,3

TRAILS=(0 1 2)
SAVE_PATH_PREFIX=ckpts/sslvmb/PGM/exp
DATA_PATH=../Datasets/

for tr in ${TRAILS[*]}
do
    SAVE_PATH=${SAVE_PATH_PREFIX}_${tr}/

    python main.py --dataset-name neutral --dataset-dir ${DATA_PATH}/PGM/ --gpu 0,1,2,3 --fp16 \
                -a sslvmb --block-drop 0.0 --classifier-drop 0.5 \
                --batch-size 256 --lr 0.001 --wd 1e-7 \
                --num-extra-stages 3 --unsupervised-training \
                --ckpt ${SAVE_PATH} -p 100

    python main.py --dataset-name interpolation --dataset-dir ${DATA_PATH}/PGM/ --gpu 0,1,2,3 --fp16 \
                -a sslvmb --block-drop 0.0 --classifier-drop 0.5 \
                --batch-size 256 --lr 0.001 --wd 1e-7 \
                --num-extra-stages 3 --unsupervised-training \
                --ckpt ${SAVE_PATH} -p 100

    # adding other sub-datasets for multiple trails
done