export CUDA_VISIBLE_DEVICES=0,1,2,3

TRAILS=(0 1 2 3 4)
SAVE_PATH_PREFIX=ckpts/sslvmb/exp
DATA_PATH=../Datasets/

for tr in ${TRAILS[*]}
do
    SAVE_PATH=${SAVE_PATH_PREFIX}_${tr}/

    python main.py --dataset-name I-RAVEN --dataset-dir ${DATA_PATH} --gpu 0,1,2,3 --fp16 \
                -a sslvmb --block-drop 0.1 --classifier-drop 0.5 \
                --batch-size 128 --lr 0.001 --wd 1e-5 \
                --num-extra-stages 3 --unsupervised-training \
                --ckpt ${SAVE_PATH} -p 50

    python main.py --dataset-name RAVEN-FAIR --dataset-dir ${DATA_PATH} --gpu 0,1,2,3 --fp16 \
                -a sslvmb --block-drop 0.1 --classifier-drop 0.5 \
                --batch-size 128 --lr 0.001 --wd 1e-5 \
                --num-extra-stages 3 --unsupervised-training \
                --ckpt ${SAVE_PATH} -p 50

    python main.py --dataset-name RAVEN --dataset-dir ${DATA_PATH} --gpu 0,1,2,3 --fp16 \
                -a sslvmb --block-drop 0.1 --classifier-drop 0.5 \
                --batch-size 128 --lr 0.001 --wd 1e-5 \
                --num-extra-stages 3 --unsupervised-training \
                --ckpt ${SAVE_PATH} -p 50
done