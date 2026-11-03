### Dataset Structure

Please prepare datasets with the following structure:

```markdown
your_dataset_root_dir/

    ├─I-RAVEN (RAVEN or RAVEN-FAIR)
    │  ├─center_single
    │  ├─distribute_four
    │  ├─distribute_nine
    │  ├─in_center_single_out_center_single
    │  ├─in_distribute_four_out_center_single
    │  ├─left_center_single_right_center_single
    │  └─up_center_single_down_center_single
    ├─PGM
    │  └─neutral
    |  └─interpolation
    |  └─extrapolation
    |  └─... (other sub-datasets)
```

### Training and Evaluation

```python
# Example for training a model on I-RAVEN
python main.py --dataset-name I-RAVEN --dataset-dir "your_dataset_root_dir" --gpu 0,1,2,3 --fp16 \
                -a sslvmb --block-drop 0.1 --classifier-drop 0.5 \
                --batch-size 128 --lr 0.001 --wd 1e-5 \
                --num-extra-stages 3 --unsupervised-training \
                --ckpt ckpts/ -p 50
```

```python
# Example for training models with multiple rounds
bash run_raven.sh (on RAVENs) or run_pgm.sh (on PGM)
```

```python
# Example for evaluating models
# using "--show-detail" to present detailed results for each configuration on RAVENs
python main.py --dataset-name I-RAVEN --dataset-dir "your_dataset_root_dir" --gpu 0,1,2,3 --fp16 \
               -a sslvmb --batch-size 128 --num-extra-stages 3 --ckpt ckpts/ -p 50 \
               -e --resume your_checkpoint_dir/model_best.pth.tar

# or see details in test_raven.sh
```
More running commands can be found in "run_raven.sh" and "run_pgm.sh" scripts.