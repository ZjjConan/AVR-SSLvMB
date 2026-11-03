### Dataset Structure

First download all **ShapeBongard_V2** dataset from [[this website]](https://drive.google.com/file/d/1-1j7EBriRpxI-xIVqE6UEXt-SzoWvwLx/view?usp=sharing), and then unzip it into `materials` directory. This dataset contains 12,000 problems which are 3,600 `free-form shape problems`, 4,000 `basic shape problems`, and 4,400 `abstract shape problems`.

After `unzip` the dataset, you can find the following structure:

```markdown
your_dataset_root_dir/

    ├─ShapeBongard_V2
    │  ├─bd
    │  ├─ff
    │  ├─hd
    │  ├─ShapeBongard_V2_split.json
```

### Training and Evaluation
First, you should replace `your_dataset_root_dir` in `configs/train_sslvmb_shapebd.yaml` with your real dataset directory.

Second, after that, you can run the following command to run the model with three different seeds:

```python
cd scripts
bash run_sslvmb.sh 
# see detailed parameters in run_sslvmb.sh
```
