# predict-circRNA

## How to Run
Run train.py, the result model will be saved to MODEL_DIR/model_xx.ckpt. Log will be saved to MODEL_DIR/train.log.
```bash
python ./train.py -o (MODEL_DIR) --seed (RANDOM_SEED)
```

### Parameters
RANDOM_SEED: random seed for split training data and validation data

## Preparation
Please copy file hg19.fa, hsa_hg19_Rybak2015.bed, all_exons.bed into the root directory of this repo.

If you have multiple GPU, set environment varaible `CUDA_VISIBLE_DEVICES` to index of GPUs you want to use.

### Dependecies
Install python3, tensorflow, biopython, scipy, sklearn.
```bash
$ pip3 install biopython
$ pip3 install scipy
$ pip3 install sklearn
```
