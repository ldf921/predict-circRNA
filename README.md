# predict-circRNA

## How to Run
Run train.py, the result model will be saved to MODEL_DIR/model_xx.ckpt. Log will be saved to MODEL_DIR/train.log.
```
python ./train.py ACTION -o MODEL_DIR [-c CHECK_POINT --seed RANDOM_SEED]
```

### Parameters
RANDOM_SEED: random seed for split training data and validation data
ACTION: Could be `train` or `test`. 
CHECK_POINT: Load a model. (modelxxxx.ckpt)

### Example
```
python ./train.py train -o model_1
python ./train.py test -c model_1/model6000.ckpt
```

## Preparation
Please copy file hg19.fa, hsa_hg19_Rybak2015.bed, all_exons.bed into the root directory of this repo.

If you have multiple GPU, set environment varaible `CUDA_VISIBLE_DEVICES` to index of GPUs you want to use.

### Dependecies
Install python3, [tensorflow](https://github.com/tensorflow/tensorflow), biopython, scipy, sklearn, tqdm.
```bash
pip3 install biopython
pip3 install scipy
pip3 install sklearn
pip3 install tqdm
```
