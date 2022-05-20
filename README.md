# RBPGAN-Video-Super-Resolution

This repo is for the Senior Project done by Dareen Hussein, Israa Fahmy, Marwah Sulaiman, Mohammed Barakat, Mohammed El-Naggar, and Zahraa Shehabeldin from the American University in Cairo, Spring 22. All needed code and documentation can be found in this repo.

## Dependencies:
Ubuntu 20.04.3 LTS

Anaconda Environment with PyTorch 1.4, Python 3.6

Python packages: numpy, matplotlib, opencv-python, pyyaml, lmdb

## Hardware setup:
64GB of DDR4 RAM, 2.80GHz

Intel Core i9-10900F CPU

NIVIDIA GetForce RTX 3090 (1 x 24 GB) GPU

## Training:

1- Download the official training dataset as follows, rename to VimeoTecoGAN/Raw, and place under ./data.

```
# Install youtube-dl for online video downloading
pip install --user --upgrade youtube-dl

# take a look of the parameters first:
python3 dataPrepare.py --help

python3 dataPrepare.py --start_id 2000 --duration 120 --disk_path TrainingDataPath --TEST

# This will create 308 subfolders under TrainingDataPath, each with 120 frames, from 28 online videos
python3 dataPrepare.py --start_id 2000 --duration 120 --REMOVE --disk_path TrainingDataPath

```

2- Generate LMDB for the ground truth data

```
python ./scripts/create_lmdb.py --dataset VimeoTecoGAN --raw_dir ./data/VimeoTecoGAN/Raw --lmdb_dir ./data/VimeoTecoGAN/GT.lmdb
```

3- Train the RBPGAN model. (Note that the gpu used can be specified/changed in train.sh) Also, you can find the training log in ./experiments_BD/RBPGAN/RBPGAN_VimeoTecoGAN_4xSR_2GPU/train/train.log.
```
bash ./train.sh BD RBPGAN/RBPGAN_VimeoTecoGAN_4xSR_2GPU
```

4- If you want to conduct a new experiment, create a new folder under experiments_BD/ , add traint.yml and test.yml with the same format as the other experiments but with the desired parameters. Then train as step 3 (bash ./train.sh BD RBPGAN/"experiment_name" )


## Testing:

1- Download Vid4 and ToS3 datasets
```
bash ./scripts/download/download_datasets.sh BD
```
2- Test RBPGAN. You will find the results (generated frames) in ./results. (Note that the model and gpu used can be specified in test.sh)
```
bash ./test.sh BD RBPGAN/RBPGAN_VimeoTecoGAN_4xSR_2GPU
```


## Acknowledgements
This code is based on [TecoGAN-TensorFlow](https://github.com/thunil/TecoGAN) and  [TecoGAN-PyTorch
](https://github.com/skycrapers/TecoGAN-PyTorch)

If you have any further questions or clarifications don't hesitate to contact me : 'israafahmy@aucegypt.edu'.
