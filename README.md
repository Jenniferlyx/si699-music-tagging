## Load environment

### Using conda

The python environment is saved in file `env.yml`. To create the environment using conda from `env.yml` file, run:
```
conda env create -f env.yml
```
Then, activate the environment `si699-music-tagging`:
```
conda activate si699-music-tagging
```

To modify the environment, run:

```
conda update --all
conda env export --no-builds > env.yml
```

### Using pyvenv


## Prepare data

### Download dataset

The codes of downloading the dataset from `mtg-jamendo-dataset` is modified from [1]. Run:
```
bash download_audio.sh ${NUM_FOLDERS_TO_DOWNLOAD}
```
to download `.mp3` files from `raw_30s` to directories under `data/raw_data`.

### Convert `.mp3` file to `.npy`

```
python preprocessing/convert_npy.py --data_path data/raw_data \
                                    --output_root data/npy \
                                    --override True \
                                    --duration 60
```
to convert audio files into `.npy` files under the directory `data/npy`.

## Train

```
python run/train.py --tag_file data/autotagging_top50tags.tsv \
                    --npy_root data/npy \
                    --batch_size 4 \
                    --learning_rate 1e-4 \
                    --num_epochs 10 \
```

The script will create a tensorboard monitor under `runs`. To activate tensorboard, run:
```
python ${PATH_TO_TENSOrBOARD_PACKAGE}/tensorboard/main.py --logdir=runs
```