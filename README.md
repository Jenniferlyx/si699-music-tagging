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
to download `.mp3` files from `autotagging_moodtheme` to directories under `data/autotagging_moodtheme`.

### Convert `.mp3` file to `.npy`

```
python preprocessing/convert_npy.py --data_path data/autotagging_moodtheme \
                                    --output_root data/waveform \
                                    --override True
```
to convert audio files into `.npy` files under the directory `data/waveform`.

## Train

```
python run/train.py --tag_file data/autotagging_moodtheme.tsv \
                    --npy_root data/waveform \
                    --model samplecnn \
                    --transform raw \
                    --is_map True \
                    --is_title False \
                    --batch_size 4 \
                    --learning_rate 1e-4 \
                    --num_epochs 10 \
```

The waveform will be transformed to melspectrogram with the parameters specified in `run/config.yaml`. The input of the model should be melspectrograms.

The script will create a tensorboard monitor under `runs`. To activate tensorboard, run:
```
python ${PATH_TO_TENSOrBOARD_PACKAGE}/tensorboard/main.py --logdir=runs
```

## Reference
[1] Bogdanov, D., Won M., Tovstogan P., Porter A., & Serra X. (2019). The MTG-Jamendo Dataset for Automatic Music Tagging. Machine Learning for Music Discovery Workshop, International Conference on Machine Learning (ICML 2019).
