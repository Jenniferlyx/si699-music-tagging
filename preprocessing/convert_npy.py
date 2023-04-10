import librosa
import argparse
from tqdm import tqdm
import glob
import os
import numpy as np
import yaml
import csv
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mp3_data_root', type=str, default='data/autotagging_moodtheme')
parser.add_argument('--output_root', type=str, default='data/npy')
parser.add_argument('--tag_file', type=str, default='data/autotagging_moodtheme.tsv')
parser.add_argument('--override', type=bool, default=False,
                    help="if set to True, override the npy files")
args = parser.parse_args()
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
np.random.seed(config['seed'])


def get_waveform(mp3_filename, npy_filename, desired_length):
    x, _ = librosa.load(mp3_filename, sr=config['sample_rate'], mono=True)
    # Pad the waveform array with mean values if it is shorter than the desired length
    if len(x) < desired_length:
        x = np.pad(x, (0, desired_length - len(x)), mode='mean')
    # Truncate the waveform array if it is longer than the desired length
    if len(x) > desired_length:
        x = np.random.choice(x, desired_length)
    assert len(x) == desired_length, "{} vs {}".format(len(x), desired_length)
    with open(npy_filename, 'wb') as fp:
        np.save(fp, x)

def read_file():
    track_set = set()
    with open(args.tag_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)
        for row in reader:
            track_set.add(row[3].replace('.mp3', '.npy'))
    return track_set


if __name__ == '__main__':
    # 1. create directories if not exists
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)
    for dir in os.listdir(args.mp3_data_root):
        if '.' in dir:
            continue
        output_dir = os.path.join(args.output_root, dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # 2. convert mp3 to npy if matched with tag tsv file
    track_set = read_file()
    desired_length = config['duration'] * config['sample_rate']
    for mp3_filename in tqdm(sorted(glob.glob(os.path.join(args.mp3_data_root, "*/*.mp3")))):
        file_id = mp3_filename.split('/')[-2] + '/' + mp3_filename.split('/')[-1].split('.')[0] + '.npy'
        if file_id not in track_set:
            continue
        npy_file_name = os.path.join(args.output_root, file_id)
        if os.path.isfile(npy_file_name) and not args.override:
            continue
        get_waveform(mp3_filename, npy_file_name, desired_length)
        # x, _ = librosa.load(file, sr=config['sample_rate'], mono=True)
        # with open(npy_file, 'wb') as f:
        #     np.save(f, x)

