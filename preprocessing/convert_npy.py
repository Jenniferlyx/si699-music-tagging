import librosa
import argparse
from tqdm import tqdm
import glob
import os
import numpy as np
import yaml
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='data/raw_data')
parser.add_argument('--output_root', type=str, default='data/npy')
parser.add_argument('--override', type=bool, default=False,
                    help="if set to True, override the npy files")
# parser.add_argument('--duration', type=int, default=120,
#                     help="Set the fixed duration in seconds to cut the audio length")
args = parser.parse_args()
with open('run/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
np.random.seed(0)


# def get_waveform(file, desired_length, npy_file):
    # x, _ = librosa.load(file, sr=config['sample_rate'], mono=True)
    # Pad the waveform array with zeros if it is shorter than the desired length
    # if len(x) < desired_length:
    #     x = np.pad(x, (0, desired_length - len(x)), mode='constant')
    # # Truncate the waveform array if it is longer than the desired length
    # if len(x) > desired_length:
    #     x = np.random.choice(x, desired_length)
    # assert len(x) == desired_length, "{} vs {}".format(len(x), desired_length)
    # with open(npy_file, 'wb') as f:
    #     np.save(f, x)


if __name__ == '__main__':
    file_path = sorted(glob.glob(os.path.join(args.data_path, "*/*.mp3")))
    # desired_length = args.duration*config['sample_rate']
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    for file in tqdm(file_path):
        npy_dir = os.path.join(args.output_root, file.split('/')[-2])
        npy_file = os.path.join(npy_dir, file.split('/')[-1].split('.')[0] + '.npy')
        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)
        if os.path.exists(npy_file) and args.override:
            os.remove(npy_file)
        if not os.path.exists(npy_file):
            # get_waveform(file, desired_length, npy_file)
            x, _ = librosa.load(file, sr=config['sample_rate'], mono=True)
            with open(npy_file, 'wb') as f:
                np.save(f, x)
