import librosa
import argparse
from tqdm import tqdm
import glob
import os
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='../../mtg-jamendo-dataset/sample_data')
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--override', type=bool, default=False)
args = parser.parse_args()

if __name__ == '__main__':
    file_path = glob.glob(os.path.join(args.data_path, "*/*.mp3"))
    npy_path = "../data/npy"
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    for file in tqdm(file_path):
        npy_dir = os.path.join(npy_path, file.split('/')[-2])
        npy_file = os.path.join(npy_dir, file.split('/')[-1].split('.')[0] + '.npy')
        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)
        if os.path.exists(npy_file) and args.override:
            os.remove(npy_path)
        if not os.path.exists(npy_file):
            try:
                x, _ = librosa.load(file, sr=args.sample_rate, mono=True, offset=0.0, duration=None)
                with open(npy_file, 'wb') as f:
                    np.save(f, x)
            except RuntimeError:
                # some audio files are broken
                print(file)
                continue
