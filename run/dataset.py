import os
import random
import torch
import librosa
from tqdm import tqdm
import numpy as np
import glob
from sklearn.preprocessing import LabelBinarizer
import csv
random.seed(42)
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

# TAGS = ['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer',
#         'genre---atmospheric', 'genre---indie', 'genre---techno', 'genre---newage',
#         'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop',
#         'genre---chillout', 'genre---metal', 'genre---lounge', 'genre---reggae',
#         'genre---popfolk', 'genre---orchestral', 'genre---poprock', 'genre---trance',
#         'genre---dance', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical',
#         'genre---electronic', 'genre---world', 'genre---experimental', 'genre---folk',
#         'genre---triphop', 'genre---jazz', 'genre---funk', 'genre---pop',
#         'instrument---strings', 'instrument---drums', 'instrument---drummachine',
#         'instrument---electricpiano', 'instrument---guitar', 'instrument---acousticguitar',
#         'instrument---piano', 'instrument---electricguitar', 'instrument---violin',
#         'instrument---voice', 'instrument---keyboard', 'instrument---bass', 'instrument---computer',
#         'mood/theme---energetic', 'mood/theme---happy', 'mood/theme---emotional', 'mood/theme---film',
#         'mood/theme---relaxing']


def clip(mel, length):
    # Padding if sample is shorter than expected - both head & tail are filled with 0s
    pad_size = length - mel.shape[-1]
    if pad_size > 0:
        offset = pad_size // 2
        mel = np.pad(mel, ((0, 0), (0, 0), (offset, pad_size - offset)), 'constant')

    # Random crop
    crop_size = mel.shape[-1] - length
    if crop_size > 0:
        start = np.random.randint(0, crop_size)
        mel = mel[..., start:start + length]
    return mel


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tag_file, npy_root, config, tags, data_type, feature_extractor_type):
        # assert len(filenames) == len(labels), f'Inconsistent length of filenames and labels.'
        self.npy_root = npy_root
        self.config = config
        self.tag_file = tag_file
        self.tags = tags
        self.mlb = LabelBinarizer().fit(self.tags)
        self.data = []
        self.labels = []
        self.data_type = data_type
        # transform waveform into spectrogram
        self.prepare_data()
        self.feature_extractor_type = feature_extractor_type
        # print('Dataset will yield mel spectrogram {} data samples in shape (1, {}, {})'.format(len(self.data),
        #                                                                                        self.config['n_mels'],
        #                                                                                        self.length))
        # self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert 0 <= index < len(self)
        waveform = self.data[index]
        target = self.labels[index]
        if self.feature_extractor_type == 'melspec':
            mel_spec = librosa.feature.melspectrogram(y=waveform,
                                                 sr=self.config['sample_rate'],
                                                 n_fft=self.config['n_fft'],
                                                 hop_length=self.config['hop_length'],
                                                 n_mels=self.config['n_mels'],
                                                 fmin=self.config['fmin'],
                                                 fmax=self.config['fmax'])
            mel_spec = torch.Tensor(mel_spec)
            print(mel_spec.shape)
        if self.feature_extractor_type == 'ast':
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                sampling_rate=self.config['sample_rate'],
                num_mel_bins=self.config['n_mels']
            )
            encoding = feature_extractor(waveform, sampling_rate=self.config['sample_rate'], annotations=target, return_tensors="pt")
            mel_spec = encoding['input_values'].squeeze()
            mel_spec = torch.transpose(mel_spec, 0, 1)
        if self.feature_extractor_type == 'wav2vec':
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base-960h"
                # "m3hrdadfi/wav2vec2-base-100k-voxpopuli-gtzan-music"
            )
            encoding = feature_extractor(waveform, sampling_rate=self.config['sample_rate'],
                                         return_tensors="pt")
            mel_spec = encoding['input_values'].squeeze()
            # print(mel_spec.shape)
        return mel_spec, target

    def read_file(self):
        tracks = {}
        with open(self.tag_file) as fp:
            reader = csv.reader(fp, delimiter='\t')
            next(reader, None)  # skip header
            for row in reader:
                if not os.path.exists(os.path.join(self.npy_root, row[3].replace('.mp3', '.npy'))):
                    continue
                track_id = row[3].replace('.mp3', '.npy')
                tags = []
                for tag in row[5:]:
                    tags.append(tag.split('---')[-1])
                tracks[track_id] = tags
        return tracks

    def prepare_data(self):
        tracks_dict = self.read_file()
        whole_filenames = sorted(glob.glob(os.path.join(self.npy_root, "*/*.npy")))
        train_size = int(len(whole_filenames) * 0.8)
        # val_size = int(len(whole_filenames) * 0.95)
        filenames = []
        random.shuffle(whole_filenames)
        if self.data_type == 'train':
            filenames = whole_filenames[:train_size]
        if self.data_type == 'valid':
            filenames = whole_filenames[train_size:]
        for filename in tqdm(filenames):
            waveform = np.load(filename)
            self.data.append(waveform)
            id = os.path.join(filename.split('/')[-2], filename.split('/')[-1])
            self.labels.append(np.sum(self.mlb.transform(tracks_dict[id]), axis=0))
