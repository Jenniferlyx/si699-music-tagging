import csv
import os
import random
import platform
import torch
import librosa
from tqdm import tqdm
import numpy as np
import glob
from sklearn.preprocessing import LabelBinarizer
import torchaudio
random.seed(42)

TAGS = ['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer',
        'genre---atmospheric', 'genre---indie', 'genre---techno', 'genre---newage',
        'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop',
        'genre---chillout', 'genre---metal', 'genre---lounge', 'genre---reggae',
        'genre---popfolk', 'genre---orchestral', 'genre---poprock', 'genre---trance',
        'genre---dance', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical',
        'genre---electronic', 'genre---world', 'genre---experimental', 'genre---folk',
        'genre---triphop', 'genre---jazz', 'genre---funk', 'genre---pop',
        'instrument---strings', 'instrument---drums', 'instrument---drummachine',
        'instrument---electricpiano', 'instrument---guitar', 'instrument---acousticguitar',
        'instrument---piano', 'instrument---electricguitar', 'instrument---violin',
        'instrument---voice', 'instrument---keyboard', 'instrument---bass', 'instrument---computer',
        'mood/theme---energetic', 'mood/theme---happy', 'mood/theme---emotional', 'mood/theme---film',
        'mood/theme---relaxing']


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tag_file, npy_root, config, type):
        # assert len(filenames) == len(labels), f'Inconsistent length of filenames and labels.'
        self.tag_file = tag_file
        self.npy_root = npy_root
        self.config = config
        self.mlb = LabelBinarizer().fit(TAGS)
        self.data = []
        self.labels = []
        self.type = type
        # transform waveform into spectrogram
        self.prepare_data()
        # self.length = self.data[0].shape[0]
        # make sure all of the data has the same dimension
        self.length = int(
            (10 * self.config['sample_rate'] + self.config['hop_length'] - 1) // self.config['hop_length'])

        print('Dataset will yield mel spectrogram {} data samples in shape (1, {}, {})'.format(len(self.data),
                                                                                               self.config['n_mels'],
                                                                                               self.length))
        # self.transforms = transforms

        # # Calculate length of clip this dataset will make
        # self.sample_length = int((cfg.clip_length * cfg.sample_rate + cfg.hop_length - 1) // cfg.hop_length)

        # # Test with first file
        # assert self[0][0].shape[-1] == self.sample_length, f'Check your files, failed to load {filenames[0]}'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert 0 <= index < len(self)
        mel_spec = self.data[index]
        # Padding if sample is shorter than expected - both head & tail are filled with 0s
        pad_size = self.length - mel_spec.shape[-1]
        if pad_size > 0:
            offset = pad_size // 2
            mel_spec = np.pad(mel_spec, ((0, 0), (0, 0), (offset, pad_size - offset)), 'constant')
        
        # Random crop
        crop_size = mel_spec.shape[-1] - self.length
        if crop_size > 0:
            start = np.random.randint(0, crop_size)
            mel_spec = mel_spec[..., start:start + self.length]
        # # Apply augmentations
        # if self.transforms is not None:
        #     log_mel_spec = self.transforms(log_mel_spec)
        return torch.Tensor(mel_spec), self.labels[index]

    def read_file(self):
        tracks = {}
        with open(self.tag_file) as fp:
            reader = csv.reader(fp, delimiter='\t')
            next(reader, None)  # skip header
            for row in reader:
                if not os.path.exists(os.path.join(self.npy_root, row[3].replace('.mp3', '.npy'))):
                    continue
                track_id = row[3].replace('.mp3', '.npy')
                tracks[track_id] = row[5:]
        return tracks

    def prepare_data(self):
        tracks = self.read_file()
        whole_filenames = []
        for id in tracks:
            whole_filenames.append(os.path.join(self.npy_root, id))
        train_size = int(len(whole_filenames) * 0.1)
        val_size = int(len(whole_filenames) * 0.02)
        filenames = []
        random.shuffle(whole_filenames)
        if self.type == 'train':
            filenames = whole_filenames[:train_size]
        if self.type == 'valid':
            filenames = whole_filenames[train_size:(train_size+val_size)]
        if self.type == 'test':
            filenames = whole_filenames[(train_size+val_size):]
        for filename in tqdm(filenames):
            waveform = np.load(filename)
            mel = librosa.feature.melspectrogram(y=waveform,
                                                 sr=self.config['sample_rate'],
                                                 n_fft=self.config['n_fft'],
                                                 hop_length=self.config['hop_length'],
                                                 n_mels=self.config['n_mels'],
                                                 fmin=self.config['fmin'],
                                                 fmax=self.config['fmax'])
            # mel = torchaudio.transforms.MelSpectrogram(waveform,
            #                                            sample_rate=self.config['sample_rate'],
            #                                            n_fft=self.config['n_fft'],
            #                                            f_min=self.config['fmin'],
            #                                            f_max=self.config['fmax'],
            #                                            n_mels=self.config['n_mels'])
            self.data.append(mel)
            id = os.path.join(filename.split('/')[-2], filename.split('/')[-1])
            self.labels.append(np.sum(self.mlb.transform(tracks[id]), axis=0))
