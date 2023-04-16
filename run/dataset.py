import os
import random
import torch
from tqdm import tqdm
import numpy as np
import glob
from sklearn.preprocessing import LabelBinarizer
random.seed(0)
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor, BertTokenizer

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tracks_dict, npy_root, config, tags, data_type, feature_extractor_type):
        self.npy_root = npy_root
        self.config = config
        self.tracks_dict = tracks_dict
        self.tags = tags
        self.mlb = LabelBinarizer().fit(self.tags)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.title_dict = {}
        self.prepare_title()
        self.data = []
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        self.data_type = data_type
        self.prepare_data()
        self.feature_extractor_type = feature_extractor_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert 0 <= index < len(self)
        waveform = self.data[index]
        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        target = self.labels[index]
        if self.feature_extractor_type == 'raw':
            mel_spec = torch.Tensor(waveform)
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
            )
            encoding = feature_extractor(waveform, sampling_rate=self.config['sample_rate'],
                                         return_tensors="pt")
            mel_spec = encoding['input_values'].squeeze()
        return mel_spec, input_ids, attention_mask, target
    
    def prepare_title(self):
        whole_filenames = sorted(glob.glob(os.path.join(self.npy_root, "*/*.npy")))
        titles = []
        for filename in whole_filenames:
            file_id = os.path.join(filename.split('/')[-2], filename.split('/')[-1].split('.')[0])
            titles.append(self.tracks_dict[file_id][1])
        encoding = self.tokenizer(titles, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        for idx, filename in enumerate(whole_filenames):
            file_id = os.path.join(filename.split('/')[-2], filename.split('/')[-1].split('.')[0])
            self.title_dict[file_id] = (input_ids[idx], attention_mask[idx])
        
    def prepare_data(self):
        whole_filenames = sorted(glob.glob(os.path.join(self.npy_root, "*/*.npy")))
        filenames = []
        random.shuffle(whole_filenames)
        train_size = int(len(whole_filenames) * 0.8)
        if self.data_type == 'train':
            filenames = whole_filenames[:train_size]
        if self.data_type == 'valid':
            filenames = whole_filenames[train_size:]
        for filename in tqdm(filenames):
            file_id = os.path.join(filename.split('/')[-2], filename.split('/')[-1].split('.')[0])
            if file_id not in self.tracks_dict:
                # check non-exit file
                print(file_id)
                continue
            self.data.append(np.load(filename))
            self.input_ids.append(self.title_dict[file_id][0])
            self.attention_mask.append(self.title_dict[file_id][1])
            self.labels.append(np.sum(self.mlb.transform(self.tracks_dict[file_id][0]), axis=0))
