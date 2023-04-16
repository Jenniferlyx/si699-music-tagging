import streamlit as st
import yaml
from dataset import *
from train import *
import torch
import pandas as pd
from models import *
import librosa
import csv
torch.manual_seed(0)
np.random.seed(0)
feature_extractor_type = 'raw'
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def convert(filename):
    waveform, _ = librosa.load(filename, sr=config['sample_rate'], mono=True)
    desired_length = config['duration'] * config['sample_rate']
    if len(waveform) < desired_length:
        waveform = np.pad(waveform, (0, desired_length - len(waveform)), mode='mean')
    # Truncate the waveform array if it is longer than the desired length
    if len(waveform) > desired_length:
        waveform = np.random.choice(waveform, desired_length)
    assert len(waveform) == desired_length, "{} vs {}".format(len(waveform), desired_length)

    if feature_extractor_type == 'raw':
        mel_spec = torch.Tensor(waveform)
    if feature_extractor_type == 'autoextractor':
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            sampling_rate=config['sample_rate'],
            num_mel_bins=config['n_mels']
        )
        encoding = feature_extractor(waveform, sampling_rate=config['sample_rate'],
                                     return_tensors="pt")
        mel_spec = encoding['input_values'].squeeze()
        mel_spec = torch.transpose(mel_spec, 0, 1)
    if feature_extractor_type == 'wav2vec':
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        encoding = feature_extractor(waveform, sampling_rate=config['sample_rate'],
                                     return_tensors="pt")
        mel_spec = encoding['input_values'].squeeze()
    return mel_spec

def get_tags(tag_file, isMap):
    if isMap:
        f = open('tag_categorize.json')
        data = json.load(f)
        categorize = {}
        for k, v in data.items():
            for i in v[1:-1].split(', '):
                categorize[i] = k
    total_tags = []
    with open(tag_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            tags = []
            for tag in row[5:]:
                if isMap:
                    tags.append(categorize[tag.split('---')[-1]])
                else:
                    tags.append(tag.split('---')[-1])
            total_tags += list(set(tags))
    return list(set(total_tags))

def predict(model_path, mel, config):
    TAGS = get_tags('autotagging_moodtheme.tsv', True)
    if ("samplecnn" in model_path):
        model = SampleCNN(len(TAGS), config)
    elif ("musicnn" in model_path):
        model = Musicnn(len(TAGS), config)
    elif ("fcn" in model_path):
        model = FCN(len(TAGS), config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    mel = mel.unsqueeze(0)
    pred = model(mel)
    prob, indices = pred.topk(k=config['topk'])
    indices = indices.detach().numpy()
    tags = []
    probs = []
    for i in range(config['topk']):
        tags.append(TAGS[indices[0][i]].capitalize())
        probs.append(np.round(prob[0][i].detach().numpy(), 4))
    output_df = pd.DataFrame(list(zip(tags, probs)), columns=['Tag', 'Probability'])
    output_df = output_df.sort_values(by='Probability', ascending=False)
    return output_df

def process_uploaded_file(file_path):
    waveform = convert(file_path)
    output_df = predict('model/musicnn_best_score_0.0001_16.pt', waveform, config)
    st.write("Result:")
    st.dataframe(output_df)

if __name__ == '__main__':
    st.set_page_config(layout='centered')
    st.title("SI699 Music Tagger")
    st.write("Welcome!! This is a Web App to tag genre, instrument, and mood of music.")
    sample_button = st.sidebar.button('Try a sample')
    sample_path = 'data/sample01.mp3'
    if sample_button:
        st.audio(sample_path)
        st.write('Getting sample result...')
        process_uploaded_file(sample_path)
    uploaded_file = st.sidebar.file_uploader("Please Upload Mp3 Audio File Here.",
                                    type=["mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file.read())
        file_path = os.path.join('data', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.write("Getting the result...")
        process_uploaded_file(file_path)
        os.remove(file_path)
