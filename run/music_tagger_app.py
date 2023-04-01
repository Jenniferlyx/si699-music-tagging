import streamlit as st
import librosa
import yaml
import os
from models import *
from dataset import *
import torch

def convert(filename):
    waveform, _ = librosa.load(filename, sr=config['sample_rate'], mono=True)
    mel = librosa.feature.melspectrogram(y=waveform,
                                         sr=config['sample_rate'],
                                         n_fft=config['n_fft'],
                                         hop_length=config['hop_length'],
                                         n_mels=config['n_mels'],
                                         fmin=config['fmin'],
                                         fmax=config['fmax'])
    length = int(
            (10 * config['sample_rate'] + config['hop_length'] - 1) // config['hop_length'])
    mel = clip(mel, length)
    return mel


def predict(mel, config, model_name):
    if model_name == 'SampleCNN':
        model = torch.load('model/samplecnn.pt')
    else:
        model = torch.load('model/samplecnn.pt')
    model.eval()
    mel = torch.tensor(mel).unsqueeze(0)
    pred = model(mel)
    prob, indices = pred.topk(k=config['topk'])
    output = []
    for i in range(config['topk']):
        output.append((TAGS[i], prob[0][i]))
    return sorted(output, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    st.title("SI699 Music Tagger")
    # st.write(
    #     "[![Star](<https://img.shields.io/github/stars/Jenniferlyx/<repo>.svg?logo=github&style=social)](<https://github.com/Jenniferlyx/si699-music-tagging.git)")
    st.write("This is a Web App to tag genre of music")
    uploaded_file = st.file_uploader("Please Upload Mp3 Audio File Here or Use Demo Of App Below using Preloaded Music",
                                    type=["mp3"])
    if uploaded_file is None:
        st.write("Please upload an mp3 file")
    else:
        st.audio(uploaded_file.read())
        model_option = st.selectbox(
            'Which model would you like to use?',
            ('SampleCNN', 'CRNN'))
        file_path = os.path.join('data', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        mel = convert(file_path)
        output = predict(mel, config, model_option)
        st.write("Result:")
        for tag, prob in output:
            st.write("{}: {}".format(tag, prob))
        os.remove(file_path)
