import streamlit as st
import librosa
import yaml
import io
from models import *
from dataset import *
from pydub import AudioSegment

def convert(file):
    audio = AudioSegment.from_file(file, 'mp3')
    # Convert the file to a WAV format that librosa can read
    with io.BytesIO() as output_file:
        audio.export(output_file, format="wav")
        output_file.seek(0)
        waveform, _ = librosa.load(output_file, sr=config['sample_rate'], mono=True)
    mel = librosa.feature.melspectrogram(y=waveform,
                                         sr=config['sample_rate'],
                                         n_fft=config['n_fft'],
                                         hop_length=config['hop_length'],
                                         n_mels=config['n_mels'],
                                         fmin=config['fmin'],
                                         fmax=config['fmax'])
    return mel


def predict(mel, config):
    model = torch.load('model/fcn.pt')
    model.eval()
    pred = model(mel)
    prob, indices = pred.topk(k=config['topk'])
    output = []
    for i in range(config['topk']):
        output.append((TAGS[i], indices[i]))
    return sorted(output, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    st.write("Music Tagger")
    st.write("This is a Web App to tag genre of music")
    file = st.file_uploader("Please Upload Mp3 Audio File Here or Use Demo Of App Below using Preloaded Music",
                                    type=["mp3"])
    if file is None:
        st.text("Please upload an mp3 file")
    else:
        st.audio(file.read())
        mel = convert(file)
        output = predict(mel, config)
        st.write("Result:")
        for tag, prob in output:
            st.write("{}: {}".format(tag, prob))
