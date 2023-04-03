import streamlit as st
import yaml
from dataset import *
import torch
from pytube import YouTube
import pandas as pd

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


def predict(mel, config):
    # if model_name == 'SampleCNN':
    #     model = torch.load('model/samplecnn.pt')
    # elif model_name == 'CRNN':
    #     model = torch.load('model/crnn.pt')
    # else:
    model = torch.load('model/samplecnn.pt')
    model.eval()
    mel = torch.tensor(mel).unsqueeze(0)
    pred = model(mel)
    prob, indices = pred.topk(k=config['topk'])
    category = []
    tags = []
    probs = []
    for i in range(config['topk']):
        category.append(TAGS[i].split('---')[0].capitalize())
        tags.append(TAGS[i].split('---')[1].capitalize())
        probs.append(np.round(prob[0][i].detach().numpy(), 4))
    output_df = pd.DataFrame(list(zip(category, tags, probs)), columns=['Category', 'Tag', 'Probability'])
    output_df = output_df.sort_values(by='Probability', ascending=False)
    return output_df

def process_uploaded_file(file_path):
    mel = convert(file_path)
    output_df = predict(mel, config)
    st.write("Result:")
    st.dataframe(output_df)

# def process_youtube(file_path):
#     url_link = st.text_input("Please give a url link of your Youtube music")
#     if url_link:
#         st.text('Loading music from Youtube...')
#         yt = YouTube(url_link)
#         if yt:
#             st.write(yt.title)
#             stream = yt.streams.filter(only_audio=True).first()
#             stream.download(filename=file_path)
#             st.audio(file_path)
#             return True
#         else:
#             st.warning('Please enter a valid YouTube video URL')



if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    st.set_page_config(layout='centered')
    st.title("SI699 Music Tagger")
    # st.write(
    #     "[![Star](<https://img.shields.io/github/stars/Jenniferlyx/si699-music-tagging.svg?logo=github&style=social)](<https://github.com/Jenniferlyx/si699-music-tagging.git)")
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

