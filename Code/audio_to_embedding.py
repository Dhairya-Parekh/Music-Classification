import librosa as lb
import numpy as np
import torch
import argparse as arg
import pickle as pkl
from FFN import FFN
import pandas as pd


SAMPLE_RATE = 22050
def audio_to_embedding(audio_file):
    music,sr = lb.load(audio_file,sr=SAMPLE_RATE)
    music_vector = np.zeros((1,29))
    music_vector[0][0] = lb.feature.chroma_stft(y=music,sr=SAMPLE_RATE).mean()
    music_vector[0][1] = lb.feature.rms(y=music).mean()
    music_vector[0][2] = lb.feature.spectral_centroid(y=music,sr=SAMPLE_RATE).mean()
    music_vector[0][3] = lb.feature.spectral_bandwidth(y=music,sr=SAMPLE_RATE).mean()
    music_vector[0][4] = lb.feature.spectral_rolloff(y=music,sr=SAMPLE_RATE).mean()
    music_vector[0][5] = lb.feature.zero_crossing_rate(y=music).mean()
    music_vector[0][6] = lb.effects.harmonic(y=music).mean()
    music_vector[0][7] = lb.effects.percussive(y=music).mean()
    music_vector[0][8] = lb.beat.tempo(y=music,sr=SAMPLE_RATE).mean()
    mfcc = lb.feature.mfcc(y=music,sr=SAMPLE_RATE,n_mfcc=20)
    for i in range(1,21):
        music_vector[0][8+i] = mfcc[i-1].mean()
    return torch.Tensor(music_vector)

if __name__=='__main__':
    parser = arg.ArgumentParser(prog = 'Music Genre Classifier')
    parser.add_argument('filename')
    args = parser.parse_args()
    
    df = pd.read_csv("../Dataset/features_30_sec.csv")
    
    df.drop(['filename','length'],axis=1,inplace=True)
    cols = df.columns
    
    for col in cols:
        if col.endswith("var"):
            df.drop([col],axis=1,inplace=True)

    # normalization
    Y = df['label']
    df.drop(['label'],axis=1,inplace=True)
    mean = df.mean(axis=0)
    sd = df.std(axis=0)

    features = (audio_to_embedding(args.filename) - torch.tensor(mean,dtype=torch.float32)) / torch.tensor(sd,dtype=torch.float32)

    model = pkl.load(open("../Model/ffn.pkl", "rb"))

    pred_genre = torch.argmax(model(features))
    labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    print(labels[pred_genre])