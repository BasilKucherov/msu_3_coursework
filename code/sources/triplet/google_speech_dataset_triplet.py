import librosa
from torch.utils.data import Dataset
import torchaudio
import torch
import os
import pandas as pd
import numpy as np


class GoogleSpeechDatasetTriplet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, subset=None):
        self.google_speech_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
                
        self.labels = self.google_speech_frame['label'].unique()
        self.indices = [[] for _ in range(len(self.labels))]

        if subset != None:
            self.google_speech_frame = self.google_speech_frame[self.google_speech_frame.index.isin(subset.indices)]

        for i, label in enumerate(self.labels):
            self.indices[i] = self.google_speech_frame.index[self.google_speech_frame['label'] == label].tolist()
        
        self.transform_series_to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=40, n_fft=2048, hop_length=512)
        
            
    def __len__(self):
        return len(self.google_speech_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_name = os.path.join(self.root_dir, self.google_speech_frame.loc[idx][0])


        label = self.google_speech_frame.loc[idx][1]
        
        waveform, sample_rate = torchaudio.load(file_name, normalize=True)

        if self.transform != None:
            waveform = self.transform(waveform)

        if waveform.shape[1] < 16000:
            padding = torch.zeros((1, 16000 - waveform.shape[1]))
            waveform = torch.cat((waveform, padding), dim=1)
        M = self.transform_series_to_mel(waveform)

        sample = (M, self.label_to_num(label))

        return sample
    
    def index_to_label(self, idx):
        return self.google_speech_frame.loc[idx, 1]

    def get_classes(self):
        return self.indices
    
    def get_labels(self):
        return self.labels
    
    def get_random_from_class(self, label):
        label_num = 0

        if type(label) == str:
            label_num = self.label_to_num(label)
        else:
            label_num = int(label)
        
        return np.random.choice(self.indices[label_num])
    
    def label_to_num(self, label):
        return torch.tensor(np.where(self.labels == label)[0][0], dtype=torch.int64)
    
    def num_to_label(self, num):
        return self.labels[num]
                