"""Google speech commands dataset."""
__author__ = 'Yuan Xu'

import os
import numpy as np

import librosa

from torch.utils.data import Dataset

class SpeechCommandsDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        self.all_classes.sort()

        # print(self.all_classes)

        self.class_to_idx = {self.all_classes[i]: i for i in range(len(self.all_classes))}
        self.idx_to_class = {idx: c for c, idx in self.class_to_idx.items()} 
        
        self.indices = [[] for _ in range(len(self.all_classes))]
 
        data = []
        cur_idx = 0

        for c in self.all_classes:
            d = os.path.join(folder, c)
            target = self.class_to_idx[c]
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append((path, target))

                self.indices[target].append(cur_idx)
                cur_idx += 1

        self.classes = self.all_classes
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_class_idx(self, idx):
        if idx in self.idx_to_class.keys():
            return self.idx_to_class[idx]
        return 'unknown'
    
    def get_idx_class(self, c):
        if c in self.class_to_idx.keys():
            return self.class_to_idx[c]
        return -1

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight

class BackgroundNoiseDataset(Dataset):
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1):
        audio_files = [d for d in os.listdir(folder) if os.path.isfile(os.path.join(folder, d)) and d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sr=sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': 1, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data
