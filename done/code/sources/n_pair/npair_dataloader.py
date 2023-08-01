from torch.utils.data import Sampler
import torch
import numpy as np

class ClassPairSampler(Sampler):
    def __init__(self, dataset, indexes):
        self.dataset = dataset
        self.class_indices = {}
        for i in indexes:
            label = int(dataset[i][1])
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)
        self.class_iterators = {label: iter(indices) for label, indices in self.class_indices.items()}
        self.num_classes = len(self.class_indices)
        
    def __iter__(self):
        for i in range(len(self.dataset) // (2*self.num_classes)):
            batch = []
            for label, iterator in self.class_iterators.items():
                try:
                    index = next(iterator)
                except StopIteration:
                    self.class_iterators[label] = iter(self.class_indices[label])
                    index = next(self.class_iterators[label])
                batch.append(index)
                try:
                    index = next(iterator)
                except StopIteration:
                    self.class_iterators[label] = iter(self.class_indices[label])
                    index = next(self.class_iterators[label])
                batch.append(index)
            yield batch
            
    def __len__(self):
        return len(self.dataset) // (2*self.num_classes)
