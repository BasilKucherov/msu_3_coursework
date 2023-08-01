from torch.utils.data import DataLoader
import torch
import numpy as np

class TripletLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        self.dataset_norm = dataset
        
    def __iter__(self):
        batch=[]
        indices = self.dataset_norm.get_classes()
        labels = np.arange(len(indices))

        for i in range((len(self.dataset_norm) + self.batch_size - 1) // self.batch_size):
            label_pairs = np.zeros((self.batch_size, 2), dtype=np.int32)

            for i in range(self.batch_size):
                anchor_label, negative_label = np.random.choice(labels, size=2, replace=False)
                label_pairs[i] = [anchor_label, negative_label]
            
            for anchor_label, negative_label in label_pairs:
                anchor_idx, positive_idx = np.random.choice(indices[anchor_label], 2, replace=False)
                negative_idx = np.random.choice(indices[negative_label])

                anchor, anchor_label = self.dataset_norm.__getitem__(anchor_idx)
                positive, positive_label = self.dataset_norm.__getitem__(positive_idx)
                negative, negative_label = self.dataset_norm.__getitem__(negative_idx)

                triplet = torch.stack((anchor, positive, negative))
                triplet_labels = torch.stack((anchor_label, positive_label, negative_label))

                if len(batch) == 0:
                    batch.append(triplet)
                    batch.append(triplet_labels)
                else:
                    batch[0] = torch.vstack((batch[0], triplet))
                    batch[1] = torch.hstack((batch[1], triplet_labels))
            
            yield batch
            batch = []
