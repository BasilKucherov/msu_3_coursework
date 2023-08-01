import torch.nn as nn
import torch


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = nn.functional.pairwise_distance(anchor, positive, 2)
        distance_negative = nn.functional.pairwise_distance(anchor, negative, 2)
        losses = nn.functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
