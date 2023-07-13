import torch.nn as nn
import torch


class NPairLoss(nn.Module):
    def __init__(self):
        super(NPairLoss, self).__init__()

    def forward(self, anchors, positives, negatives):     
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss
    


        # distance_positive = torch.exp(torch.pairwise_distance(anchor, positive, 2))
        # distance_negative = torch.exp(torch.pairwise_distance(anchor.unsqueeze(1).repeat(1, negative.shape[1], 1), negative, 2)).sum(dim=1)
        
        # loss = -torch.log(distance_positive / (distance_negative + distance_negative))

        # return loss.mean()
