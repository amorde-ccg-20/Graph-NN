import torch
from torch import nn

from q3.models.forecast import ForecastModel


class AdaptiveForecastModel(nn.Module):
    def __init__(self, sensors=716, embedding_dim=16, hybrid=False):
        super().__init__()
        self.forecast = ForecastModel(hidden=16)
        self.embedding_1 = nn.Parameter(torch.randn(sensors, embedding_dim) * 0.1)
        self.embedding_2 = nn.Parameter(torch.randn(sensors, embedding_dim) * 0.1)
        self.mix_logit = nn.Parameter(torch.tensor(0.0)) if hybrid else None

    def adaptive_adjacency(self):
        scores = torch.relu(self.embedding_1 @ self.embedding_2.T)
        return torch.softmax(scores, dim=1)

    def forward(self, x, fixed_adjacency):
        adjacency = self.adaptive_adjacency()
        if self.mix_logit is not None:
            weight = torch.sigmoid(self.mix_logit)
            adjacency = weight * fixed_adjacency + (1 - weight) * adjacency
        return self.forecast(x, adjacency)
