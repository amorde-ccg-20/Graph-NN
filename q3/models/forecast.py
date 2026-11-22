import torch
from torch import nn


class ForecastModel(nn.Module):
    def __init__(self, hidden=16, use_graph=True):
        super().__init__()
        self.temporal = nn.Conv1d(1, hidden, kernel_size=8)
        self.graph = nn.Linear(hidden, hidden) if use_graph else None
        self.head = nn.Linear(hidden, 4)

    def forward(self, x, adjacency=None):
        batch, history, sensors = x.shape
        z = x.permute(0, 2, 1).reshape(batch * sensors, 1, history)
        z = torch.relu(self.temporal(z)[:, :, -1])
        z = z.reshape(batch, sensors, -1)
        if self.graph is not None:
            z = torch.relu(self.graph(torch.matmul(adjacency, z)))
        return self.head(z).permute(0, 2, 1)
