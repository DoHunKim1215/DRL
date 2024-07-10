import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Q(nn.Module):

    def __init__(self):
        super(Q, self).__init__()


class FCQ(Q):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(32, 32), activation=F.relu):
        super(FCQ, self).__init__()

        self.activation = activation

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x


class FCDuelingQ(Q):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(32, 32), activation=F.relu):
        super(FCDuelingQ, self).__init__()

        self.activation = activation

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.value_output_layer = nn.Linear(hidden_dims[-1], 1)
        self.adv_output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        a = self.adv_output_layer(x)
        v = self.value_output_layer(x).expand_as(a)
        q = v + a - a.mean(1, keepdim=True).expand_as(a)
        return q
