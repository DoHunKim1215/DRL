import torch
from torch import nn, Tensor
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()


class FCQ(QNetwork):

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


class FCDuelingQ(QNetwork):

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


class FCQV(QNetwork):

    def __init__(self,
                 input_state_dim,
                 input_action_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(FCQV, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            if i == 0:
                in_dim += input_action_dim
            hidden_layer = nn.Linear(in_dim, hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state, action):
        x = self.activation_fc(self.input_layer(state))
        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat((x, action), dim=1)
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)
