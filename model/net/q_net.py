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


class FCTQV(QNetwork):

    def __init__(self,
                 input_state_dim,
                 input_action_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(FCTQV, self).__init__()

        self.activation_fc = activation_fc

        self.input_layer_a = nn.Linear(input_state_dim + input_action_dim, hidden_dims[0])
        self.input_layer_b = nn.Linear(input_state_dim + input_action_dim, hidden_dims[0])

        self.hidden_layers_a = nn.ModuleList()
        self.hidden_layers_b = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer_a = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers_a.append(hidden_layer_a)

            hidden_layer_b = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers_b.append(hidden_layer_b)

        self.output_layer_a = nn.Linear(hidden_dims[-1], 1)
        self.output_layer_b = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        xa = self.activation_fc(self.input_layer_a(x))
        xb = self.activation_fc(self.input_layer_b(x))
        for hidden_layer_a, hidden_layer_b in zip(self.hidden_layers_a, self.hidden_layers_b):
            xa = self.activation_fc(hidden_layer_a(xa))
            xb = self.activation_fc(hidden_layer_b(xb))
        xa = self.output_layer_a(xa)
        xb = self.output_layer_b(xb)
        return xa, xb

    def Qa(self, state, action):
        x = torch.cat((state, action), dim=1)
        xa = self.activation_fc(self.input_layer_a(x))
        for hidden_layer_a in self.hidden_layers_a:
            xa = self.activation_fc(hidden_layer_a(xa))
        return self.output_layer_a(xa)


class FCQSA(nn.Module):

    def __init__(self,
                 input_state_dim,
                 input_action_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(FCQSA, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_state_dim + input_action_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x

