from abc import abstractmethod

from torch import nn
import torch.nn.functional as F


class VNetwork(nn.Module):

    @abstractmethod
    def forward(self, state):
        pass


class FCV(VNetwork):
    def __init__(self,
                 input_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(FCV, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state):
        x = self.activation_fc(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)