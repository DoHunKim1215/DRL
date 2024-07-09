import torch
from torch import nn, Tensor
import torch.nn.functional as F


class FCQ(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(32, 32), activation=F.relu):
        super(FCQ, self).__init__()

        self.activation = activation

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _format(self, x) -> Tensor:
        if not isinstance(x, Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._format(x)
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x

    def numpy_float_to_device(self, variable):
        return torch.from_numpy(variable).float().to(self.device)

    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = self.numpy_float_to_device(states)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = self.numpy_float_to_device(new_states)
        rewards = self.numpy_float_to_device(rewards)
        is_terminals = self.numpy_float_to_device(is_terminals)
        return states, actions, rewards, new_states, is_terminals
