from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class FCDAP(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(FCDAP, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, state: Tensor) -> Tensor:
        x = self.activation_fc(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)

    def select_action_informatively(self, state: Tensor) -> Tuple:
        action_logit_probs = self.forward(state)
        dist = torch.distributions.Categorical(logits=action_logit_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        is_exploratory = action != np.argmax(action_logit_probs.detach().cpu().numpy())
        return action.item(), is_exploratory.item(), log_prob, entropy

    def select_action(self, state: Tensor):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def select_greedy_action(self, state: Tensor):
        logits = self.forward(state)
        return np.argmax(logits.detach().cpu().numpy())
