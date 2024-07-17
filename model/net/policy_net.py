from abc import abstractmethod
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class StochasticPNetwork(nn.Module):

    @abstractmethod
    def select_action_informatively(self, state: Tensor, *args) -> Tuple:
        pass

    @abstractmethod
    def select_action(self, state: Tensor):
        pass

    @abstractmethod
    def select_greedy_action(self, state: Tensor):
        pass


class FCDAP(StochasticPNetwork):

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


class FCAC(StochasticPNetwork):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(FCAC, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.value_output_layer = nn.Linear(hidden_dims[-1], 1)
        self.policy_output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, state: Tensor):
        x = self.activation_fc(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.policy_output_layer(x), self.value_output_layer(x)

    def select_action_informatively(self, state: Tensor):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        action = action.item() if len(action) == 1 else action.data.numpy()
        is_exploratory = action != np.argmax(logits.detach().numpy(), axis=int(len(state) != 1))
        return action, is_exploratory, log_prob, entropy, value

    def select_action(self, state: Tensor):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action = action.item() if len(action) == 1 else action.data.numpy()
        return action

    def select_greedy_action(self, state: Tensor):
        logits, _ = self.forward(state)
        return np.argmax(logits.detach().numpy())

    def evaluate_state(self, state: Tensor) -> Tensor:
        _, value = self.forward(state)
        return value


class FCGP(StochasticPNetwork):

    def __init__(self,
                 input_dim,
                 action_bounds,
                 log_std_min=-20,
                 log_std_max=2,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu,
                 entropy_lr=0.001,
                 device=torch.device('cpu')):
        super(FCGP, self).__init__()

        self.activation_fc = activation_fc
        self.env_min, self.env_max = action_bounds

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(
                hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.output_layer_mean = nn.Linear(hidden_dims[-1], len(self.env_max))
        self.output_layer_log_std = nn.Linear(hidden_dims[-1], len(self.env_max))

        self.device = device
        self.env_min = torch.tensor(self.env_min, device=self.device, dtype=torch.float32)
        self.env_max = torch.tensor(self.env_max, device=self.device, dtype=torch.float32)

        self.nn_min = F.tanh(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = F.tanh(torch.Tensor([float('inf')])).to(self.device)

        self.target_entropy = -np.prod(self.env_max.shape)
        self.logalpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.logalpha], lr=entropy_lr)

    def rescale(self, x):
        return (x - self.nn_min) * (self.env_max - self.env_min) / (self.nn_max - self.nn_min) + self.env_min

    def forward(self, state):
        x = self.activation_fc(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x_mean = self.output_layer_mean(x)
        x_log_std = self.output_layer_log_std(x)
        x_log_std = torch.clamp(x_log_std, self.log_std_min, self.log_std_max)
        return x_mean, x_log_std

    def select_action_informatively(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)

        pi_s = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh_action = pi_s.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = self.rescale(tanh_action)

        log_prob = pi_s.log_prob(pre_tanh_action) - torch.log((1 - tanh_action.pow(2)).clamp(0, 1) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, self.rescale(torch.tanh(mean))

    def _update_exploration_ratio(self, greedy_action, action_taken):
        env_min, env_max = self.env_min.cpu().numpy(), self.env_max.cpu().numpy()
        self.exploration_ratio = np.mean(abs((greedy_action - action_taken) / (env_max - env_min)))

    def _get_actions(self, state):
        mean, log_std = self.forward(state)

        action = self.rescale(torch.tanh(torch.distributions.Normal(mean, log_std.exp()).sample()))
        greedy_action = self.rescale(torch.tanh(mean))
        random_action = np.random.uniform(low=self.env_min.cpu().numpy(),
                                          high=self.env_max.cpu().numpy())

        action_shape = self.env_max.cpu().numpy().shape
        action = action.detach().cpu().numpy().reshape(action_shape)
        greedy_action = greedy_action.detach().cpu().numpy().reshape(action_shape)
        random_action = random_action.reshape(action_shape)

        return action, greedy_action, random_action

    def select_random_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, random_action)
        return random_action

    def select_greedy_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, greedy_action)
        return greedy_action

    def select_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, action)
        return action


class DeterministicPNetwork(nn.Module):
    pass


class FCDP(DeterministicPNetwork):

    def __init__(self,
                 input_dim,
                 action_bounds,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu,
                 out_activation_fc=F.tanh,
                 device=torch.device('cpu')):
        super(FCDP, self).__init__()
        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc
        self.env_min, self.env_max = action_bounds
        self.device = device

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], len(self.env_max))

        self.env_min = torch.tensor(self.env_min, device=self.device, dtype=torch.float32)
        self.env_max = torch.tensor(self.env_max, device=self.device, dtype=torch.float32)
        self.nn_min = self.out_activation_fc(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = self.out_activation_fc(torch.Tensor([float('inf')])).to(self.device)

    def rescale(self, x):
        return (x - self.nn_min) * (self.env_max - self.env_min) / (self.nn_max - self.nn_min) + self.env_min

    def forward(self, state):
        x = self.activation_fc(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        x = self.out_activation_fc(x)
        return self.rescale(x)


class FCCA(StochasticPNetwork):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(FCCA, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, states):
        x = self.activation_fc(self.input_layer(states))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)

    def np_pass(self, states):
        logits = self.forward(states)
        np_logits = logits.detach().cpu().numpy()
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        np_actions = actions.detach().cpu().numpy()
        logpas = dist.log_prob(actions)
        np_logpas = logpas.detach().cpu().numpy()
        is_exploratory = np_actions != np.argmax(np_logits, axis=1)
        return np_actions, np_logpas, is_exploratory

    def select_action(self, states):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.detach().cpu().item()

    def get_predictions(self, states, actions):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        logpas = dist.log_prob(actions)
        entropies = dist.entropy()
        return logpas, entropies

    def select_greedy_action(self, states):
        logits = self.forward(states)
        return np.argmax(logits.detach().squeeze().cpu().numpy())
