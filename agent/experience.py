import torch


class Experience:
    def __init__(self, idxs, weights, states, actions, rewards, next_states, is_terminals):
        self.idxs = idxs
        self.weights = weights
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.is_terminals = is_terminals

    def load(self, device):
        self.weights = torch.from_numpy(self.weights).float().to(device)
        self.states = torch.from_numpy(self.states).float().to(device)
        self.actions = torch.from_numpy(self.actions).long().to(device)
        self.next_states = torch.from_numpy(self.next_states).float().to(device)
        self.rewards = torch.from_numpy(self.rewards).float().to(device)
        self.is_terminals = torch.from_numpy(self.is_terminals).float().to(device)

    def decompose(self):
        return self.idxs, self.weights, (self.states, self.actions, self.rewards, self.next_states, self.is_terminals)
