import numpy as np
from torch import nn
from abc import ABCMeta, abstractmethod


class ExplorationStrategy(metaclass=ABCMeta):
    def __init__(self, exploratory_action_taken):
        self.exploratory_action_taken = exploratory_action_taken

    @abstractmethod
    def select_action(self, model: nn.Module, state: np.ndarray):
        pass

    @staticmethod
    def _forward(model: nn.Module, state) -> np.ndarray:
        return model(state).detach().cpu().numpy().squeeze()


class GreedyStrategy(ExplorationStrategy):
    """
    Greedy Strategy
    Always chooses action with the highest value. (Always exploit)
    """
    def __init__(self):
        super().__init__(False)

    def select_action(self, model: nn.Module, state):
        q_values = self._forward(model, state)
        return np.argmax(q_values)


class EGreedyStrategy(ExplorationStrategy):
    """
    Epsilon Greedy Strategy
    With epsilon probability, chooses action uniformly at random.
    With 1 - epsilon probability, chooses action with the highest value.
    Always balance between exploration and exploitation with constant exploration rate.
    """
    def __init__(self, epsilon: float = 0.1):
        super().__init__(None)
        self.epsilon = epsilon

    def select_action(self, model: nn.Module, state):
        q_values = self._forward(model, state)
        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action


class LinearlyDecayingEGreedyStrategy(ExplorationStrategy):
    """
    Linearly Decaying Epsilon Greedy Strategy
    With epsilon probability, chooses action uniformly at random.
    With 1 - epsilon probability, chooses action with the highest value.
    Always balance between exploration and exploitation with linearly decaying exploration rate.
    """
    def __init__(self, init_epsilon: float = 1.0, min_epsilon: float = 0.1, decay_steps: int = 20000):
        super().__init__(None)
        self.t: int = 0
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps

    def _epsilon_update(self):
        epsilon = (self.min_epsilon - self.init_epsilon) / self.decay_steps * self.t + self.init_epsilon
        epsilon = np.clip(epsilon, self.min_epsilon, self.init_epsilon)
        self.t += 1
        return epsilon

    def select_action(self, model: nn.Module, state):
        q_values = self._forward(model, state)
        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))
        self.epsilon = self._epsilon_update()
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action


class ExponentiallyDecayingEGreedyStrategy(ExplorationStrategy):
    """
    Exponentially Decaying Epsilon Greedy Strategy
    With epsilon probability, chooses action uniformly at random.
    With 1 - epsilon probability, chooses action with the highest value.
    Always balance between exploration and exploitation with exponentially decaying exploration rate.
    """
    def __init__(self, init_epsilon: float = 1.0, min_epsilon: float = 0.1, decay_steps: int = 20000):
        super().__init__(None)
        self.t: int = 0
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps
        self.epsilons = ((0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01) *
                         (init_epsilon - min_epsilon) + min_epsilon)

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, model: nn.Module, state):
        q_values = self._forward(model, state)
        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))
        self.epsilon = self._epsilon_update()
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action


class SoftmaxStrategy(ExplorationStrategy):
    """
    Softmax Strategy
    Chooses action from softmax distribution over Q / temperature with linearly decaying temperature.
    """
    def __init__(self,
                 init_temp: float = 1.0,
                 min_temp: float = 0.3,
                 decay_steps: int = 20000):
        super().__init__(None)
        self.t: int = 0
        self.init_temp = init_temp
        self.decay_steps = decay_steps
        self.min_temp = min_temp

    def _update_temp(self):
        temp = 1 - self.t / self.decay_steps
        temp = (self.init_temp - self.min_temp) * temp + self.min_temp
        temp = np.clip(temp, self.min_temp, self.init_temp)
        self.t += 1
        return temp

    def select_action(self, model: nn.Module, state):
        temp = self._update_temp()

        q_values = self._forward(model, state)
        scaled_qs = q_values / temp
        norm_qs = scaled_qs - scaled_qs.max()
        e = np.exp(norm_qs)
        probs = e / np.sum(e)
        assert np.isclose(probs.sum(), 1.0)

        action = np.random.choice(np.arange(len(probs)), size=1, p=probs)[0]
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action


class BoundedGreedyStrategy(ExplorationStrategy):

    def __init__(self, bounds):
        super().__init__(False)

        self.low, self.high = bounds
        self.ratio_noise_injected = 0

    def select_action(self, model: nn.Module, state):
        greedy_action = self._forward(model, state)
        greedy_action = np.clip(greedy_action, self.low, self.high)
        return np.reshape(greedy_action, self.low.shape)


class GaussianNoiseStrategy(ExplorationStrategy):

    def __init__(self, bounds, exploration_noise_ratio: float = 0.1):
        super().__init__(None)

        self.low, self.high = bounds
        self.exploration_noise_ratio = exploration_noise_ratio
        self.ratio_noise_injected = 0

    def select_action(self, model: nn.Module, state, max_exploration: bool = False) -> np.ndarray:
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.exploration_noise_ratio * self.high

        greedy_action = self._forward(model, state)
        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)

        self.ratio_noise_injected = np.mean(abs((greedy_action - action) / (self.high - self.low)))
        return action
