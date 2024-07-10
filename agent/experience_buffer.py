from abc import ABC, abstractmethod

import numpy as np

from agent.experience import Experience


class ExperienceBuffer(ABC):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def store(self, sample):
        pass

    @abstractmethod
    def sample(self, batch_size=None) -> Experience:
        pass

    @abstractmethod
    def update(self, idxs, priorities):
        pass

    @abstractmethod
    def can_optimize(self):
        pass

    @abstractmethod
    def after_opti_process(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class ExhaustingBuffer(ExperienceBuffer):
    def __init__(self, batch_size: int = 1024):
        super().__init__(batch_size)
        self.experiences = []

    def clear(self):
        self.experiences.clear()

    def store(self, sample):
        self.experiences.append(sample)

    def sample(self, batch_size=None):
        experiences = np.array(self.experiences, dtype=object)
        batches = [np.vstack(sars) for sars in experiences.T]
        return Experience(np.vstack([0 for _ in range(batch_size)]),
                          np.vstack([1.0 for _ in range(batch_size)]),
                          batches[0],
                          batches[1],
                          batches[2],
                          batches[3],
                          batches[4])

    def update(self, idxs, priorities):
        pass

    def can_optimize(self) -> bool:
        return len(self) >= self.batch_size

    def after_opti_process(self):
        self.clear()

    def __len__(self):
        return len(self.experiences)


class ReplayBuffer(ExperienceBuffer):
    def __init__(self,
                 max_size: int = 10000,
                 batch_size: int = 64,
                 n_warmup_batches: int = 5):
        super().__init__(batch_size)
        self.state_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.action_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.reward_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.next_state_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.terminated_mem = np.empty(shape=max_size, dtype=np.ndarray)

        self.max_size = max_size
        self.n_warmup_batches = n_warmup_batches
        self._idx = 0
        self.size = 0

    def clear(self):
        self.state_mem = np.empty(shape=self.max_size, dtype=np.ndarray)
        self.action_mem = np.empty(shape=self.max_size, dtype=np.ndarray)
        self.reward_mem = np.empty(shape=self.max_size, dtype=np.ndarray)
        self.next_state_mem = np.empty(shape=self.max_size, dtype=np.ndarray)
        self.terminated_mem = np.empty(shape=self.max_size, dtype=np.ndarray)

        self._idx = 0
        self.size = 0

    def store(self, sample):
        state, action, reward, next_state, terminated = sample
        self.state_mem[self._idx] = state
        self.action_mem[self._idx] = action
        self.reward_mem[self._idx] = reward
        self.next_state_mem[self._idx] = next_state
        self.terminated_mem[self._idx] = terminated

        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idxs = np.random.choice(self.size, batch_size, replace=False)
        return Experience(np.vstack([0 for _ in range(batch_size)]),
                          np.vstack([1.0 for _ in range(batch_size)]),
                          np.vstack(self.state_mem[idxs]),
                          np.vstack(self.action_mem[idxs]),
                          np.vstack(self.reward_mem[idxs]),
                          np.vstack(self.next_state_mem[idxs]),
                          np.vstack(self.terminated_mem[idxs]))

    def update(self, idxs, priorities):
        pass

    def can_optimize(self) -> bool:
        return len(self) >= self.batch_size * self.n_warmup_batches

    def after_opti_process(self):
        pass

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer(ExperienceBuffer):

    def __init__(self,
                 max_samples=10000,
                 batch_size=64,
                 rank_based=False,
                 alpha=0.6,
                 beta0=0.1,
                 beta_rate=0.99992,
                 n_warmup_batches: int = 5):
        super().__init__(batch_size)
        self.max_samples = max_samples
        self.memory = np.empty(shape=(self.max_samples, 2), dtype=np.ndarray)
        self.n_entries = 0
        self.next_index = 0
        self.td_error_index = 0
        self.sample_index = 1
        self.rank_based = rank_based
        self.alpha = alpha
        self.beta = beta0
        self.beta0 = beta0
        self.beta_rate = beta_rate
        self.n_warmup_batches = n_warmup_batches

    def clear(self):
        self.memory = np.empty(shape=(self.max_samples, 2), dtype=np.ndarray)
        self.n_entries = 0
        self.next_index = 0
        self.td_error_index = 0
        self.sample_index = 1
        self.beta = self.beta0

    def update(self, idxs, td_errors):
        self.memory[idxs, self.td_error_index] = np.abs(td_errors)
        if self.rank_based:
            sorted_arg = self.memory[:self.n_entries, self.td_error_index].argsort()[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]

    def store(self, sample):
        priority = 1.0
        if self.n_entries > 0:
            priority = self.memory[:self.n_entries, self.td_error_index].max()
        self.memory[self.next_index, self.td_error_index] = priority
        self.memory[self.next_index, self.sample_index] = np.array(sample, dtype=object)
        self.n_entries = min(self.n_entries + 1, self.max_samples)
        self.next_index += 1
        self.next_index = self.next_index % self.max_samples

    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate ** -1)
        return self.beta

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        self._update_beta()
        entries = self.memory[:self.n_entries]

        if self.rank_based:
            priorities = 1 / (np.arange(self.n_entries) + 1)
        else:  # proportional
            priorities = entries[:, self.td_error_index] + 1e-6
        scaled_priorities = priorities ** self.alpha
        probs = np.array(scaled_priorities / np.sum(scaled_priorities), dtype=np.float64)

        weights = (self.n_entries * probs) ** -self.beta
        normalized_weights = weights / weights.max()
        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)
        samples = np.array([entries[idx] for idx in idxs])

        samples_stacks = [np.vstack(batch_type) for batch_type in np.vstack(samples[:, self.sample_index]).T]
        idxs_stack = np.vstack(idxs)
        weights_stack = np.vstack(normalized_weights[idxs])
        return Experience(idxs_stack,
                          weights_stack,
                          samples_stacks[0],
                          samples_stacks[1],
                          samples_stacks[2],
                          samples_stacks[3],
                          samples_stacks[4])

    def can_optimize(self) -> bool:
        return len(self) >= self.batch_size * self.n_warmup_batches

    def after_opti_process(self):
        pass

    def __len__(self):
        return self.n_entries
