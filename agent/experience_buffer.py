from abc import ABC, abstractmethod

import numpy as np


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
    def sample(self, batch_size=None):
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
        return batches

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
        experiences = np.vstack(self.state_mem[idxs]), \
            np.vstack(self.action_mem[idxs]), \
            np.vstack(self.reward_mem[idxs]), \
            np.vstack(self.next_state_mem[idxs]), \
            np.vstack(self.terminated_mem[idxs])
        return experiences

    def can_optimize(self) -> bool:
        return len(self) >= self.batch_size * self.n_warmup_batches

    def after_opti_process(self):
        pass

    def __len__(self):
        return self.size
