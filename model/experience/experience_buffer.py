import gymnasium
from torch import nn

import gc
import time
from abc import ABC, abstractmethod

import numpy as np
import torch

from model.experience.experience import Experience


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


class EpisodeBuffer:

    def __init__(self,
                 state_dim,
                 gamma,
                 tau,
                 n_workers,
                 max_episodes,
                 max_episode_steps,
                 device):

        assert max_episodes >= n_workers

        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.n_workers = n_workers
        self.max_episodes = max_episodes
        self.max_episode_steps = max_episode_steps

        self.discounts = np.logspace(0, max_episode_steps + 1, num=max_episode_steps + 1,
                                     base=gamma, endpoint=False, dtype=np.longdouble)
        self.tau_discounts = np.logspace(0, max_episode_steps + 1, num=max_episode_steps + 1,
                                         base=gamma * tau, endpoint=False, dtype=np.longdouble)

        self.device = device

        self.states_mem = None
        self.actions_mem = None
        self.returns_mem = None
        self.gaes_mem = None
        self.log_probs_mem = None

        self.episode_steps = None
        self.episode_reward = None
        self.episode_exploration = None
        self.episode_seconds = None

        self.current_ep_idxs = None

        self.clear()

    def clear(self):
        self.states_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps, self.state_dim), dtype=np.float64)
        self.states_mem[:] = np.nan

        self.actions_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.uint8)
        self.actions_mem[:] = 0

        self.returns_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.returns_mem[:] = np.nan

        self.gaes_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.gaes_mem[:] = np.nan

        self.log_probs_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.log_probs_mem[:] = np.nan

        self.episode_steps = np.zeros(shape=self.max_episodes, dtype=np.uint16)
        self.episode_reward = np.zeros(shape=self.max_episodes, dtype=np.float32)
        self.episode_exploration = np.zeros(shape=self.max_episodes, dtype=np.float32)
        self.episode_seconds = np.zeros(shape=self.max_episodes, dtype=np.float64)

        self.current_ep_idxs = np.arange(self.n_workers, dtype=np.uint16)
        gc.collect()

    def fill(self, envs, policy_model, value_model):
        states = envs.reset()

        worker_rewards = np.zeros(shape=(self.n_workers, self.max_episode_steps), dtype=np.float32)
        worker_exploratory = np.zeros(shape=(self.n_workers, self.max_episode_steps), dtype=bool)
        worker_steps = np.zeros(shape=self.n_workers, dtype=np.uint16)
        worker_seconds = np.array([time.time(), ] * self.n_workers, dtype=np.float64)

        buffer_full = False
        while not buffer_full and len(self.episode_steps[self.episode_steps > 0]) < self.max_episodes / 2:
            with torch.no_grad():
                x = torch.tensor(states, device=self.device, dtype=torch.float32)
                if len(x.size()) == 1:
                    x = x.unsqueeze(0)
                actions, log_probs, are_exploratory = policy_model.np_pass(x)

            next_states, rewards, terminals, truncated = envs.step(actions)
            self.states_mem[self.current_ep_idxs, worker_steps] = states
            self.actions_mem[self.current_ep_idxs, worker_steps] = actions
            self.log_probs_mem[self.current_ep_idxs, worker_steps] = log_probs

            worker_exploratory[np.arange(self.n_workers), worker_steps] = are_exploratory
            worker_rewards[np.arange(self.n_workers), worker_steps] = rewards.reshape(-1,)

            for w_idx in range(self.n_workers):
                if worker_steps[w_idx] + 1 == self.max_episode_steps:
                    terminals[w_idx] = 1
                    truncated[w_idx] = True

            states = next_states
            worker_steps += 1

            if terminals.sum():
                idx_terminals = np.flatnonzero(terminals)
                next_values = np.zeros(shape=self.n_workers)

                if truncated.sum():
                    idx_truncated = np.flatnonzero(truncated)
                    with torch.no_grad():
                        x = torch.tensor(next_states[idx_truncated], device=self.device, dtype=torch.float32)
                        if len(x.size()) == 1:
                            x = x.unsqueeze(0)
                        next_values[idx_truncated] = value_model(x).cpu().numpy()

                new_states = np.stack([envs.reset(rank=idx_terminal) for idx_terminal in idx_terminals])
                states[idx_terminals] = new_states

                for w_idx in range(self.n_workers):
                    if w_idx not in idx_terminals:
                        continue

                    e_idx = self.current_ep_idxs[w_idx]
                    T = worker_steps[w_idx]
                    self.episode_steps[e_idx] = T
                    self.episode_reward[e_idx] = worker_rewards[w_idx, :T].sum()
                    self.episode_exploration[e_idx] = worker_exploratory[w_idx, :T].mean()
                    self.episode_seconds[e_idx] = time.time() - worker_seconds[w_idx]

                    ep_rewards = np.concatenate((worker_rewards[w_idx, :T], [next_values[w_idx]]))
                    ep_discounts = self.discounts[:T + 1]
                    ep_returns = np.array([np.sum(ep_discounts[:T + 1 - t] * ep_rewards[t:]) for t in range(T)])
                    self.returns_mem[e_idx, :T] = ep_returns

                    ep_states = self.states_mem[e_idx, :T]
                    with torch.no_grad():
                        x = torch.tensor(ep_states, device=self.device, dtype=torch.float32)
                        if len(x.size()) == 1:
                            x = x.unsqueeze(0)
                        ep_values = torch.cat((value_model(x), torch.tensor([next_values[w_idx]],
                                                                            device=self.device,
                                                                            dtype=torch.float32)))
                    np_ep_values = ep_values.view(-1).cpu().numpy()
                    deltas = ep_rewards[:-1] + self.gamma * np_ep_values[1:] - np_ep_values[:-1]
                    gaes = np.array([np.sum(self.tau_discounts[:T - t] * deltas[t:]) for t in range(T)])
                    self.gaes_mem[e_idx, :T] = gaes

                    worker_exploratory[w_idx, :] = 0
                    worker_rewards[w_idx, :] = 0
                    worker_steps[w_idx] = 0
                    worker_seconds[w_idx] = time.time()

                    new_ep_id = max(self.current_ep_idxs) + 1
                    if new_ep_id >= self.max_episodes:
                        buffer_full = True
                        break

                    self.current_ep_idxs[w_idx] = new_ep_id

        ep_idxs = self.episode_steps > 0
        ep_t = self.episode_steps[ep_idxs]

        self.states_mem = [row[:ep_t[i]] for i, row in enumerate(self.states_mem[ep_idxs])]
        self.states_mem = np.concatenate(self.states_mem)
        self.actions_mem = [row[:ep_t[i]] for i, row in enumerate(self.actions_mem[ep_idxs])]
        self.actions_mem = np.concatenate(self.actions_mem)
        self.returns_mem = [row[:ep_t[i]] for i, row in enumerate(self.returns_mem[ep_idxs])]
        self.returns_mem = torch.tensor(np.concatenate(self.returns_mem), device=self.device)
        self.gaes_mem = [row[:ep_t[i]] for i, row in enumerate(self.gaes_mem[ep_idxs])]
        self.gaes_mem = torch.tensor(np.concatenate(self.gaes_mem), device=self.device)
        self.log_probs_mem = [row[:ep_t[i]] for i, row in enumerate(self.log_probs_mem[ep_idxs])]
        self.log_probs_mem = torch.tensor(np.concatenate(self.log_probs_mem), device=self.device)

        ep_r = self.episode_reward[ep_idxs]
        ep_x = self.episode_exploration[ep_idxs]
        ep_s = self.episode_seconds[ep_idxs]
        return ep_t, ep_r, ep_x, ep_s

    def get_stacks(self):
        return self.states_mem, self.actions_mem, self.returns_mem, self.gaes_mem, self.log_probs_mem

    def __len__(self):
        return self.episode_steps[self.episode_steps > 0].sum()\
