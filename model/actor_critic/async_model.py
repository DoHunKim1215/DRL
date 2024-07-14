import argparse
import gc
import random
import time
from itertools import count
from typing import Callable, List

import gymnasium
import torch.multiprocessing as mp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from model.model import RLModel
from model.policy_arch import PNetwork
from model.value_arch import VNetwork


class AsyncActorCriticModel(RLModel):

    def __init__(self,
                 policy_model_fn: Callable[[int, int], PNetwork],
                 policy_optimizer_fn: Callable[[nn.Module, float], Optimizer],
                 policy_optimizer_lr: float,
                 entropy_loss_weight: float,
                 args: argparse.Namespace,
                 value_model_fn: Callable[[int], VNetwork],
                 value_optimizer_fn: Callable[[nn.Module, float], Optimizer],
                 value_optimizer_lr: float,
                 max_n_steps: int,
                 n_workers: int,
                 tau: float):
        super(AsyncActorCriticModel, self).__init__(args)

        self.policy_model_fn = policy_model_fn
        self.policy_model_max_grad_norm = args.policy_model_max_gradient_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr

        self.value_model_fn = value_model_fn
        self.value_model_max_grad_norm = args.value_model_max_gradient_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr

        self.entropy_loss_weight = entropy_loss_weight
        self.max_n_steps = max_n_steps
        self.n_workers = n_workers
        self.tau = tau

        self.max_minutes = 0
        self.max_episodes = 0
        self.goal_mean_100_reward = 0

        self.stats = {}
        self.get_out_lock = None
        self.get_out_signal = None
        self.reached_max_minutes = None
        self.reached_max_episodes = None
        self.reached_goal_mean_reward = None
        self.training_start = None

        self.shared_policy_model = None
        self.shared_policy_optimizer = None
        self.shared_value_model = None
        self.shared_value_optimizer = None

    def optimize_model(self,
                       log_probs: List,
                       entropies: List,
                       rewards: List,
                       values: List,
                       local_policy_model: PNetwork,
                       local_value_model: VNetwork):
        T = len(rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T - t] * rewards[t:]) for t in range(T)])

        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        values = torch.cat(values)

        np_values = values.view(-1).data.numpy()
        tau_discounts = np.logspace(0, T - 1, num=T - 1, base=self.gamma * self.tau, endpoint=False)
        advs = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]
        gaes = np.array([np.sum(tau_discounts[:T - 1 - t] * advs[t:]) for t in range(T - 1)])

        values = values[:-1, ...]
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)
        gaes = torch.FloatTensor(gaes).unsqueeze(1)

        policy_loss = -(discounts * gaes.detach() * log_probs).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss
        self.shared_policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_policy_model.parameters(), self.policy_model_max_grad_norm)

        for param, shared_param in zip(local_policy_model.parameters(), self.shared_policy_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_policy_optimizer.step()
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())

        value_error = returns - values
        value_loss = value_error.pow(2).mul(0.5).mean()
        self.shared_value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_value_model.parameters(), self.value_model_max_grad_norm)

        for param, shared_param in zip(local_value_model.parameters(), self.shared_value_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        self.shared_value_optimizer.step()
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

    @staticmethod
    def interaction_step(state: np.ndarray,
                         env: gymnasium.Env,
                         local_policy_model: PNetwork,
                         local_value_model: VNetwork,
                         log_probs: List,
                         entropies: List,
                         rewards: List,
                         values: List):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action, is_exploratory, log_prob, entropy = local_policy_model.select_action_informatively(state)
        new_state, reward, terminated, truncated, _ = env.step(action)

        log_probs.append(log_prob)
        entropies.append(entropy)
        rewards.append(reward)
        values.append(local_value_model(state) * (1 - float(terminated and not truncated)))

        return new_state, reward, terminated or truncated, truncated, is_exploratory

    def work(self,
             rank: int,
             env: gymnasium.Env,
             local_policy_model: PNetwork,
             local_value_model: VNetwork):
        last_debug_time = float('-inf')
        self.stats['n_active_workers'].add_(1)

        local_seed = self.seed + rank
        env.reset(seed=local_seed)

        torch.manual_seed(local_seed)
        np.random.seed(local_seed)
        random.seed(local_seed)

        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

        debug_message = ''

        global_episode_idx = self.stats['episode'].add_(1).item() - 1
        while not self.get_out_signal:
            episode_start = time.time()
            state, _ = env.reset()

            n_steps_start = 0
            total_episode_rewards = 0
            total_episode_steps = 0
            total_episode_exploration = 0
            log_probs = []
            entropies = []
            rewards = []
            values = []

            for step in count(start=1):
                state, reward, is_terminal, is_truncated, is_exploratory \
                    = self.interaction_step(state,
                                            env,
                                            local_policy_model,
                                            local_value_model,
                                            log_probs,
                                            entropies,
                                            rewards,
                                            values)

                total_episode_steps += 1
                total_episode_rewards += float(reward) * pow(self.gamma, step)
                total_episode_exploration += int(is_exploratory)

                if is_terminal or step - n_steps_start == self.max_n_steps:
                    next_value = 0
                    if not is_terminal or is_truncated:
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        next_value = local_value_model(state_tensor).detach().item()
                    rewards.append(next_value)
                    values.append(torch.FloatTensor([[next_value, ], ]))

                    self.optimize_model(log_probs, entropies, rewards, values, local_policy_model, local_value_model)
                    log_probs = []
                    entropies = []
                    rewards = []
                    values = []
                    n_steps_start = step

                if is_terminal:
                    gc.collect()
                    break

            # save global stats
            episode_elapsed = time.time() - episode_start
            evaluation_score, _ = self.evaluate(local_policy_model, env)
            self.save_checkpoint(global_episode_idx, local_policy_model)

            self.stats['episode_elapsed'][global_episode_idx].add_(episode_elapsed)
            self.stats['episode_timestep'][global_episode_idx].add_(total_episode_steps)
            self.stats['episode_reward'][global_episode_idx].add_(total_episode_rewards)
            self.stats['episode_exploration'][global_episode_idx].add_(total_episode_exploration / total_episode_steps)
            self.stats['evaluation_scores'][global_episode_idx].add_(evaluation_score)

            mean_10_reward = self.stats['episode_reward'][:global_episode_idx + 1][-10:].mean().item()
            mean_100_reward = self.stats['episode_reward'][:global_episode_idx + 1][-100:].mean().item()
            mean_100_eval_score = self.stats['evaluation_scores'][:global_episode_idx + 1][-100:].mean().item()
            mean_100_exp_rat = self.stats['episode_exploration'][:global_episode_idx + 1][-100:].mean().item()

            std_10_reward = self.stats['episode_reward'][:global_episode_idx + 1][-10:].std().item() \
                if len(self.stats['episode_reward'][:global_episode_idx + 1][-10:]) > 1 else 0
            std_100_reward = self.stats['episode_reward'][:global_episode_idx + 1][-100:].std().item() \
                if len(self.stats['episode_reward'][:global_episode_idx + 1][-100:]) > 1 else 0
            std_100_eval_score = self.stats['evaluation_scores'][:global_episode_idx + 1][-100:].std().item() \
                if len(self.stats['evaluation_scores'][:global_episode_idx + 1][-100:]) > 1 else 0
            std_100_exp_rat = self.stats['episode_exploration'][:global_episode_idx + 1][-100:].std().item() \
                if len(self.stats['episode_exploration'][:global_episode_idx + 1][-100:]) > 1 else 0

            if std_10_reward != std_10_reward:
                std_10_reward = 0
            if std_100_reward != std_100_reward:
                std_100_reward = 0
            if std_100_eval_score != std_100_eval_score:
                std_100_eval_score = 0
            if std_100_exp_rat != std_100_exp_rat:
                std_100_exp_rat = 0

            global_n_steps = self.stats['episode_timestep'][:global_episode_idx + 1].sum().item()
            global_training_elapsed = self.stats['episode_elapsed'][:global_episode_idx + 1].sum().item()
            wallclock_elapsed = time.time() - self.training_start

            self.stats['result'][global_episode_idx][0].add_(global_n_steps)
            self.stats['result'][global_episode_idx][1].add_(mean_100_reward)
            self.stats['result'][global_episode_idx][2].add_(mean_100_eval_score)
            self.stats['result'][global_episode_idx][3].add_(global_training_elapsed)
            self.stats['result'][global_episode_idx][4].add_(wallclock_elapsed)

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.training_start))
            debug_message = '[{}] episode {:04}, cumulated step {:06}, '
            debug_message += 'return (MA10) {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'return (MA100) {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'exploration rate (MA100) {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'evaluation score (MA100) {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, global_episode_idx, global_n_steps, mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)

            if rank == 0:
                print(debug_message, end='\r', flush=True)
                if time.time() - last_debug_time >= self.leave_print_every_n_secs:
                    print(self.ERASE_LINE + debug_message, flush=True)
                    last_debug_time = time.time()

            with self.get_out_lock:
                potential_next_global_episode_idx = self.stats['episode'].item()
                self.reached_goal_mean_reward.add_(mean_100_eval_score >= self.goal_mean_100_reward)
                self.reached_max_minutes.add_(time.time() - self.training_start >= self.max_minutes * 60)
                self.reached_max_episodes.add_(potential_next_global_episode_idx >= self.max_episodes)
                if self.reached_max_episodes or self.reached_max_minutes or self.reached_goal_mean_reward:
                    self.get_out_signal.add_(1)
                    break
                # else go work on another episode
                global_episode_idx = self.stats['episode'].add_(1).item() - 1

        while rank == 0 and self.stats['n_active_workers'].item() > 1:
            pass  # Busy-wait

        if rank == 0:
            print(self.ERASE_LINE + debug_message)
            if self.reached_max_minutes:
                print(u'--> reached_max_minutes \u2715')
            if self.reached_max_episodes:
                print(u'--> reached_max_episodes \u2715')
            if self.reached_goal_mean_reward:
                print(u'--> reached_goal_mean_reward \u2713')

        env.close()
        del env
        self.stats['n_active_workers'].sub_(1)

    def train(self,
              make_env_fn: Callable,
              make_env_kwargs: dict,
              seed: int,
              gamma: float,
              max_minutes: int,
              max_episodes: int,
              goal_mean_100_reward: int):
        self.make_env_fn = make_env_fn
        self.make_env_kwargs = make_env_kwargs
        self.seed = seed
        self.gamma = gamma

        self.max_minutes = max_minutes
        self.max_episodes = max_episodes
        self.goal_mean_100_reward = goal_mean_100_reward

        env = self.make_env_fn(**self.make_env_kwargs)
        nS, nA = env.observation_space.shape[0], env.action_space.n

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.stats.clear()
        self.stats['episode'] = torch.zeros(1, dtype=torch.int).share_memory_()
        self.stats['result'] = torch.zeros([max_episodes, 5]).share_memory_()
        self.stats['evaluation_scores'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_reward'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_timestep'] = torch.zeros([max_episodes], dtype=torch.int).share_memory_()
        self.stats['episode_exploration'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['episode_elapsed'] = torch.zeros([max_episodes]).share_memory_()
        self.stats['n_active_workers'] = torch.zeros(1, dtype=torch.int).share_memory_()

        self.shared_policy_model = self.policy_model_fn(nS, nA).share_memory()
        self.replay_model = self.shared_policy_model
        self.shared_policy_optimizer = self.policy_optimizer_fn(self.shared_policy_model, self.policy_optimizer_lr)
        self.shared_value_model = self.value_model_fn(nS).share_memory()
        self.shared_value_optimizer = self.value_optimizer_fn(self.shared_value_model, self.value_optimizer_lr)

        self.get_out_lock = mp.Manager().Lock()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_minutes = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_max_episodes = torch.zeros(1, dtype=torch.int).share_memory_()
        self.reached_goal_mean_reward = torch.zeros(1, dtype=torch.int).share_memory_()

        print(f'{self.name} Training start. (seed: {self.seed})')

        self.training_start = time.time()

        workers = [mp.Process(target=self.work,
                              args=(rank, self.make_env_fn(**self.make_env_kwargs), self.policy_model_fn(nS, nA),
                                    self.value_model_fn(nS))) for rank in range(self.n_workers)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        wallclock_time = time.time() - self.training_start

        env.reset(seed=self.seed)
        final_eval_score, score_std = self.evaluate(self.shared_policy_model, env, n_episodes=100)
        env.close()
        del env

        final_episode = self.stats['episode'].item()
        training_time = self.stats['episode_elapsed'][:final_episode + 1].sum().item()

        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(final_eval_score, score_std, training_time, wallclock_time))

        self.stats['result'] = self.stats['result'].numpy()
        self.stats['result'][final_episode:, ...] = np.nan
        self.get_cleaned_checkpoints()
        return self.stats['result'], final_eval_score, training_time, wallclock_time

    def evaluate(self,
                 eval_policy_model: nn.Module,
                 eval_env: gymnasium.Env,
                 n_episodes: int = 1,
                 render: bool = False):
        self.frames.clear()
        results = []
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            results.append(0)
            for step in count():
                # To tensor
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                # Interaction
                action = eval_policy_model.select_greedy_action(state)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                results[-1] += float(reward) * pow(self.gamma, step)
                if render:
                    self.frames.append(eval_env.render())
                if terminated or truncated:
                    break
        return np.mean(results), np.std(results)