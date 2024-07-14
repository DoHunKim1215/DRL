import argparse
import random
import time
from itertools import count
from typing import Callable

import gymnasium
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from model.model import RLModel
from model.policy_arch import PNetwork


class SyncActorCriticModel(RLModel):

    def __init__(self,
                 ac_model_fn: Callable[[int, int], PNetwork],
                 ac_optimizer_fn: Callable[[nn.Module, float], Optimizer],
                 ac_optimizer_lr: float,
                 policy_loss_weight: float,
                 value_loss_weight: float,
                 entropy_loss_weight: float,
                 max_n_steps: int,
                 n_workers: int,
                 tau: float,
                 args: argparse.Namespace):
        super(SyncActorCriticModel, self).__init__(args)

        assert n_workers > 1

        self.make_envs_fn = None
        self.ac_model = None
        self.ac_optimizer = None

        self.ac_model_fn = ac_model_fn
        self.ac_model_max_gradient_norm = args.ac_model_max_gradient_norm
        self.ac_optimizer_fn = ac_optimizer_fn
        self.ac_optimizer_lr = ac_optimizer_lr

        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

        self.max_n_steps = max_n_steps
        self.n_workers = n_workers
        self.tau = tau

        self.running_timestep = None
        self.running_reward = None
        self.running_exploration = None
        self.running_seconds = None

        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.values = []

    def optimize_model(self):
        log_prob = torch.stack(self.log_probs).squeeze()
        entropies = torch.stack(self.entropies).squeeze()
        values = torch.stack(self.values).squeeze()

        T = len(self.rewards)
        gamma = self.gamma * 0.99
        discounts = np.logspace(0, T, num=T, base=gamma, endpoint=False)
        rewards = np.array(self.rewards).squeeze()
        returns = np.array([[np.sum(discounts[:T - t] * rewards[t:, w]) for t in range(T)]
                            for w in range(self.n_workers)])

        np_values = values.data.numpy()
        tau_discounts = np.logspace(0, T - 1, num=T - 1, base=gamma * self.tau, endpoint=False)
        advs = rewards[:-1] + gamma * np_values[1:] - np_values[:-1]
        gaes = np.array([[np.sum(tau_discounts[:T - 1 - t] * advs[t:, w]) for t in range(T - 1)]
                         for w in range(self.n_workers)])
        discounted_gaes = discounts[:-1] * gaes

        values = values[:-1, ...].view(-1).unsqueeze(1)
        log_prob = log_prob.view(-1).unsqueeze(1)
        entropies = entropies.view(-1).unsqueeze(1)
        returns = torch.FloatTensor(returns.T[:-1]).contiguous().view(-1).unsqueeze(1)
        discounted_gaes = torch.FloatTensor(discounted_gaes.T).contiguous().view(-1).unsqueeze(1)

        T -= 1
        T *= self.n_workers
        assert returns.size() == (T, 1)
        assert values.size() == (T, 1)
        assert log_prob.size() == (T, 1)
        assert entropies.size() == (T, 1)

        value_error = returns.detach() - values
        value_loss = value_error.pow(2).mul(0.5).mean() * self.value_loss_weight
        policy_loss = -(discounted_gaes.detach() * log_prob).mean() * self.policy_loss_weight
        entropy_loss = -entropies.mean() * self.entropy_loss_weight
        self.ac_optimizer.zero_grad()
        (value_loss + policy_loss + entropy_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), self.ac_model_max_gradient_norm)
        self.ac_optimizer.step()

    def interaction_step(self, states, envs):
        states = torch.tensor(states, dtype=torch.float32)
        actions, is_exploratory, log_probs, entropies, values = self.ac_model.select_action_informatively(states)
        new_states, rewards, is_terminals, is_truncated = envs.step(actions)

        self.log_probs.append(log_probs)
        self.entropies.append(entropies)
        self.rewards.append(rewards)
        is_failures = np.logical_and(is_terminals, np.logical_not(is_truncated))
        self.values.append(values * torch.Tensor(1. - is_failures))

        self.running_reward += rewards
        self.running_timestep += 1
        self.running_exploration += is_exploratory[:, np.newaxis].astype(np.int32)

        return new_states, np.logical_or(is_terminals, is_truncated), is_truncated

    def train(self,
              make_envs_fn: Callable,
              make_env_fn: Callable,
              make_env_kwargs: dict,
              seed: int,
              gamma: float,
              max_minutes: int,
              max_episodes: int,
              goal_mean_100_reward: int):
        training_start = time.time()
        last_debug_time = float('-inf')

        self.make_envs_fn = make_envs_fn
        self.make_env_fn = make_env_fn
        self.make_env_kwargs = make_env_kwargs
        self.seed = seed
        self.gamma = gamma

        env = self.make_env_fn(**self.make_env_kwargs)
        env.reset(seed=self.seed)
        envs = self.make_envs_fn(make_env_fn, make_env_kwargs, self.seed, self.n_workers)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.running_timestep = np.array([[0.], ] * self.n_workers)
        self.running_reward = np.array([[0.], ] * self.n_workers)
        self.running_exploration = np.array([[0.], ] * self.n_workers)
        self.running_seconds = np.array([[time.time()], ] * self.n_workers)
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        self.ac_model = self.ac_model_fn(nS, nA)
        self.replay_model = self.ac_model
        self.ac_optimizer = self.ac_optimizer_fn(self.ac_model, self.ac_optimizer_lr)

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        states = envs.reset(seed=self.seed)

        print(f'{self.name} Training start. (seed: {self.seed})')

        # collect n_steps rollout
        episode = 0
        n_steps_start = 0
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.values = []
        for step in count(start=1):
            states, is_terminals, is_truncated = self.interaction_step(states, envs)

            if is_terminals.sum() or step - n_steps_start == self.max_n_steps:
                states_tensor = torch.tensor(states, dtype=torch.float32)
                is_failure = np.logical_and(is_terminals, np.logical_not(is_truncated))
                next_values = self.ac_model.evaluate_state(states_tensor).detach().numpy() * (1 - is_failure)
                self.rewards.append(next_values)
                self.values.append(torch.Tensor(next_values))

                self.optimize_model()

                self.log_probs = []
                self.entropies = []
                self.rewards = []
                self.values = []
                n_steps_start = step

            # stats
            if is_terminals.sum():
                episode_done = time.time()
                evaluation_score, _ = self.evaluate(self.ac_model, env)
                self.save_checkpoint(episode, self.ac_model)

                for i in range(self.n_workers):
                    if is_terminals[i]:
                        states[i] = envs.reset(rank=i)
                        self.episode_timestep.append(self.running_timestep[i][0])
                        self.episode_reward.append(self.running_reward[i][0])
                        self.episode_exploration.append(self.running_exploration[i][0] / self.running_timestep[i][0])
                        self.episode_seconds.append(episode_done - self.running_seconds[i][0])
                        training_time += self.episode_seconds[-1]
                        self.evaluation_scores.append(evaluation_score)
                        episode += 1

                        mean_10_reward = np.mean(self.episode_reward[-10:])
                        std_10_reward = np.std(self.episode_reward[-10:])
                        mean_100_reward = np.mean(self.episode_reward[-100:])
                        std_100_reward = np.std(self.episode_reward[-100:])
                        mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
                        std_100_eval_score = np.std(self.evaluation_scores[-100:])
                        mean_100_exp_rat = np.mean(self.episode_exploration[-100:])
                        std_100_exp_rat = np.std(self.episode_exploration[-100:])

                        total_step = int(np.sum(self.episode_timestep))
                        wallclock_elapsed = time.time() - training_start
                        result[episode - 1] = total_step, mean_100_reward, \
                            mean_100_eval_score, training_time, wallclock_elapsed

                # debug stuff
                reached_debug_time = time.time() - last_debug_time >= self.leave_print_every_n_secs
                reached_max_minutes = wallclock_elapsed >= max_minutes * 60
                reached_max_episodes = episode + self.n_workers >= max_episodes
                reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
                training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward

                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
                debug_message = '[{}] episode {:04}, cumulated step {:06}, '
                debug_message += 'MA10 reward {:05.1f}\u00B1{:05.1f}, '
                debug_message += 'MA100 reward {:05.1f}\u00B1{:05.1f}, '
                debug_message += 'MA100 exp rate {:02.1f}\u00B1{:02.1f}, '
                debug_message += 'MA100 evaluation return {:05.1f}\u00B1{:05.1f}'
                debug_message = debug_message.format(
                    elapsed_str, episode - 1, total_step, mean_10_reward, std_10_reward,
                    mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                    mean_100_eval_score, std_100_eval_score)

                print(debug_message, end='\r', flush=True)

                if reached_debug_time or training_is_over:
                    print(self.ERASE_LINE + debug_message, flush=True)
                    last_debug_time = time.time()

                if training_is_over:
                    if reached_max_minutes:
                        print(u'--> reached_max_minutes \u2715')
                    if reached_max_episodes:
                        print(u'--> reached_max_episodes \u2715')
                    if reached_goal_mean_reward:
                        print(u'--> reached_goal_mean_reward \u2713')
                    break

                # reset running variables for next time around
                self.running_timestep *= 1 - is_terminals
                self.running_reward *= 1 - is_terminals
                self.running_exploration *= 1 - is_terminals
                self.running_seconds[is_terminals.astype(np.bool_)] = time.time()

        final_eval_score, score_std = self.evaluate(self.ac_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(final_eval_score, score_std, training_time, wallclock_time))
        env.close()
        del env
        envs.close()
        del envs
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time

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
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = eval_policy_model.select_greedy_action(state)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                results[-1] += float(reward) * pow(self.gamma, step)
                if render:
                    self.frames.append(eval_env.render())
                if terminated or truncated:
                    break
        return np.mean(results), np.std(results)
