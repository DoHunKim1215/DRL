import argparse
import gc
import random
import time
from itertools import count
from typing import Callable

import gymnasium
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from model.agent.rl_model import RLModel


class PolicyModel(RLModel):

    def __init__(self,
                 policy_model_fn: Callable[[int, int], nn.Module],
                 policy_optimizer_fn: Callable[[nn.Module, float], Optimizer],
                 policy_optimizer_lr: float,
                 entropy_loss_weight: float,
                 args: argparse.Namespace,
                 use_value_model: bool = False,
                 value_model_fn: Callable[[int], nn.Module] = None,
                 value_optimizer_fn: Callable[[nn.Module, float], Optimizer] = None,
                 value_optimizer_lr: float = None):
        super(PolicyModel, self).__init__(args)

        if use_value_model:
            assert value_model_fn is not None and value_optimizer_fn is not None and value_optimizer_lr is not None, \
                'If you use value model, you must specify the information required for making value model.'

        self.policy_model_fn = policy_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr

        self.use_value_model = use_value_model
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.entropy_loss_weight = entropy_loss_weight

        self.policy_model_max_gradient_norm = args.policy_model_max_gradient_norm
        self.value_model_max_gradient_norm = args.value_model_max_gradient_norm

        self.policy_model = None
        self.policy_optimizer = None
        self.value_model = None
        self.value_optimizer = None

        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.values = []

    def optimize_only_policy_model(self):
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T - t] * self.rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts).to(self.device).unsqueeze(1)
        returns = torch.FloatTensor(returns).to(self.device).unsqueeze(1)

        self.log_probs = torch.cat(self.log_probs)
        self.entropies = torch.cat(self.entropies)

        policy_loss = -(discounts * returns * self.log_probs).mean()
        entropy_loss = -self.entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def optimize_policy_and_value_model(self):
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T - t] * self.rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts).to(self.device).unsqueeze(1)
        returns = torch.FloatTensor(returns).to(self.device).unsqueeze(1)

        self.log_probs = torch.cat(self.log_probs)
        self.entropies = torch.cat(self.entropies)
        self.values = torch.cat(self.values)

        value_error = returns - self.values
        policy_loss = -(discounts * value_error.detach() * self.log_probs).mean()
        entropy_loss = -self.entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.policy_model_max_gradient_norm)
        self.policy_optimizer.step()

        value_loss = value_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.value_model_max_gradient_norm)
        self.value_optimizer.step()

    def optimize_model_fn(self):
        if self.use_value_model:
            return self.optimize_policy_and_value_model
        else:
            return self.optimize_only_policy_model

    def interaction_step(self, state: np.ndarray, env: gymnasium.Env, step: int):
        # To tensor
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)

        # Interaction
        action, is_exploratory, log_prob, entropy = self.policy_model.select_action_informatively(state_tensor)
        new_state, reward, terminated, truncated, _ = env.step(action)

        # Record
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.entropies.append(entropy)
        if self.use_value_model:
            is_failure = terminated and not truncated
            self.values.append(self.value_model(state_tensor) * (1.0 - float(is_failure)))

        self.episode_reward[-1] += float(reward) * pow(self.gamma, step)
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(is_exploratory)
        return new_state, terminated or truncated

    def train(self,
              make_env_fn: Callable,
              make_env_kwargs: dict,
              seed: int,
              gamma: float,
              max_minutes: int,
              max_episodes: int,
              goal_mean_100_reward: int):
        training_start = time.time()
        last_debug_time = float('-inf')

        self.make_env_fn = make_env_fn
        self.make_env_kwargs = make_env_kwargs
        self.seed = seed
        self.gamma = gamma

        env = self.make_env_fn(**self.make_env_kwargs)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n
        self.episode_timestep.clear()
        self.episode_reward.clear()
        self.episode_seconds.clear()
        self.episode_exploration.clear()
        self.evaluation_scores.clear()

        self.policy_model = self.policy_model_fn(nS, nA).to(self.device)
        self.replay_model = self.policy_model
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, self.policy_optimizer_lr)

        if self.use_value_model:
            self.value_model = self.value_model_fn(nS).to(self.device)
            self.value_optimizer = self.value_optimizer_fn(self.value_model, self.value_optimizer_lr)

        optimize_model = self.optimize_model_fn()

        print(f'{self.name} Training start. (seed: {self.seed})')

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()

            state, _ = env.reset(seed=self.seed)
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            # collect rollout
            self.log_probs = []
            self.rewards = []
            self.entropies = []
            self.values = []
            for step in count():
                state, terminated = self.interaction_step(state, env, step)
                if terminated:
                    gc.collect()
                    break
            optimize_model()

            # elapsed time stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed

            # evaluation
            evaluation_score, _ = self.evaluate(self.policy_model, env)
            self.evaluation_scores.append(evaluation_score)

            # save model weight
            self.save_checkpoint(episode - 1, self.policy_model)

            # stats
            cumulated_step = int(np.sum(self.episode_timestep))
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            result[episode - 1] = cumulated_step, mean_100_reward, mean_100_eval_score, training_time, wallclock_elapsed

            reached_debug_time = time.time() - last_debug_time >= self.leave_print_every_n_secs

            # termination conditions
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward

            # debug message
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = '[{}] episode {:04}, cumulated step {:06}, '
            debug_message += 'MA10 reward {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'MA100 reward {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'MA100 exp rate {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'MA100 evaluation return {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode - 1, cumulated_step, mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score
            )
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

        final_eval_score, score_std = self.evaluate(self.policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(final_eval_score, score_std, training_time, wallclock_time))
        env.close()
        del env
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
                # To tensor
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
                state = state.unsqueeze(0)
                # Interaction
                action = eval_policy_model.select_greedy_action(state)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                results[-1] += float(reward) * pow(self.gamma, step)
                if render:
                    self.frames.append(eval_env.render())
                if terminated or truncated:
                    break
        return np.mean(results), np.std(results)
