import argparse
import gc
import random
import time
from itertools import count
from typing import Callable, Any

import gymnasium
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from model.agent.rl_model import RLModel
from model.experience.experience_buffer import ExperienceBuffer
from model.net.policy_net import StochasticPNetwork, DeterministicPNetwork
from model.net.q_net import QNetwork
from model.strategy.exploration_strategy import ExplorationStrategy


class AdvantageActorCriticModel(RLModel):

    def __init__(self,
                 ac_model_fn: Callable[[int, int], StochasticPNetwork],
                 ac_optimizer_fn: Callable[[nn.Module, float], Optimizer],
                 ac_optimizer_lr: float,
                 policy_loss_weight: float,
                 value_loss_weight: float,
                 entropy_loss_weight: float,
                 max_n_steps: int,
                 n_workers: int,
                 tau: float,
                 args: argparse.Namespace):
        super(AdvantageActorCriticModel, self).__init__(args)

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
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        rewards = np.array(self.rewards).squeeze()
        returns = np.array([[np.sum(discounts[:T - t] * rewards[t:, w]) for t in range(T)]
                            for w in range(self.n_workers)])

        np_values = values.data.numpy()
        tau_discounts = np.logspace(0, T - 1, num=T - 1, base=self.gamma * self.tau, endpoint=False)
        advs = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]
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
                results[-1] += float(reward)
                if render:
                    self.frames.append(eval_env.render())
                if terminated or truncated:
                    break
        return np.mean(results), np.std(results)


class DeepDeterministicPolicyGradientModel(RLModel):

    def __init__(self,
                 replay_buffer_fn: Callable[[], ExperienceBuffer],
                 policy_model_fn: [[int, Any, Any], DeterministicPNetwork],
                 policy_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 policy_optimizer_lr: float,
                 policy_max_gradient_norm: float,
                 value_model_fn: [[int, int], QNetwork],
                 value_optimizer_fn: Callable[[nn.Module, float], torch.optim.Optimizer],
                 value_optimizer_lr: float,
                 value_max_gradient_norm: float,
                 training_strategy_fn: Callable[[Any], ExplorationStrategy],
                 evaluation_strategy_fn: Callable[[Any], ExplorationStrategy],
                 n_warmup_batches: int,
                 update_value_target_every_steps: int,
                 update_policy_target_every_steps: int,
                 train_policy_every_steps: int,
                 tau: float,
                 policy_noise_ratio: float,
                 policy_noise_clip_ratio: float,
                 use_double_learning: bool,
                 args: argparse.Namespace):
        super().__init__(args)

        self.replay_buffer_fn = replay_buffer_fn
        self.replay_buffer = None

        self.policy_model_fn = policy_model_fn
        self.policy_max_grad_norm = policy_max_gradient_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr

        self.target_policy_model = None
        self.online_policy_model = None
        self.policy_optimizer = None

        self.value_model_fn = value_model_fn
        self.value_max_grad_norm = value_max_gradient_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr

        self.target_value_model = None
        self.online_value_model = None
        self.value_optimizer = None

        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.training_strategy = None
        self.evaluation_strategy = None

        self.n_warmup_batches = n_warmup_batches
        self.update_value_target_every_steps = update_value_target_every_steps
        self.update_policy_target_every_steps = update_policy_target_every_steps
        self.train_policy_every_steps = train_policy_every_steps
        self.tau = tau
        self.min_samples = 0
        self.policy_noise_ratio = policy_noise_ratio
        self.policy_noise_clip_ratio = policy_noise_clip_ratio
        self.use_double_learning = use_double_learning

    def optimize_model_with_single_learning(self, experiences):
        _, __, (states, actions, rewards, next_states, is_terminals) = experiences.decompose()

        with torch.no_grad():
            action_min, action_max = self.target_policy_model.env_min, self.target_policy_model.env_max
            noise_min, noise_max = self.policy_noise_clip_ratio * action_min, self.policy_noise_clip_ratio * action_max

            a_noise = torch.randn(size=actions.shape).to(self.device) * self.policy_noise_ratio * (action_max - action_min)
            a_noise = torch.clamp(input=a_noise, min=noise_min, max=noise_max)

            argmax_a_q_sp = self.target_policy_model(next_states)
            noisy_argmax_a_q_sp = argmax_a_q_sp + a_noise
            noisy_argmax_a_q_sp = torch.clamp(noisy_argmax_a_q_sp, min=action_min, max=action_max)

            max_a_q_sp = self.target_value_model(next_states, noisy_argmax_a_q_sp)
            target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)

        q_sa = self.online_value_model(states, actions)
        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), self.value_max_grad_norm)
        self.value_optimizer.step()

        if np.sum(self.episode_timestep) % self.train_policy_every_steps == 0:
            argmax_a_q_s = self.online_policy_model(states)
            max_a_q_s = self.online_value_model(states, argmax_a_q_s)
            policy_loss = -max_a_q_s.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), self.policy_max_grad_norm)
            self.policy_optimizer.step()

    def optimize_model_with_double_learning(self, experiences):
        _, __, (states, actions, rewards, next_states, is_terminals) = experiences.decompose()

        with torch.no_grad():
            action_min, action_max = self.target_policy_model.env_min, self.target_policy_model.env_max
            noise_min, noise_max = self.policy_noise_clip_ratio * action_min, self.policy_noise_clip_ratio * action_max

            a_noise = torch.randn(size=actions.shape).to(self.device) * self.policy_noise_ratio * (action_max - action_min)
            a_noise = torch.clamp(input=a_noise, min=noise_min, max=noise_max)

            argmax_a_q_sp = self.target_policy_model(next_states)
            noisy_argmax_a_q_sp = argmax_a_q_sp + a_noise
            noisy_argmax_a_q_sp = torch.clamp(input=noisy_argmax_a_q_sp, min=action_min, max=action_max)

            max_a_q_sp_a, max_a_q_sp_b = self.target_value_model(next_states, noisy_argmax_a_q_sp)
            max_a_q_sp = torch.min(max_a_q_sp_a, max_a_q_sp_b)

            target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)

        q_sa_a, q_sa_b = self.online_value_model(states, actions)
        td_error_a = q_sa_a - target_q_sa
        td_error_b = q_sa_b - target_q_sa

        value_loss = td_error_a.pow(2).mul(0.5).mean() + td_error_b.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), self.value_max_grad_norm)
        self.value_optimizer.step()

        if np.sum(self.episode_timestep) % self.train_policy_every_steps == 0:
            argmax_a_q_s = self.online_policy_model(states)
            max_a_q_s = self.online_value_model.Qa(states, argmax_a_q_s)

            policy_loss = -max_a_q_s.mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), self.policy_max_grad_norm)
            self.policy_optimizer.step()

    def optimize_model_fn(self) -> Callable:
        if self.use_double_learning:
            return self.optimize_model_with_double_learning
        else:
            return self.optimize_model_with_single_learning

    def interaction_step(self, state: np.ndarray, env: gymnasium.Env):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.training_strategy.select_action(
            self.online_policy_model, state_tensor, len(self.replay_buffer) < self.min_samples)
        new_state, reward, is_terminal, is_truncated, _ = env.step(action)
        experience = (state, action, reward, new_state, float(is_terminal and not is_truncated))

        self.replay_buffer.store(experience)
        self.episode_reward[-1] += float(reward)
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += self.training_strategy.ratio_noise_injected
        return new_state, is_terminal or is_truncated

    def update_value_network(self, tau: float):
        for target, online in zip(self.target_value_model.parameters(), self.online_value_model.parameters()):
            mixed_weights = (1.0 - tau) * target.data + tau * online.data
            target.data.copy_(mixed_weights)

    def update_policy_network(self, tau: float):
        for target, online in zip(self.target_policy_model.parameters(), self.online_policy_model.parameters()):
            mixed_weights = (1.0 - tau) * target.data + tau * online.data
            target.data.copy_(mixed_weights)

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
        env.reset(seed=self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
        action_bounds = env.action_space.low, env.action_space.high
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        self.target_value_model = self.value_model_fn(nS, nA).to(self.device)
        self.online_value_model = self.value_model_fn(nS, nA).to(self.device)
        self.update_value_network(tau=1.0)

        self.target_policy_model = self.policy_model_fn(nS, action_bounds, self.device).to(self.device)
        self.online_policy_model = self.policy_model_fn(nS, action_bounds, self.device).to(self.device)
        self.update_policy_network(tau=1.0)

        self.replay_model = self.online_policy_model

        self.value_optimizer = self.value_optimizer_fn(self.online_value_model, self.value_optimizer_lr)
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model, self.policy_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn(action_bounds)
        self.evaluation_strategy = self.evaluation_strategy_fn(action_bounds)

        self.min_samples = self.replay_buffer.batch_size * self.n_warmup_batches

        optimize_model = self.optimize_model_fn()

        print(f'{self.name} Training start. (seed: {self.seed})')

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()

            state, _ = env.reset()
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for _ in count():
                state, is_terminal = self.interaction_step(state, env)
                if self.replay_buffer.can_optimize():
                    experiences = self.replay_buffer.sample()
                    experiences.load(self.device)
                    optimize_model(experiences)

                if np.sum(self.episode_timestep) % self.update_value_target_every_steps == 0:
                    self.update_value_network(self.tau)

                if np.sum(self.episode_timestep) % self.update_policy_target_every_steps == 0:
                    self.update_policy_network(self.tau)

                if is_terminal:
                    gc.collect()
                    break

            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_policy_model, env)
            self.save_checkpoint(episode - 1, self.online_policy_model)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

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
            result[episode - 1] = total_step, mean_100_reward, \
                mean_100_eval_score, training_time, wallclock_elapsed

            reached_debug_time = time.time() - last_debug_time >= self.leave_print_every_n_secs
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = '[{}] episode {:04}, cumulated step {:07}, '
            debug_message += 'reward MA10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'reward MA100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'exp rate MA100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'eval score MA100 {:05.1f}\u00B1{:05.1f}'
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

        final_eval_score, score_std = self.evaluate(self.online_policy_model, env, n_episodes=100)
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
        results = []
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            results.append(0)
            for _ in count():
                state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
                action = self.evaluation_strategy.select_action(eval_policy_model, state)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                results[-1] += float(reward)
                if render:
                    self.frames.append(eval_env.render())
                if terminated or truncated:
                    break
        return np.mean(results), np.std(results)
