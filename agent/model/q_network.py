import argparse
from typing import Callable, Tuple

import gymnasium
from torch import Tensor
from torch.optim import Optimizer

import torch
import torch.nn as nn

import numpy as np
from itertools import count

import os.path
import random
import glob
import time
import os
import gc

from agent.arch import Q
from agent.experience import Experience
from agent.exploration_strategy import ExplorationStrategy
from agent.experience_buffer import ExperienceBuffer
from utils.utils import create_video


class QNetwork:
    ERASE_LINE = '\x1b[2K'

    def __init__(self,
                 value_model_fn: Callable[[int, int], Q],
                 value_optimizer_fn: Callable[[nn.Module, float], Optimizer],
                 value_optimizer_lr: float,
                 experience_buffer: ExperienceBuffer,
                 use_target_network: bool,
                 use_double_learning: bool,
                 training_strategy_fn: Callable[[], ExplorationStrategy],
                 evaluation_strategy_fn: Callable[[], ExplorationStrategy],
                 epochs: int,
                 update_target_every_steps: int,
                 tau: float,
                 args: argparse.Namespace):
        assert use_target_network is True or use_double_learning is False, \
            'If you want to use double learning, must use target network with it.'

        self.name = args.model_name
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.make_env_fn = None
        self.make_env_kwargs = None

        self.seed = 0
        self.gamma = 1.0
        self.epochs = epochs
        self.tau = tau
        self.use_double_learning = use_double_learning
        self.max_gradient_norm = args.max_gradient_norm
        self.leave_print_every_n_secs = args.leave_print_every_n_secs
        self.model_out_dir = args.model_out_dir
        self.video_out_dir = args.video_out_dir

        # stats
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []
        self.frames = []

        self.online_model = None
        self.target_model = None
        self.use_target_network = use_target_network
        self.update_target_every_steps = update_target_every_steps
        self.value_optimizer = None
        self.experience_buffer = experience_buffer
        self.training_strategy = None
        self.evaluation_strategy = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def optimize_double_learning_model_with_target(self, experiences: Experience):
        idxs, weights, (states, actions, rewards, next_states, is_terminals) = experiences.decompose()
        batch_size = len(is_terminals)

        argmax_a_q_sp = self.online_model(next_states).max(1)[1]
        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        q_sa = self.online_model(states).gather(1, actions)

        td_errors = q_sa - target_q_sa
        value_loss = (weights * td_errors).pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.max_gradient_norm)
        self.value_optimizer.step()
        self.experience_buffer.update(idxs, np.abs(td_errors.detach().cpu().numpy()))

    def optimize_model_with_target(self, experiences: Experience):
        idxs, weights, (states, actions, rewards, next_states, is_terminals) = experiences.decompose()

        max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions)

        td_errors = q_sa - target_q_sa
        value_loss = (weights * td_errors).pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.max_gradient_norm)
        self.value_optimizer.step()
        self.experience_buffer.update(idxs, np.abs(td_errors.detach().cpu().numpy()))

    def optimize_model_without_target(self, experiences: Experience):
        idxs, weights, (states, actions, rewards, next_states, is_terminals) = experiences.decompose()

        max_a_q_sp = self.online_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_s = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        q_sa = self.online_model(states).gather(1, actions)

        td_errors = q_sa - target_q_s
        value_loss = (weights * td_errors).pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.max_gradient_norm)
        self.value_optimizer.step()
        self.experience_buffer.update(idxs, np.abs(td_errors.detach().cpu().numpy()))

    def optimize_model_fn(self) -> Callable[[Experience], None]:
        if self.use_double_learning and self.use_target_network:
            return self.optimize_double_learning_model_with_target
        elif self.use_double_learning is False and self.use_target_network:
            return self.optimize_model_with_target
        else:
            return self.optimize_model_without_target

    def interaction_step(self, state: np.ndarray, env: gymnasium.Env, step: int):
        """
        Get experience tuple from interaction with environment.
        :param state: Current state
        :param env: Environment
        :param step: Current step
        :return: tuple (new state, is_terminal)
        """
        if not isinstance(state, Tensor):
            x = torch.tensor(state, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        action = self.training_strategy.select_action(self.online_model, x)
        new_state, reward, terminated, truncated, _ = env.step(action)
        experience = (state, action, reward, new_state, float(terminated and not truncated))

        self.experience_buffer.store(experience)
        self.episode_reward[-1] += reward * pow(self.gamma, step)
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        return new_state, terminated or truncated

    def update_network(self):
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.data.copy_(online.data)

    def update_network_with_polyak_averaging(self):
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.data.copy_((1.0 - self.tau) * target.data + self.tau * online.data)

    def update_network_fn(self):
        if self.tau == 1.0:
            return self.update_network
        else:
            return self.update_network_with_polyak_averaging

    def train(self,
              make_env_fn: Callable,
              make_env_kwargs: dict,
              seed: int,
              gamma: float,
              max_minutes: int,
              max_episodes: int,
              goal_mean_100_reward: int):
        training_start, last_debug_time = time.time(), float('-inf')

        self.make_env_fn = make_env_fn
        self.make_env_kwargs = make_env_kwargs
        self.seed = seed
        self.gamma = gamma

        env = self.make_env_fn(**self.make_env_kwargs)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        n_states, n_actions = env.observation_space.shape[0], env.action_space.n
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        self.online_model = self.value_model_fn(n_states, n_actions).to(self.device)
        if self.use_target_network:
            self.target_model = self.value_model_fn(n_states, n_actions).to(self.device)
            self.update_network()
        update_network = self.update_network_fn()
        self.value_optimizer = self.value_optimizer_fn(self.online_model, self.value_optimizer_lr)
        optimize_model = self.optimize_model_fn()

        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn()
        self.experience_buffer.clear()

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

            for step in count():
                state, is_terminal = self.interaction_step(state, env, step)

                if self.experience_buffer.can_optimize():
                    experiences = self.experience_buffer.sample()
                    experiences.load(self.device)
                    for _ in range(self.epochs):
                        optimize_model(experiences)
                    self.experience_buffer.after_opti_process()

                if self.use_target_network and np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    update_network()

                if is_terminal:
                    gc.collect()
                    break

            # elapsed time stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed

            # evaluation
            evaluation_score, _ = self.evaluate(self.online_model, env)
            self.evaluation_scores.append(evaluation_score)

            # save model weight
            self.save_checkpoint(episode - 1, self.online_model)

            # stats
            cumulated_step = int(np.sum(self.episode_timestep))
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
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

        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
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
        """
        Evaluate a model on the given environment.
        :param eval_policy_model: action-value model
        :param eval_env: Environment
        :param n_episodes: Number of episodes
        :param render: Whether to render the environment
        :return: Tuple (mean of returns, std of returns)
        """
        self.frames.clear()
        results = []
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            results.append(0)
            for step in count():
                if not isinstance(state, Tensor):
                    state = torch.tensor(state, device=self.device, dtype=torch.float32)
                    state = state.unsqueeze(0)
                action = self.evaluation_strategy.select_action(eval_policy_model, state)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                results[-1] += reward * pow(self.gamma, step)
                if render:
                    self.frames.append(eval_env.render())
                if terminated or truncated:
                    break
        return np.mean(results), np.std(results)

    def get_cleaned_checkpoints(self, n_checkpoints: int = 5):
        try:
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.model_out_dir, '*.tar'))
        paths_dic = {}
        for path in paths:
            if int(path.split('_')[-1].split('.')[0]) == self.seed:
                paths_dic[int(path.split('_')[-3])] = path
        last_ep = max(paths_dic.keys())
        checkpoint_idxs = np.linspace(1, last_ep + 1, n_checkpoints, endpoint=True, dtype=np.int32) - 1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def demo_last(self, n_episodes: int = 3):
        """
        Record last episode video of current agent in training.
        """
        env = self.make_env_fn(**self.make_env_kwargs)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.online_model.load_state_dict(torch.load(checkpoint_paths[last_ep], weights_only=True))

        for ep in range(n_episodes):
            self.evaluate(self.online_model, env, n_episodes=1, render=True)
            create_video(
                self.frames,
                env.metadata['render_fps'],
                os.path.join(
                    self.video_out_dir,
                    'env_{}_model_{}_seed_{}_trial_{}_last'.format(
                        self.make_env_kwargs['env_name'],
                        self.name,
                        checkpoint_paths[last_ep].split('_')[-1].split('.')[0],
                        ep
                    )
                )
            )

        env.close()
        del env

    def demo_progression(self):
        """
        Record checkpoint videos of current agent in training.
        The checkpoints were sampled at equal intervals from 1 to last episode.
        """
        env = self.make_env_fn(**self.make_env_kwargs)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.online_model.load_state_dict(torch.load(checkpoint_paths[i], weights_only=True))
            self.evaluate(self.online_model, env, n_episodes=1, render=True)
            create_video(
                self.frames,
                env.metadata['render_fps'],
                os.path.join(
                    self.video_out_dir,
                    'env_{}_model_{}_ep_{}_seed_{}'.format(
                        self.make_env_kwargs['env_name'],
                        self.name,
                        checkpoint_paths[i].split('_')[-3],
                        checkpoint_paths[i].split('_')[-1].split('.')[0],
                    )
                )
            )

        env.close()
        del env

    def save_checkpoint(self, episode_idx: int, model: nn.Module):
        torch.save(
            model.state_dict(),
            os.path.join(
                self.model_out_dir,
                'env_{}_model_{}_ep_{}_seed_{}.tar'.format(
                    self.make_env_kwargs['env_name'], self.name, episode_idx, self.seed)
            ),
        )