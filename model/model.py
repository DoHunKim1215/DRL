import argparse
import glob
import os
from abc import ABC, abstractmethod
from typing import Callable

import gymnasium
import numpy as np
import torch
from torch import nn

from utils.utils import create_video


class RLModel(ABC):
    ERASE_LINE = '\x1b[2K'

    def __init__(self, args: argparse.Namespace):
        self.name = args.model_name

        self.make_env_fn = None
        self.make_env_kwargs = None

        self.seed = 0
        self.gamma = 1.0

        self.model_out_dir = args.model_out_dir
        self.video_out_dir = args.video_out_dir

        self.replay_model = None

        # stats
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []
        self.frames = []

        self.leave_print_every_n_secs = args.leave_print_every_n_secs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def evaluate(self,
                 eval_model: nn.Module,
                 eval_env: gymnasium.Env,
                 n_episodes: int,
                 render: bool):
        pass

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
        env = self.make_env_fn(**self.make_env_kwargs)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.replay_model.load_state_dict(torch.load(checkpoint_paths[last_ep], weights_only=True))

        for ep in range(n_episodes):
            self.evaluate(self.replay_model, env, n_episodes=1, render=True)
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
        env = self.make_env_fn(**self.make_env_kwargs)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.replay_model.load_state_dict(torch.load(checkpoint_paths[i], weights_only=True))
            self.evaluate(self.replay_model, env, n_episodes=1, render=True)
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
