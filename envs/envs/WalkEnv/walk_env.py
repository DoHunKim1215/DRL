import sys
from os import path

import numpy as np
import gymnasium as gym
from gymnasium import spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from typing import Optional, Tuple, Any, Dict
from string import ascii_uppercase

from gymnasium.error import DependencyNotInstalled
from six import StringIO

WEST, EAST = 0, 1


class WalkEnv(gym.Env):

    metadata = {'render_modes': ['human', 'ansi', 'rgb_array'], 'render_fps': 60}

    def __init__(self, n_states=7, p_stay=0.0, p_backward=0.5, render_mode=None):

        assert n_states % 2 == 1, f"The count of non-terminal states must be odd. (n_states : {n_states})"

        # two terminal states added
        self.ncol = n_states + 2
        self.nrow = 1
        self.shape = (1, n_states + 2)
        self.start_state_index = self.shape[1] // 2

        self.desc = np.asarray(["H" + "F" * (n_states // 2) + "S" + "F" * (n_states // 2) + "G"], dtype="c")

        self.nS = nS = np.prod(self.shape)
        self.nA = nA = 2

        self.P = {}
        for s in range(nS):
            self.P[s] = {}
            for a in range(nA):
                p_forward = 1.0 - p_stay - p_backward

                s_forward = np.clip(s - 1 if a == WEST else s + 1, 0, nS - 1) if s != 0 and s != nS - 1 else s
                s_backward = np.clip(s + 1 if a == WEST else s - 1, 0, nS - 1) if s != 0 and s != nS - 1 else s

                r_forward = 1.0 if s == nS - 2 and s_forward == nS - 1 else 0.0
                r_backward = 1.0 if s == nS - 2 and s_backward == nS - 1 else 0.0

                d_forward = s >= nS - 2 and s_forward == nS - 1 or s <= 1 and s_forward == 0
                d_backward = s >= nS - 2 and s_backward == nS - 1 or s <= 1 and s_backward == 0

                self.P[s][a] = [
                    (p_forward, s_forward, r_forward, d_forward),
                    (p_stay, s, 0.0, s == nS - 1 or s == 0),
                    (p_backward, s_backward, r_backward, d_backward)
                ]

        self.isd = np.zeros(nS)
        self.isd[self.start_state_index] = 1.0
        self.last_action = None

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.s = categorical_sample(self.isd, self.np_random)

        # rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * self.ncol, 512), min(64 * self.nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

    def step(self, action) -> Tuple[int, float, bool, bool, Dict[str, float]]:
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.last_action = action

        if self.render_mode == "human":
            self.render()
        return int(s), r, d, False, {"prob": p}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[int, dict]:
        super().reset(seed=seed)
        self.s = categorical_sample(self.isd, self.np_random)
        self.last_action = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {}

    def render(self):
        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

    def _render_text(self):
        outfile = StringIO()
        desc = np.asarray(['[' + ascii_uppercase[:self.shape[1] - 2] + ']'], dtype='c').tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        color = 'red' if self.s == 0 else 'green' if self.s == self.nS - 1 else 'yellow'
        desc[0][self.s] = utils.colorize(desc[0][self.s], color, highlight=True)
        outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        return outfile

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
                self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.last_action if self.last_action is not None else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )