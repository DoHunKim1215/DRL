from typing import Callable, Tuple

import plotly.graph_objects as go
import gymnasium
import numpy as np
from prettytable import PrettyTable


class FrozenLakeAgent:
    LEFT, DOWN, RIGHT, UP = range(4)

    @staticmethod
    def get_action_name(action: int) -> str:
        if action == FrozenLakeAgent.UP:
            return 'UP'
        elif action == FrozenLakeAgent.DOWN:
            return 'DOWN'
        elif action == FrozenLakeAgent.LEFT:
            return 'LEFT'
        elif action == FrozenLakeAgent.RIGHT:
            return 'RIGHT'
        else:
            assert False

    def __init__(self, env: gymnasium.Env, n_cols: int, n_rows:int, policy: Callable[[int], int], gamma: float = 1.0):
        self.env = env
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.n_states = n_cols * n_rows
        self.n_actions = 4  # LEFT, DOWN, RIGHT, UP
        self.policy = policy
        self.gamma = gamma
        self.table = PrettyTable()

    def play(self, n_episodes: int = 1):
        frames = []
        state, _ = self.env.reset()
        frames.append(self.env.render())
        for episode in range(n_episodes):
            while True:
                state, reward, terminated, truncated, _ = self.env.step(self.policy(state))
                frames.append(self.env.render())
                if terminated or truncated:
                    state, _ = self.env.reset()
                    break
        return frames

    def print_policy(self):
        self.table.clear()
        self.table.title = "Policy"
        self.table.field_names = [''] + [i for i in range(self.n_cols)]
        for i in range(self.n_rows):
            self.table.add_row(
                [i] + [self.get_action_name(self.policy(i * self.n_rows + j)) for j in range(self.n_cols)]
            )
        print(self.table)

    def calc_success_prob(self, n_episodes=1000, max_steps=200) -> float:
        results = []
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            terminated = False
            steps = 0
            while not terminated and steps < max_steps:
                state, reward, terminated, truncated, info = self.env.step(self.policy(state))
                steps += 1
            results.append(state == self.n_states - 1)
        return np.sum(results) / len(results)

    def calc_mean_return(self, n_episodes=1000, max_steps=200) -> float:
        results = []
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            terminated = False
            steps = 0
            results.append(0.0)
            while not terminated and steps < max_steps:
                state, reward, terminated, truncated, info = self.env.step(self.policy(state))
                results[-1] += reward * pow(self.gamma, steps)
                steps += 1
        return np.mean(results)

    def evaluate_policy(self, theta=1e-10, verbose: bool = True) -> np.ndarray:
        mdp = self.env.unwrapped.P
        state_value = np.zeros(self.n_states, dtype=np.float64)
        step = 0

        if verbose:
            self.table.clear()
            self.table.title = "Policy Evaluation Log"
            self.table.field_names = ['State'] + [i for i in range(self.n_states)]

        while True:
            if verbose:
                self.table.add_row([f'Iter {step}'] + [format(state_value[i], "9.8f") for i in range(self.n_states)])

            new_state_value = np.zeros(self.n_states, dtype=np.float64)
            for state in range(self.n_states):
                transitions = mdp[state][self.policy(state)]
                for t in transitions:
                    prob, next_state, reward, done = t
                    new_state_value[state] += prob * (reward + self.gamma * state_value[next_state] * (not done))
            if np.max(np.abs(state_value - new_state_value)) < theta:
                break
            state_value = new_state_value
            step += 1

        if verbose:
            print(self.table)

        return state_value

    def get_improved_policy(self, state_value: np.ndarray, verbose: bool = True) -> Callable[[int], int]:
        mdp = self.env.unwrapped.P
        action_value = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

        if verbose:
            self.table.clear()
            self.table.title = "Action-Value Function and Optimal Policy"
            self.table.field_names = ['State'] + [i for i in range(self.n_states)]

        for state in range(self.n_states):
            for action in range(self.n_actions):
                transitions = mdp[state][action]
                for t in transitions:
                    prob, next_state, reward, done = t
                    action_value[state][action] += prob * (reward + self.gamma * state_value[next_state] * (not done))

        new_policy = lambda s: {
            s: int(a) for s, a in enumerate(np.argmax(action_value, axis=1))
        }[s]

        if verbose:
            for i in range(self.n_actions):
                self.table.add_row(
                    [f'Action-Value ({self.get_action_name(i)})'] +
                    [format(action_value[j][i], "5.4f") for j in range(self.n_states)]
                )
            self.table.add_row(['Policy'] + [self.get_action_name(new_policy(i)) for i in range(self.n_states)])
            print(self.table)

        return new_policy

    def policy_iteration(self, theta: float = 1e-10, verbose: bool = True) -> Tuple[np.ndarray, Callable[[int], int]]:
        mdp = self.env.unwrapped.P
        step = 0
        random_actions = np.random.choice(a=tuple(mdp[0].keys()), size=len(mdp))
        self.policy = lambda s: {
            s: int(a) for s, a in enumerate(random_actions)
        }[s]

        frames = []
        if verbose:
            frames.extend(self.play())

        while True:
            old_policy = {s: self.policy(s) for s in range(len(mdp))}
            state_value = self.evaluate_policy(theta=theta, verbose=False)
            improved_policy = self.get_improved_policy(state_value, verbose=False)
            if old_policy == {s: improved_policy(s) for s in range(len(mdp))}:
                break
            self.policy = improved_policy
            if verbose:
                frames.extend(self.play())
            step += 1

        if verbose:
            fig = go.Figure(
                data=[go.Image(z=frames[0])],
                layout=go.Layout(
                    autosize=False,
                    width=400, height=400,
                    margin=dict(l=0, r=0, b=0, t=30),
                    xaxis={"title": f"Action History During Policy Iteration"},
                    updatemenus=[
                        dict(
                            type="buttons",
                            buttons=[
                                # play button
                                dict(
                                    label="Play", method="animate",
                                    args=[
                                        None,
                                        {
                                            "frame": {"duration": 50, "redraw": True},
                                            "fromcurrent": True,
                                            "transition": {"duration": 50, "easing": "quadratic-in-out"}
                                        }
                                    ]
                                ),
                                # pause button
                                dict(
                                    label="Pause", method="animate",
                                    args=[
                                        [None],
                                        {
                                            "frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}
                                        }
                                    ]
                                )
                            ],
                            direction="left", pad={"r": 10, "t": 87}, showactive=False,
                            x=0.1, xanchor="right", y=0, yanchor="top"
                        )
                    ],  # updatemenus = [
                ),  # layout = go.Layout(
                frames=[
                    {
                        'data': [go.Image(z=frames[t])],
                        'name': t,
                        'layout': {
                            'xaxis': {'title': f"Action History During Policy Iteration"}
                        }
                    } for t in range(len(frames))
                ]
            )

            # Slider Configuration
            sliders_dict = {
                "active": 0, "yanchor": "top", "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 15}, "prefix": "input time:",
                    "visible": True, "xanchor": "right"
                },
                "transition": {"duration": 100, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9, "x": 0.1, "y": 0,
                "steps": []
            }

            for t in range(len(frames)):
                slider_step = {
                    "label": f"{frames[t]}", "method": "animate",
                    "args": [
                        [t],  # frame 이름과 일치해야 연결됨
                        {
                            "frame": {"duration": 100, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 100}
                        }
                    ],
                }

                sliders_dict["steps"].append(slider_step)

            fig["layout"]["sliders"] = [sliders_dict]

            fig.show()

        return state_value, self.policy

    def value_iteration(self, theta: float = 1e-10, verbose: bool = True) -> Tuple[np.ndarray, Callable[[int], int]]:
        mdp = self.env.unwrapped.P
        state_value = np.zeros(self.n_states, dtype=np.float64)
        step = 0

        while True:
            action_value = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    transitions = mdp[state][action]
                    for t in transitions:
                        prob, next_state, reward, done = t
                        action_value[state][action] += prob * (
                                    reward + self.gamma * state_value[next_state] * (not done))

            new_state_value = np.max(action_value, axis=1)
            if np.max(np.abs(new_state_value - state_value)) < theta:
                break
            state_value = new_state_value
            step += 1

        self.policy = lambda s: {
            s: int(a) for s, a in enumerate(np.argmax(action_value, axis=1))
        }[s]

        return state_value, self.policy

    def __repr__(self):
        self.table.clear()
        self.table.title = "Current Agent Information"
        self.table.add_column(fieldname='State', column=['Policy', 'State-Value'])
        state_value = self.evaluate_policy(verbose=False)
        for i in range(self.n_states):
            self.table.add_column(
                fieldname=f'{i}', column=[self.get_action_name(self.policy(i)), format(state_value[i], "5.4f")])

        repr_str = (self.table.get_string() +
                    f"\nSuccess Rate: {self.calc_success_prob() * 100:.2f}%, " +
                    f"Mean discounted reward sum: {self.calc_mean_return()}")

        return repr_str
