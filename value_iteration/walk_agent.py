from typing import Callable, Tuple

import gymnasium
import numpy as np
from prettytable import PrettyTable


class WalkAgent:
    LEFT, RIGHT = range(2)

    def __init__(self, env: gymnasium.Env, n_states: int, policy: Callable[[int], int], gamma: float = 1.0):
        self.env = env
        self.n_states = n_states
        self.n_actions = 2  # LEFT, RIGHT
        self.policy = policy
        self.gamma = gamma
        self.table = PrettyTable()

    def print_policy(self):
        self.table.clear()
        self.table.title = "Policy"
        self.table.add_column(fieldname='State', column=['Policy'])
        for i in range(self.n_states):
            self.table.add_column(fieldname=f'{i}', column=['<-' if self.policy(i) is self.LEFT else '->'])
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
            self.table.add_row(
                [f'Action-Value (<-)'] + [format(action_value[i][0], "5.4f") for i in range(self.n_states)]
            )
            self.table.add_row(
                [f'Action-Value (->)'] + [format(action_value[i][1], "5.4f") for i in range(self.n_states)]
            )
            self.table.add_row([f'Policy'] + ['<-' if new_policy(i) is self.LEFT else '->' for i in range(self.n_states)])
            print(self.table)

        return new_policy

    def policy_iteration(self, theta: float = 1e-10, verbose: bool = True) -> Tuple[np.ndarray, Callable[[int], int]]:
        mdp = self.env.unwrapped.P
        step = 0
        random_actions = np.random.choice(a=tuple(mdp[0].keys()), size=len(mdp))
        self.policy = lambda s: {
            s: int(a) for s, a in enumerate(random_actions)
        }[s]

        if verbose:
            self.table.clear()
            self.table.title = "Policy Iteration Log"
            self.table.field_names = ['State'] + [i for i in range(self.n_states)]

        while True:
            old_policy = {s: self.policy(s) for s in range(len(mdp))}

            if verbose:
                self.table.add_row(
                    [f'Policy({step})'] + ['<-' if old_policy[i] is self.LEFT else '->' for i in range(self.n_states)]
                )

            state_value = self.evaluate_policy(theta=theta, verbose=False)
            improved_policy = self.get_improved_policy(state_value, verbose=False)
            if old_policy == {s: improved_policy(s) for s in range(len(mdp))}:
                break
            self.policy = improved_policy
            step += 1

        if verbose:
            print(self.table)

        return state_value, self.policy

    def value_iteration(self, theta: float = 1e-10, verbose: bool = True) -> Tuple[np.ndarray, Callable[[int], int]]:
        mdp = self.env.unwrapped.P
        state_value = np.zeros(self.n_states, dtype=np.float64)
        step = 0

        if verbose:
            self.table.clear()
            self.table.title = "Value Iteration Log"
            self.table.field_names = ['State'] + [i for i in range(self.n_states)]

        while True:
            action_value = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    transitions = mdp[state][action]
                    for t in transitions:
                        prob, next_state, reward, done = t
                        action_value[state][action] += prob * (
                                    reward + self.gamma * state_value[next_state] * (not done))

            if verbose:
                self.table.add_row(
                    [f'Action-Value {step}'] + [format(action_value[i][0], "5.4f") +
                                                "/" + format(action_value[i][1], "5.4f") for i in range(self.n_states)]
                )

            new_state_value = np.max(action_value, axis=1)
            if np.max(np.abs(new_state_value - state_value)) < theta:
                break
            state_value = new_state_value
            step += 1

        self.policy = lambda s: {
            s: int(a) for s, a in enumerate(np.argmax(action_value, axis=1))
        }[s]

        if verbose:
            self.table.add_row(
                ['Optimal Policy'] + ['<-' if self.policy(i) is self.LEFT else '->' for i in range(self.n_states)]
            )
            print(self.table)

        return state_value, self.policy

    def __repr__(self):
        self.table.clear()
        self.table.title = "Current Agent Information"
        self.table.add_column(fieldname='State', column=['Policy', 'State-Value'])
        state_value = self.evaluate_policy(verbose=False)
        for i in range(self.n_states):
            self.table.add_column(
                fieldname=f'{i}', column=['<-' if self.policy(i) is self.LEFT else '->', format(state_value[i], "5.4f")])

        repr_str = (self.table.get_string() +
                    f"\nSuccess Rate: {self.calc_success_prob() * 100:.2f}%, " +
                    f"Mean discounted reward sum: {self.calc_mean_return()}")

        return repr_str
