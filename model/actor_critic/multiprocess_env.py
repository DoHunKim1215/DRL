from typing import Callable

import gymnasium
import numpy as np
import torch.multiprocessing as mp


def work(seed: int, env: gymnasium.Env, worker_end):
    env.reset(seed=seed)
    while True:
        cmd, kwargs = worker_end.recv()
        if cmd == 'reset':
            worker_end.send(env.reset(**kwargs))
        elif cmd == 'step':
            worker_end.send(env.step(**kwargs))
        else:
            # including close command
            env.close()
            del env
            worker_end.close()
            break


class MultiprocessEnv:
    CLOSE = 'close'
    RESET = 'reset'
    STEP = 'step'

    def __init__(self,
                 make_env_fn: Callable,
                 make_env_kwargs: dict,
                 seed: int,
                 n_workers: int):

        self.make_env_fn = make_env_fn
        self.make_env_kwargs = make_env_kwargs
        self.seed = seed
        self.n_workers = n_workers

        self.pipes = []
        self.workers = []
        for rank in range(self.n_workers):
            parent_conn, child_conn = mp.Pipe()
            self.workers.append(
                mp.Process(target=work, args=(self.seed + rank, self.make_env_fn(**self.make_env_kwargs), child_conn))
            )
            self.pipes.append(parent_conn)
        for w in self.workers:
            w.start()

    def reset(self, rank: int = None, **kwargs):
        if rank is not None:
            self.send_msg((self.RESET, kwargs), rank)
            initial_state, _ = self.pipes[rank].recv()
            return initial_state
        else:
            self.broadcast_msg((self.RESET, kwargs))
            return np.vstack([parent_end.recv()[0] for parent_end in self.pipes])

    def step(self, actions):
        assert len(actions) == self.n_workers
        for rank in range(self.n_workers):
            self.send_msg((self.STEP, {'action': actions[rank]}), rank)
        results = []
        for rank in range(self.n_workers):
            observation, reward, terminated, truncated, _ = self.pipes[rank].recv()
            results.append((observation,
                            np.array(reward, dtype=np.float32),
                            np.array(terminated, dtype=np.float32),
                            np.array(truncated, dtype=np.float32)))
        return [np.vstack(block) for block in np.array(results, dtype=object).T]

    def close(self):
        self.broadcast_msg((self.CLOSE, {}))
        for w in self.workers:
            w.join()

    def send_msg(self, msg, rank):
        self.pipes[rank].send(msg)

    def broadcast_msg(self, msg):
        for parent_end in self.pipes:
            parent_end.send(msg)
