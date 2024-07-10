import argparse

import torch

from agent.arch import FCQ, FCDuelingQ
from agent.exploration_strategy import get_strategy, GreedyStrategy
from agent.model.q_network import QNetwork
from agent.experience_buffer import ReplayBuffer, ExhaustingBuffer, PrioritizedReplayBuffer


def create_q_network(model_name: str, args: argparse.Namespace):
    value_optimizer_fn = lambda net, lr: torch.optim.RMSprop(net.parameters(), lr=lr)
    value_optimizer_lr = args.lr
    training_strategy = get_strategy(args.strategy_name)
    evaluation_strategy = lambda: GreedyStrategy()

    if model_name == 'NFQ':
        value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512, 128))
        replay_buffer = ExhaustingBuffer(batch_size=1024)
        return QNetwork(value_model_fn,
                        value_optimizer_fn,
                        value_optimizer_lr,
                        replay_buffer,
                        False,
                        False,
                        training_strategy,
                        evaluation_strategy,
                        40,
                        args.update_target_every_steps,
                        1.0,
                        args)
    elif model_name == 'DQN':
        value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512, 128))
        replay_buffer = ReplayBuffer(max_size=args.max_buffer_size,
                                     batch_size=64,
                                     n_warmup_batches=args.n_warmup_batches)
        return QNetwork(value_model_fn,
                        value_optimizer_fn,
                        value_optimizer_lr,
                        replay_buffer,
                        True,
                        False,
                        training_strategy,
                        evaluation_strategy,
                        1,
                        args.update_target_every_steps,
                        1.0,
                        args)
    elif model_name == 'DDQN':
        value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512, 128))
        replay_buffer = ReplayBuffer(max_size=args.max_buffer_size,
                                     batch_size=64,
                                     n_warmup_batches=args.n_warmup_batches)
        return QNetwork(value_model_fn,
                        value_optimizer_fn,
                        value_optimizer_lr,
                        replay_buffer,
                        True,
                        True,
                        training_strategy,
                        evaluation_strategy,
                        1,
                        args.update_target_every_steps,
                        1.0,
                        args)
    elif model_name == 'DuelingDDQN':
        value_model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512, 128))
        replay_buffer = ReplayBuffer(max_size=args.max_buffer_size,
                                     batch_size=64,
                                     n_warmup_batches=args.n_warmup_batches)
        return QNetwork(value_model_fn,
                        value_optimizer_fn,
                        value_optimizer_lr,
                        replay_buffer,
                        True,
                        True,
                        training_strategy,
                        evaluation_strategy,
                        1,
                        1,
                        0.1,
                        args)
    elif model_name == 'DuelingDDQN-PER':
        value_model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512, 128))
        replay_buffer = PrioritizedReplayBuffer(max_samples=args.max_buffer_size, batch_size=64, rank_based=True,
                                                alpha=0.6, beta0=0.1, beta_rate=0.99995)
        return QNetwork(value_model_fn,
                        value_optimizer_fn,
                        value_optimizer_lr,
                        replay_buffer,
                        True,
                        True,
                        training_strategy,
                        evaluation_strategy,
                        1,
                        1,
                        0.01,
                        args)
    else:
        assert False, 'No such model name {}'.format(model_name)
