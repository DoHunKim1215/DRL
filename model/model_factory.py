import argparse

import torch

from model.model import RLModel
from model.policy.p_arch import FCDAP
from model.policy.p_model import PModel
from model.policy.v_arch import FCV
from model.value.arch import FCQ, FCDuelingQ
from model.value.exploration_strategy import get_strategy, GreedyStrategy
from model.value.q_model import QModel
from model.value.experience_buffer import ReplayBuffer, ExhaustingBuffer, PrioritizedReplayBuffer


def create_rl_model(model_name: str, args: argparse.Namespace) -> RLModel:
    optimizer_lr = args.lr
    training_strategy = get_strategy(args.strategy_name)
    evaluation_strategy = lambda: GreedyStrategy()

    if model_name == 'NFQ':
        model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512, 128))
        optimizer_fn = lambda net, lr: torch.optim.RMSprop(net.parameters(), lr=lr)
        replay_buffer = ExhaustingBuffer(batch_size=1024)
        return QModel(model_fn,
                      optimizer_fn,
                      optimizer_lr,
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
        model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512, 128))
        optimizer_fn = lambda net, lr: torch.optim.RMSprop(net.parameters(), lr=lr)
        replay_buffer = ReplayBuffer(max_size=args.max_buffer_size,
                                     batch_size=64,
                                     n_warmup_batches=args.n_warmup_batches)
        return QModel(model_fn,
                      optimizer_fn,
                      optimizer_lr,
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
        model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(512, 128))
        optimizer_fn = lambda net, lr: torch.optim.RMSprop(net.parameters(), lr=lr)
        replay_buffer = ReplayBuffer(max_size=args.max_buffer_size,
                                     batch_size=64,
                                     n_warmup_batches=args.n_warmup_batches)
        return QModel(model_fn,
                      optimizer_fn,
                      optimizer_lr,
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
        model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512, 128))
        optimizer_fn = lambda net, lr: torch.optim.RMSprop(net.parameters(), lr=lr)
        replay_buffer = ReplayBuffer(max_size=args.max_buffer_size,
                                     batch_size=64,
                                     n_warmup_batches=args.n_warmup_batches)
        return QModel(model_fn,
                      optimizer_fn,
                      optimizer_lr,
                      replay_buffer,
                      True,
                      True,
                      training_strategy,
                      evaluation_strategy,
                      1,
                      1,
                      0.1,
                      args)
    elif model_name == 'DuelingDDQN+PER':
        model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(512, 128))
        optimizer_fn = lambda net, lr: torch.optim.RMSprop(net.parameters(), lr=lr)
        replay_buffer = PrioritizedReplayBuffer(max_samples=args.max_buffer_size, batch_size=64, rank_based=True,
                                                alpha=0.6, beta0=0.1, beta_rate=0.99995)
        return QModel(model_fn,
                      optimizer_fn,
                      optimizer_lr,
                      replay_buffer,
                      True,
                      True,
                      training_strategy,
                      evaluation_strategy,
                      1,
                      1,
                      0.01,
                      args)
    elif model_name == 'REINFORCE':
        model_fn = lambda nS, nA: FCDAP(nS, nA, hidden_dims=(128, 64))
        optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        return PModel(model_fn, optimizer_fn, optimizer_lr, 0.0, args)
    elif model_name == 'VPG':
        policy_model_fn = lambda nS, nA: FCDAP(nS, nA, hidden_dims=(128, 64))
        policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        policy_optimizer_lr = args.policy_lr

        value_model_fn = lambda nS: FCV(nS, hidden_dims=(256, 128))
        value_optimizer_fn = lambda net, lr: torch.optim.RMSprop(net.parameters(), lr=lr)
        value_optimizer_lr = args.value_lr
        return PModel(policy_model_fn,
                      policy_optimizer_fn,
                      policy_optimizer_lr,
                      0.001,
                      args,
                      True,
                      value_model_fn,
                      value_optimizer_fn,
                      value_optimizer_lr)
    else:
        assert False, 'No such model name {}'.format(model_name)
