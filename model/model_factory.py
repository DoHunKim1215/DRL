import argparse

import torch

from model.agent.async_ac_model import AsyncActorCriticModel
from model.agent.policy_model import PolicyModel
from model.agent.q_model import QModel
from model.agent.rl_model import RLModel
from model.agent.sync_ac_model import DeepDeterministicPolicyGradientModel, AdvantageActorCriticModel
from model.experience.experience_buffer import ExhaustingBuffer, ReplayBuffer, PrioritizedReplayBuffer
from model.multiprocess.shared_optimizer import SharedAdam, SharedRMSprop
from model.net.policy_net import FCDP, FCDAP, FCAC
from model.net.q_net import FCQV, FCQ, FCDuelingQ
from model.net.value_net import FCV
from model.strategy.exploration_strategy import GaussianNoiseStrategy, GreedyStrategy, BoundedGreedyStrategy
from model.strategy.strategy_factory import get_strategy


# Pickle can't serialize lambda function. So, For async model (e.g. A3C), they should be replaced into global function.
def get_fcdap(nS, nA):
    return FCDAP(nS, nA, hidden_dims=(128, 64))


def get_shared_adam(net, lr):
    return SharedAdam(net.parameters(), lr=lr)


def get_fcv(nS):
    return FCV(nS, hidden_dims=(256, 128))


def get_shared_rmsprop(net, lr):
    return SharedRMSprop(net.parameters(), lr=lr)


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
        return PolicyModel(model_fn, optimizer_fn, optimizer_lr, 0.0, args)
    elif model_name == 'VPG':
        policy_model_fn = lambda nS, nA: FCDAP(nS, nA, hidden_dims=(128, 64))
        policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        policy_optimizer_lr = args.policy_lr

        value_model_fn = lambda nS: FCV(nS, hidden_dims=(256, 128))
        value_optimizer_fn = lambda net, lr: torch.optim.RMSprop(net.parameters(), lr=lr)
        value_optimizer_lr = args.value_lr
        return PolicyModel(policy_model_fn,
                           policy_optimizer_fn,
                           policy_optimizer_lr,
                           0.001,
                           args,
                           True,
                           value_model_fn,
                           value_optimizer_fn,
                           value_optimizer_lr)
    elif model_name == 'A3C':
        policy_optimizer_lr = args.policy_lr
        value_optimizer_lr = args.value_lr
        return AsyncActorCriticModel(get_fcdap,
                                     get_shared_adam,
                                     policy_optimizer_lr,
                                     0.001,
                                     args,
                                     get_fcv,
                                     get_shared_rmsprop,
                                     value_optimizer_lr,
                                     50,
                                     8,
                                     1.0)
    elif model_name == 'GAE':
        policy_optimizer_lr = args.policy_lr
        value_optimizer_lr = args.value_lr
        return AsyncActorCriticModel(get_fcdap,
                                     get_shared_adam,
                                     policy_optimizer_lr,
                                     0.001,
                                     args,
                                     get_fcv,
                                     get_shared_rmsprop,
                                     value_optimizer_lr,
                                     50,
                                     8,
                                     0.99)
    elif model_name == 'A2C':
        ac_model_fn = lambda nS, nA: FCAC(nS, nA, hidden_dims=(256, 128))
        ac_optimizer_fn = lambda net, lr: torch.optim.RMSprop(net.parameters(), lr=lr)
        ac_optimizer_lr = args.actor_critic_lr
        return AdvantageActorCriticModel(ac_model_fn,
                                         ac_optimizer_fn,
                                         ac_optimizer_lr,
                                         1.0,
                                         0.6,
                                         0.001,
                                         10,
                                         8,
                                         0.95,
                                         args)
    elif model_name == 'DDPG':
        policy_model_fn = lambda nS, bounds, device: FCDP(nS, bounds, hidden_dims=(256, 256), device=device)
        policy_max_grad_norm = float('inf')
        policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        policy_optimizer_lr = 0.0003

        value_model_fn = lambda nS, nA: FCQV(nS, nA, hidden_dims=(256, 256))
        value_max_grad_norm = float('inf')
        value_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        value_optimizer_lr = 0.0003

        training_strategy_fn = lambda bounds: GaussianNoiseStrategy(bounds, exploration_noise_ratio=0.1)
        evaluation_strategy_fn = lambda bounds: BoundedGreedyStrategy(bounds)

        replay_buffer_fn = lambda: ReplayBuffer(max_size=100000, batch_size=256)
        return DeepDeterministicPolicyGradientModel(replay_buffer_fn,
                                                    policy_model_fn,
                                                    policy_optimizer_fn,
                                                    policy_optimizer_lr,
                                                    policy_max_grad_norm,
                                                    value_model_fn,
                                                    value_optimizer_fn,
                                                    value_optimizer_lr,
                                                    value_max_grad_norm,
                                                    training_strategy_fn,
                                                    evaluation_strategy_fn,
                                                    args.n_warmup_batches,
                                                    1,
                                                    0.005,
                                                    args)
    else:
        assert False, 'No such model name {}'.format(model_name)
