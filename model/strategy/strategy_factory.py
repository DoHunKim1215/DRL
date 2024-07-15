from typing import Callable

from model.strategy.exploration_strategy import ExplorationStrategy, GreedyStrategy, EGreedyStrategy, \
    LinearlyDecayingEGreedyStrategy, ExponentiallyDecayingEGreedyStrategy, SoftmaxStrategy


def get_strategy(name: str) -> Callable[[], ExplorationStrategy]:
    if name == 'greedy':
        return lambda: GreedyStrategy()
    elif name == 'epsilon_greedy':
        return lambda: EGreedyStrategy(epsilon=0.5)
    elif name == 'linearly_decaying_epsilon_greedy':
        return lambda: LinearlyDecayingEGreedyStrategy(init_epsilon=1.0,
                                                       min_epsilon=0.3,
                                                       decay_steps=20000)
    elif name == 'exponentially_decaying_epsilon_greedy':
        return lambda: ExponentiallyDecayingEGreedyStrategy(init_epsilon=1.0,
                                                            min_epsilon=0.3,
                                                            decay_steps=20000)
    elif name == 'softmax':
        return lambda: SoftmaxStrategy(init_temp=1.0,
                                       min_temp=0.1,
                                       decay_steps=20000)
    else:
        assert False, 'Invalid strategy name'
