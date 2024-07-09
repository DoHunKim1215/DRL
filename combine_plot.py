from utils.plot_manager import plot_from_file

if __name__ == '__main__':
    res_dict = {
        'DQN + Exp. Decaying Epsilon Greedy(ep=1.0~0.3)': 'results/DQN/logs/env_CartPole-v1_model_DQN.npy',
        'DDQN + Exp. Decaying Epsilon Greedy(ep=1.0~0.3)': 'results/DDQN/logs/env_CartPole-v1_model_DDQN.npy',
    }
    plot_from_file(res_dict, 'results', 'plot_2')
