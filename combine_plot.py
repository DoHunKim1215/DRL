from utils.plot_manager import plot_from_file

if __name__ == '__main__':
    res_dict = {
        'DDQN + Exp. Decaying Epsilon Greedy(ep=1.0~0.3)': 'results/DDQN/logs/env_CartPole-v1_model_DDQN.npy',
        'Dueling DDQN + Exp. Decaying Epsilon Greedy(ep=1.0~0.3)': 'results/DuelingDDQN/logs/env_CartPole-v1_model_DuelingDDQN.npy',
        'Dueling DDQN + PER + Exp. Decaying Epsilon Greedy(ep=1.0~0.3)': 'results/DuelingDDQN-PER/logs/env_CartPole-v1_model_DuelingDDQN-PER.npy',
    }
    plot_from_file(res_dict, 'results', 'plot_3')
