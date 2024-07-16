from utils.plot_manager import plot_from_file

if __name__ == '__main__':
    res_dict = {
        'DDPG': 'results/DDPG/logs/env_Pendulum-v1_model_DDPG.npy',
        'TD3': 'results/TD3/logs/env_Pendulum-v1_model_TD3.npy'
    }
    plot_from_file(res_dict, 'results', 'plot_5')
