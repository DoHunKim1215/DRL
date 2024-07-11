from utils.plot_manager import plot_from_file

if __name__ == '__main__':
    res_dict = {
        'REINFORCE': 'results/REINFORCE/logs/env_CartPole-v1_model_REINFORCE.npy',
        'VPG': 'results/VPG/logs/env_CartPole-v1_model_VPG.npy',
    }
    plot_from_file(res_dict, 'results', 'plot_4')
