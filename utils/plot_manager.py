import os
from typing import List, Dict

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


def plot_result(results: List, label: str, fig_out_dir: str, fig_name: str):
    results = np.array(results)
    max_t, max_r, max_s, max_sec, max_rt = np.max(results, axis=0).T
    min_t, min_r, min_s, min_sec, min_rt = np.min(results, axis=0).T
    mean_t, mean_r, mean_s, mean_sec, mean_rt = np.mean(results, axis=0).T
    x = np.arange(len(mean_s))

    fig, axs = plt.subplots(5, 1, figsize=(10, 11), sharey='none', sharex='all')

    axs[0].plot(max_r, 'y', linewidth=1)
    axs[0].plot(min_r, 'y', linewidth=1)
    axs[0].plot(mean_r, 'y', label=label, linewidth=2)
    axs[0].fill_between(x, min_r, max_r, facecolor='y', alpha=0.3)

    axs[1].plot(max_s, 'y', linewidth=1)
    axs[1].plot(min_s, 'y', linewidth=1)
    axs[1].plot(mean_s, 'y', label=label, linewidth=2)
    axs[1].fill_between(x, min_s, max_s, facecolor='y', alpha=0.3)

    axs[2].plot(max_t, 'y', linewidth=1)
    axs[2].plot(min_t, 'y', linewidth=1)
    axs[2].plot(mean_t, 'y', label=label, linewidth=2)
    axs[2].fill_between(x, min_t, max_t, facecolor='y', alpha=0.3)

    axs[3].plot(max_sec, 'y', linewidth=1)
    axs[3].plot(min_sec, 'y', linewidth=1)
    axs[3].plot(mean_sec, 'y', label=label, linewidth=2)
    axs[3].fill_between(x, min_sec, max_sec, facecolor='y', alpha=0.3)

    axs[4].plot(max_rt, 'y', linewidth=1)
    axs[4].plot(min_rt, 'y', linewidth=1)
    axs[4].plot(mean_rt, 'y', label=label, linewidth=2)
    axs[4].fill_between(x, min_rt, max_rt, facecolor='y', alpha=0.3)

    axs[0].set_title('Moving Avg Reward (Training)')
    axs[1].set_title('Moving Avg Reward (Evaluation)')
    axs[2].set_title('Cumulated Steps')
    axs[3].set_title('Training Time')
    axs[4].set_title('Wall-clock Time')
    plt.xlabel('Episodes')
    axs[0].legend(loc='upper left')
    plt.savefig(os.path.join(fig_out_dir, '{}_model_{}.png'.format(fig_name, label)), dpi=300, format='png')
    plt.show()


def plot_from_file(file_names: Dict[str, str], fig_out_dir: str, fig_name: str):
    colors = ['g', 'b', 'y', 'c', 'r', 'k']

    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharey='none', sharex='all')

    for idx, (k, v) in enumerate(file_names.items()):
        results = np.load(v)
        max_t, max_r, max_s, max_sec, max_rt = np.max(results, axis=0).T
        min_t, min_r, min_s, min_sec, min_rt = np.min(results, axis=0).T
        mean_t, mean_r, mean_s, mean_sec, mean_rt = np.mean(results, axis=0).T
        x = np.arange(len(mean_s))

        color = colors[idx % len(colors)]
        axs[0].plot(max_r, color, linewidth=1)
        axs[0].plot(min_r, color, linewidth=1)
        axs[0].plot(mean_r, color, label=k, linewidth=2)
        axs[0].fill_between(x, min_r, max_r, facecolor=color, alpha=0.3)

        axs[1].plot(max_s, color, linewidth=1)
        axs[1].plot(min_s, color, linewidth=1)
        axs[1].plot(mean_s, color, label=k, linewidth=2)
        axs[1].fill_between(x, min_s, max_s, facecolor=color, alpha=0.3)

        axs[2].plot(max_t, color, linewidth=1)
        axs[2].plot(min_t, color, linewidth=1)
        axs[2].plot(mean_t, color, label=k, linewidth=2)
        axs[2].fill_between(x, min_t, max_t, facecolor=color, alpha=0.3)

        axs[3].plot(max_sec, color, linewidth=1)
        axs[3].plot(min_sec, color, linewidth=1)
        axs[3].plot(mean_sec, color, label=k, linewidth=2)
        axs[3].fill_between(x, min_sec, max_sec, facecolor=color, alpha=0.3)

        axs[4].plot(max_rt, color, linewidth=1)
        axs[4].plot(min_rt, color, linewidth=1)
        axs[4].plot(mean_rt, color, label=k, linewidth=2)
        axs[4].fill_between(x, min_rt, max_rt, facecolor=color, alpha=0.3)

    axs[0].set_title('Moving Avg Reward (Training)')
    axs[1].set_title('Moving Avg Reward (Evaluation)')
    axs[2].set_title('Cumulated Steps')
    axs[3].set_title('Training Time')
    axs[4].set_title('Wall-clock Time')
    plt.xlabel('Episodes')
    axs[0].legend(loc='upper left')
    plt.savefig(os.path.join(fig_out_dir, '{}_combined.png'.format(fig_name)), dpi=300, format='png')
    plt.show()
