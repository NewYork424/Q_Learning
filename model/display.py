# %%
import numpy as np
import random
from collections import defaultdict
from ipywidgets import interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt

# %%
def save_X(arr, name):
    """Save the X data to a file"""
    filename = name + '.txt'
    np.savetxt(filename, np.array(arr))
def read_X(name):
    """Read the X data from a file"""
    filename = name + '.txt'
    return np.loadtxt(filename)
n_agents = 20

# %%
pathl = []
for i in range(n_agents):
    pathname = 'path' + str(i)
    path = read_X(pathname)
    pathl.append(path)
maze_ = read_X('map')

# %%
# 将迷宫地图转换为numpy数组
maze = np.array(maze_)

def plot_path(agent_index=0, step_all=0):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(0.5, len(maze[0])+0.5), visible=False)
    ax.set_yticks(np.arange(0.5, len(maze)+0.5), visible=False)

    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i, j] == 1:
                ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, edgecolor='black', facecolor='grey'))
            else:
                ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, edgecolor='black', facecolor='white'))

    for i in range(n_agents):
        path = pathl[i]
        color = plt.cm.viridis(i / n_agents)
        
        for j in range(step_all + 1):
            ax.plot([path[j][0], path[j+1][0]], [path[j][1], path[j+1][1]], color=color, linewidth=3, alpha=0.5)
        ax.scatter(path[0][0], path[0][1], color='red', marker='o')  # 起点
        ax.scatter(path[min(step_all, len(path) - 1)][0], path[min(step_all, len(path) - 1)][1], color='green', marker='o')  # 当前或终点

    ax.axis('off')
    plt.tight_layout()

interact_manual(plot_path, agent_index=widgets.IntSlider(min=0, max=n_agents-1, step=1, value=0), 
               step_all=widgets.IntSlider(min=0, max=max([len(path) for path in pathl])-2, step=1, value=0))


