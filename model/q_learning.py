# %%
import numpy as np
import random
from collections import defaultdict
from ipywidgets import interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt

# %%
def def_map(height, width, obs_num):
    map = [[0] * width for _ in range(height)]
    
    for i in range(height):
        map[i][0] = 1
        map[i][width-1] = 1
    
    for i in range(width):
        map[0][i] = 1
        map[height-1][i] = 1
    
    obs = []
    for i in range(obs_num):
        x = random.randint(1, width-2)
        y = random.randint(1, height-2)
        if map[y][x] == 1:
            i -= 1
            continue
        map[y][x] = 1
        obs.append([y, x])
    return map, obs

def save_X(arr, name):
    """Save the X data to a file"""
    filename = name + '.txt'
    np.savetxt(filename, np.array(arr))

def read_X(name):
    """Read the X data from a file"""
    filename = name + '.txt'
    return np.loadtxt(filename)


# %%
maze_map, obs = def_map(20,20,60)
n_agents = 20
state_size = len(maze_map) * len(maze_map[0]) # Simplified state representation
MAZE_H, MAZE_W = len(maze_map), len(maze_map[0]) 
action_size = 4

# %%
'''sample maze
maze_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1], 
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1], 
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1], 
            [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1], 
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], 
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], 
            [1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1], 
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], 
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], 
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], 
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], 
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], 
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
goal = [(10, 3), (6, 2), (3, 15), (14, 6), (12, 11), (4, 10), (14, 3), (15, 6), (2, 10), (15, 2),
        (1, 16),(11, 9), (7, 6), (4, 7), (11, 8), (3, 18), (11, 3), (4, 11), (11, 17), (6, 10)]        
'''

# %%
class MultiAgentMaze:
    def __init__(self, maze_map, n_agents):
        self.maze_map = maze_map
        self.n_agents = n_agents
        self.agent_positions = [[0,0] for _ in range(n_agents)]

    def reset(self):
        for j in range(0,self.n_agents):
            xp,yp = 0,0
            for i in range(0, 1000000):
                xp = random.randint(0, len(maze_map)-1)
                yp = random.randint(0, len(maze_map[0])-1)
                if(maze_map[xp][yp] == 0):
                    break
            self.agent_positions[j] = [xp,yp]
        return self.agent_positions

    def step(self, agent, actions, dones):
        new_positions = []
        rewards = [0]*self.n_agents
        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]
            if dones[i] == 1:
                new_positions.append([x,y])
                continue
            if action == 0 and (y-1)>=0 :
                y -= 1
            elif action == 1 and (y+1)<len(self.maze_map[0]) :
                y += 1
            elif action == 2 and (x-1)>=0 :
                x -= 1
            elif action == 3 and (x+1)<len(self.maze_map) :
                x += 1
            rewards[i] -= 10
            for j in range(0,self.n_agents):
                if (x,y) == self.agent_positions[j] and i!=j:
                    rewards[i] -= 1000
                    dones[i] = 1
            if self.maze_map[x][y] == 1:
                #print('fall')
                rewards[i] -= 100000
                dones[i] = 1
            if (x, y) == agent[i].goal:
                rewards[i] += 10000
                dones[i] = 1
            new_positions.append([x, y])
            
        self.agent_positions = new_positions
        return new_positions, rewards, dones, {}
    
    def step_eval(self, agent, actions, dones):
        new_positions = []
        rewards = [0]*self.n_agents
        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]
            if dones[i] == 1:
                new_positions.append([x, y])
                continue
            if action == 0 and (y-1)>=0 :
                y -= 1
            elif action == 1 and (y+1)<len(self.maze_map[0]) :
                y += 1
            elif action == 2 and (x-1)>=0 :
                x -= 1
            elif action == 3 and (x+1)<len(self.maze_map) :
                x += 1
            if (x, y) == agent[i].goal:
                dones[i] = 1
            new_positions.append([x, y])
            
        self.agent_positions = new_positions
        return new_positions, rewards, dones, {}



# %%
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate, discount_factor, exploration_rate, exploration_min, exploration_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.done = 0
        xt,yt = 0,0
        for i in range(0,1000000):
            xt = random.randint(0, len(maze_map)-1)
            yt = random.randint(0, len(maze_map[0])-1)
            if(maze_map[xt][yt] == 0):
                break
        self.goal = (xt,yt)
        print((xt,yt))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0,self.action_size-1)
        else:
            return np.argmax(self.q_table[state])
        
    def obsandact(self, state):
        return np.argmax(self.q_table[state])
            
    def learn(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state][action] = new_q

    def epsilon_decay(self):
        if self.epsilon > self.exploration_min:
            self.epsilon *= self.exploration_decay


# %%
env = MultiAgentMaze(maze_map, n_agents)
env.reset()
pos = env.agent_positions
agents = [QLearningAgent(state_size, action_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=3, 
                         exploration_min=0.01, exploration_decay=0.9999) for i in range(n_agents)]

# %%
episodes = 50000
for episode in range(episodes):
    states = env.reset()
    states_encoded = [np.ravel_multi_index(pos, (len(maze_map), len(maze_map[0]))) for pos in states]  # Encode state
    dones = [0 for _ in range(n_agents)]
    while not all(dones):
        actions = [agent.act(state_encoded) for agent, state_encoded in zip(agents, states_encoded)]
        #print(actions)
        next_states, rewards, dones, _ = env.step(agents,actions,dones)
        next_states_encoded = [np.ravel_multi_index(pos, (len(maze_map), len(maze_map[0]))) for pos in next_states]  # Encode state
        for (agent, state_encoded, next_state_encoded, reward) in zip(agents, states_encoded, next_states_encoded, rewards):
            agent.learn(state_encoded, actions[states_encoded.index(state_encoded)], reward, next_state_encoded)
            #agent.q_deal(obs)
        states_encoded = next_states_encoded
        #print(next_states)
    agent.epsilon_decay()
    if episode % 1000 == 0:
        print("Episode:", episode, "Epsilon:", agent.epsilon)
print("Training finished")

# %%
pathall = []
#print(agents[0].q_table)
states = env.reset()
pathall.append(states)
states_encoded = [np.ravel_multi_index(pos, (len(maze_map), len(maze_map[0]))) for pos in states]  # Encode state
dones = [0 for _ in range(n_agents)]
while not all(dones):
    actions = [agent.obsandact(state_encoded) for agent, state_encoded in zip(agents, states_encoded)]
    next_states, rewards, dones, _ = env.step_eval(agents,actions,dones)
    pathall.append(next_states)
    #print(next_states)
    next_states_encoded = [np.ravel_multi_index(pos, (len(maze_map), len(maze_map[0]))) for pos in next_states]  # Encode state
    states_encoded = next_states_encoded
print(pathall)


# %%


# 假设maze_map和pathall已经定义，且n_agents已知

# 将迷宫地图转换为numpy数组
maze = np.array(maze_map)

def transpose_list_v2(lst):
    return [[row[i] for row in lst] for i in range(len(lst[0]))]

pathal = transpose_list_v2(pathall)

# 创建网格图
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks(np.arange(0.5, len(maze[0])+0.5), visible=False)
ax.set_yticks(np.arange(0.5, len(maze)+0.5), visible=False)

# 绘制迷宫
for i in range(len(maze)):
    for j in range(len(maze[0])):
        if maze[i, j] == 1:
            ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, edgecolor='black', facecolor='grey'))  # 障碍物为灰色
        else:
            ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, edgecolor='black', facecolor='white'))  # 空地为白色

# 移除轴
ax.axis('off')

# 使用不同的颜色为每个代理的路径着色
colors = plt.cm.viridis(np.linspace(0, 1, n_agents))  # 生成n_agents种颜色

for i in range(n_agents):
    path = pathal[i]
    color = colors[i]  # 指定每条路径的颜色
    
    # 绘制路径
    for j in range(len(path) - 1):
        ax.plot([path[j][0], path[j+1][0]], [path[j][1], path[j+1][1]], color=color, linewidth=2)

    # 在路径的起点和终点添加标记
    ax.scatter(path[0][0], path[0][1], color='red', marker='o')  # 起点为红色
    ax.scatter(path[-1][0], path[-1][1], color='green', marker='o')  # 终点为绿色

# 添加轴标签和标题
ax.set_title('Path Visualization with Different Colors')

# 显示图像
plt.show()

# %%
for i in range(len(pathal)):
    pathname = 'path' + str(i)
    save_X(pathal[i], pathname)
save_X(maze_map, 'map')

# %%
pathal = []
for i in range(n_agents):
    pathname = 'path' + str(i)
    path = read_X(pathname)
    pathal.append(path)
maze = read_X('map')

# %%
# 将迷宫地图转换为numpy数组
maze = np.array(maze_map)

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
        path = pathal[i]
        color = plt.cm.viridis(i / n_agents)
        
        for j in range(step_all + 1):
            ax.plot([path[j][0], path[j+1][0]], [path[j][1], path[j+1][1]], color=color, linewidth=3, alpha=0.5)
        ax.scatter(path[0][0], path[0][1], color='red', marker='o')  # 起点
        ax.scatter(path[min(step_all, len(path) - 1)][0], path[min(step_all, len(path) - 1)][1], color='green', marker='o')  # 当前或终点

    ax.axis('off')
    plt.tight_layout()

interact_manual(plot_path, agent_index=widgets.IntSlider(min=0, max=n_agents-1, step=1, value=0), 
               step_all=widgets.IntSlider(min=0, max=max([len(path) for path in pathal])-2, step=1, value=0))


