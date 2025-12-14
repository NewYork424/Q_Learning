import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==================== 环境定义 ====================
class MultiAgentMaze:
    """多智能体迷宫环境"""
    def __init__(self, maze_map, n_agents):
        self.maze_map = maze_map
        self.n_agents = n_agents
        self.height = len(maze_map)
        self.width = len(maze_map[0])
        self.agent_positions = [[0, 0] for _ in range(n_agents)]

    def reset(self):
        """随机初始化智能体位置"""
        for j in range(self.n_agents):
            for _ in range(10000):
                xp = random.randint(1, self.height - 2)
                yp = random.randint(1, self.width - 2)
                if self.maze_map[xp][yp] == 0:
                    # 确保不与其他智能体重叠
                    if [xp, yp] not in self.agent_positions[:j]:
                        self.agent_positions[j] = [xp, yp]
                        break
        return self.agent_positions

    def step(self, agents, actions, dones):
        """执行动作并返回新状态、奖励和完成标志"""
        new_positions = []
        rewards = [0] * self.n_agents
        
        for i, action in enumerate(actions):
            if dones[i] == 1:
                new_positions.append(self.agent_positions[i])
                rewards[i] = 0
                continue
                
            x, y = self.agent_positions[i]
            new_x, new_y = x, y
            
            # 执行动作
            if action == 0 and y > 0:  # 左
                new_y = y - 1
            elif action == 1 and y < self.width - 1:  # 右
                new_y = y + 1
            elif action == 2 and x > 0:  # 上
                new_x = x - 1
            elif action == 3 and x < self.height - 1:  # 下
                new_x = x + 1
            
            # 基础移动惩罚
            rewards[i] = -1
            
            # 检查墙壁碰撞
            if self.maze_map[new_x][new_y] == 1:
                rewards[i] = -50
                new_x, new_y = x, y  # 保持原位
            
            # 检查与其他智能体碰撞（预测位置）
            collision = False
            for j in range(self.n_agents):
                if i != j and not dones[j]:
                    if (new_x, new_y) == tuple(self.agent_positions[j]):
                        collision = True
                        break
            
            if collision:
                rewards[i] = -30
                new_x, new_y = x, y  # 保持原位
            
            # 到达目标
            if (new_x, new_y) == agents[i].goal:
                rewards[i] = 100
                dones[i] = 1
            # 距离奖励（引导向目标移动）
            else:
                old_dist = abs(x - agents[i].goal[0]) + abs(y - agents[i].goal[1])
                new_dist = abs(new_x - agents[i].goal[0]) + abs(new_y - agents[i].goal[1])
                if new_dist < old_dist:
                    rewards[i] += 2  # 靠近目标奖励
                elif new_dist > old_dist:
                    rewards[i] -= 2  # 远离目标惩罚
                
            new_positions.append([new_x, new_y])
        
        self.agent_positions = new_positions
        return new_positions, rewards, dones
    
    def get_state_representation(self, agent_idx):
        """获取智能体的状态表示（包含周围环境信息）"""
        x, y = self.agent_positions[agent_idx]
        state = []
        
        # 当前位置
        state.extend([x / self.height, y / self.width])
        
        # 目标位置相对坐标
        # state.extend([
        #     (agents[agent_idx].goal[0] - x) / self.height,
        #     (agents[agent_idx].goal[1] - y) / self.width
        # ])
        
        # 周围8个方向的障碍物信息
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width:
                state.append(self.maze_map[nx][ny])
            else:
                state.append(1)  # 边界视为墙
        
        return np.array(state, dtype=np.float32)

# ==================== 改进的DQN网络 ====================
class ImprovedDQN(nn.Module):
    """改进的DQN网络结构"""
    def __init__(self, state_dim, action_dim):
        super(ImprovedDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# ==================== 改进的DQN智能体 ====================
class DQNAgent:
    """改进的深度Q网络智能体"""
    def __init__(self, state_dim, action_size, maze_map, seed=0):
        self.state_dim = state_dim
        self.action_size = action_size
        torch.manual_seed(seed)
        
        # Q网络
        self.qnetwork_local = ImprovedDQN(state_dim, action_size)
        self.qnetwork_target = ImprovedDQN(state_dim, action_size)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
        # 优化器和经验回放
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.0005)
        self.memory = deque(maxlen=20000)
        
        # 改进的超参数
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        self.tau = 0.005
        self.batch_size = 128
        self.update_every = 4
        self.learn_step = 0
        
        # 随机生成目标
        self.goal = self._generate_goal(maze_map)
        
    def _generate_goal(self, maze_map):
        """生成随机目标位置"""
        for _ in range(10000):
            x = random.randint(1, len(maze_map) - 2)
            y = random.randint(1, len(maze_map[0]) - 2)
            if maze_map[x][y] == 0:
                return (x, y)
        return (1, 1)
    
    def act(self, state, training=True):
        """ε-贪婪策略选择动作"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        return np.argmax(action_values.cpu().numpy())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self):
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return
        
        self.learn_step += 1
        
        # 采样经验
        experiences = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.qnetwork_local(next_states).argmax(1, keepdim=True)
            next_q = self.qnetwork_target(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        current_q = self.qnetwork_local(states).gather(1, actions)
        
        # Huber损失（更稳定）
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()
        
        # 定期更新目标网络
        if self.learn_step % self.update_every == 0:
            self.soft_update()
    
    def soft_update(self):
        """软更新目标网络"""
        for target_param, local_param in zip(
            self.qnetwork_target.parameters(), 
            self.qnetwork_local.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ==================== Q-Learning智能体 ====================
class QLearningAgent:
    """Q-Learning智能体"""
    def __init__(self, state_size, action_size, maze_map, 
                 learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_min=0.01, 
                 exploration_decay=0.997):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = exploration_min
        self.epsilon_decay = exploration_decay
        
        # Q表
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # 随机生成目标
        self.goal = self._generate_goal(maze_map)
    
    def _generate_goal(self, maze_map):
        """生成随机目标位置"""
        for _ in range(10000):
            x = random.randint(1, len(maze_map) - 2)
            y = random.randint(1, len(maze_map[0]) - 2)
            if maze_map[x][y] == 0:
                return (x, y)
        return (1, 1)
    
    def act(self, state, training=True):
        """ε-贪婪策略选择动作"""
        if training and np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Q-Learning更新规则"""
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q * (1 - done) - current_q
        )
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ==================== 训练函数 ====================
def train_agents(agent_type, maze_map, n_agents, episodes=500, max_steps=150):
    """训练智能体并记录性能指标"""
    env = MultiAgentMaze(maze_map, n_agents)
    state_size = len(maze_map) * len(maze_map[0])
    action_size = 4
    state_dim = 10  # 2(位置) + 8(周围环境)
    
    # 创建智能体
    if agent_type == 'DQN':
        agents = [DQNAgent(state_dim, action_size, maze_map, seed=i) 
                  for i in range(n_agents)]
    else:
        agents = [QLearningAgent(state_size, action_size, maze_map) 
                  for i in range(n_agents)]
    
    # 性能指标
    episode_rewards = []
    episode_steps = []
    success_rate = []
    
    for episode in range(episodes):
        states = env.reset()
        
        if agent_type == 'DQN':
            states_encoded = [env.get_state_representation(i) for i in range(n_agents)]
        else:
            states_encoded = [
                np.ravel_multi_index(pos, (len(maze_map), len(maze_map[0]))) 
                for pos in states
            ]
        
        dones = [0] * n_agents
        total_reward = 0
        steps = 0
        
        while not all(dones) and steps < max_steps:
            actions = [agent.act(state) for agent, state in 
                      zip(agents, states_encoded)]
            next_states, rewards, dones = env.step(agents, actions, dones)
            
            if agent_type == 'DQN':
                next_states_encoded = [env.get_state_representation(i) 
                                      for i in range(n_agents)]
            else:
                next_states_encoded = [
                    np.ravel_multi_index(pos, (len(maze_map), len(maze_map[0]))) 
                    for pos in next_states
                ]
            
            # 学习
            for i, agent in enumerate(agents):
                if not dones[i] or rewards[i] > 0:  # 只在未完成或成功时学习
                    if agent_type == 'DQN':
                        agent.remember(states_encoded[i], actions[i], 
                                     rewards[i], next_states_encoded[i], dones[i])
                        agent.learn()
                    else:
                        agent.learn(states_encoded[i], actions[i], 
                                  rewards[i], next_states_encoded[i], dones[i])
            
            states_encoded = next_states_encoded
            total_reward += sum(rewards)
            steps += 1
        
        # 更新探索率
        for agent in agents:
            agent.decay_epsilon()
        
        # 记录指标
        episode_rewards.append(total_reward / n_agents)
        episode_steps.append(steps)
        success_rate.append(sum(dones) / n_agents)
        
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else episode_rewards[-1]
            avg_success = np.mean(success_rate[-50:]) if len(success_rate) >= 50 else success_rate[-1]
            print(f"{agent_type} - Episode {episode}/{episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Success Rate: {avg_success:.2%}, "
                  f"Epsilon: {agents[0].epsilon:.3f}")
    
    return agents, episode_rewards, episode_steps, success_rate

# ==================== 评估和可视化 ====================
def evaluate_and_visualize(agents, maze_map, n_agents, agent_type):
    """评估智能体并可视化路径"""
    env = MultiAgentMaze(maze_map, n_agents)
    states = env.reset()
    
    if agent_type == 'DQN':
        states_encoded = [env.get_state_representation(i) for i in range(n_agents)]
    else:
        states_encoded = [
            np.ravel_multi_index(pos, (len(maze_map), len(maze_map[0]))) 
            for pos in states
        ]
    
    dones = [0] * n_agents
    pathall = [states.copy()]
    
    while not all(dones) and len(pathall) < 150:
        actions = [agent.act(state, training=False) for agent, state in 
                  zip(agents, states_encoded)]
        next_states, _, dones = env.step(agents, actions, dones)
        pathall.append(next_states.copy())
        
        if agent_type == 'DQN':
            states_encoded = [env.get_state_representation(i) 
                            for i in range(n_agents)]
        else:
            states_encoded = [
                np.ravel_multi_index(pos, (len(maze_map), len(maze_map[0]))) 
                for pos in next_states
            ]
    
    # 可视化路径
    plot_paths(maze_map, pathall, n_agents, agents, agent_type)
    
    # 打印评估结果
    print(f"\n{agent_type} Evaluation:")
    print(f"  Steps taken: {len(pathall)}")
    print(f"  Agents reached goal: {sum(dones)}/{n_agents}")
    for i, agent in enumerate(agents):
        final_pos = pathall[-1][i]
        print(f"  Agent {i}: Start {pathall[0][i]} -> End {final_pos}, Goal {agent.goal}")
    
    return pathall

def plot_paths(maze_map, pathall, n_agents, agents, title):
    """绘制迷宫和智能体路径"""
    maze = np.array(maze_map)
    
    # 转置路径数据
    paths_by_agent = [[step[i] for step in pathall] for i in range(n_agents)]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制迷宫
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            color = 'grey' if maze[i, j] == 1 else 'white'
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, 
                edgecolor='black', facecolor=color, linewidth=0.5
            ))
    
    # 绘制路径
    colors = plt.cm.Set3(np.linspace(0, 1, n_agents))
    
    for i, path in enumerate(paths_by_agent):
        color = colors[i]
        
        # 绘制轨迹
        for j in range(len(path) - 1):
            ax.plot([path[j][1], path[j+1][1]], 
                   [path[j][0], path[j+1][0]], 
                   color=color, linewidth=2.5, alpha=0.6)
        
        # 标记起点
        ax.scatter(path[0][1], path[0][0], 
                  color='red', marker='o', s=150, zorder=5, 
                  edgecolors='black', linewidths=2,
                  label=f'Start {i+1}')
        
        # 标记终点
        ax.scatter(path[-1][1], path[-1][0], 
                  color='lime', marker='s', s=150, zorder=5,
                  edgecolors='black', linewidths=2,
                  label=f'End {i+1}')
        
        # 标记目标
        ax.scatter(agents[i].goal[1], agents[i].goal[0], 
                  color='gold', marker='*', s=300, zorder=5, 
                  edgecolors='black', linewidths=2,
                  label=f'Goal {i+1}')
    
    ax.set_xlim(-0.5, len(maze[0]) - 0.5)
    ax.set_ylim(-0.5, len(maze) - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(f'{title} - Path Visualization\n(Steps: {len(pathall)})', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    
    plt.tight_layout()
    plt.show()

def compare_performance(dqn_metrics, qlearning_metrics):
    """对比两种算法的性能"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 平滑处理
    window = 30
    
    def smooth(data, window_size):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # 奖励对比
    dqn_smooth_reward = smooth(dqn_metrics[0], window)
    ql_smooth_reward = smooth(qlearning_metrics[0], window)
    
    axes[0].plot(dqn_smooth_reward, label='DQN', linewidth=2, color='blue')
    axes[0].plot(ql_smooth_reward, label='Q-Learning', linewidth=2, color='orange')
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Average Reward', fontsize=12)
    axes[0].set_title('Reward Comparison (Smoothed)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 步数对比
    dqn_smooth_steps = smooth(dqn_metrics[1], window)
    ql_smooth_steps = smooth(qlearning_metrics[1], window)
    
    axes[1].plot(dqn_smooth_steps, label='DQN', linewidth=2, color='blue')
    axes[1].plot(ql_smooth_steps, label='Q-Learning', linewidth=2, color='orange')
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Steps per Episode', fontsize=12)
    axes[1].set_title('Steps Comparison (Smoothed)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # 成功率对比
    dqn_smooth_success = smooth(dqn_metrics[2], window)
    ql_smooth_success = smooth(qlearning_metrics[2], window)
    
    axes[2].plot(dqn_smooth_success, label='DQN', linewidth=2, color='blue')
    axes[2].plot(ql_smooth_success, label='Q-Learning', linewidth=2, color='orange')
    axes[2].set_xlabel('Episode', fontsize=12)
    axes[2].set_ylabel('Success Rate', fontsize=12)
    axes[2].set_title('Success Rate Comparison (Smoothed)', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.show()

# ==================== 主程序 ====================
def main():
    # 定义迷宫
    maze_map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    
    n_agents = 10  # 增加到3个智能体
    episodes = 200  # 增加训练轮数
    
    print("=" * 60)
    print("Training DQN Agents...")
    print("=" * 60)
    dqn_agents, dqn_rewards, dqn_steps, dqn_success = train_agents(
        'DQN', maze_map, n_agents, episodes
    )
    
    print("\n" + "=" * 60)
    print("Training Q-Learning Agents...")
    print("=" * 60)
    ql_agents, ql_rewards, ql_steps, ql_success = train_agents(
        'Q-Learning', maze_map, n_agents, episodes
    )
    
    print("\n" + "=" * 60)
    print("Evaluating and Visualizing Results...")
    print("=" * 60)
    
    # 评估和可视化
    evaluate_and_visualize(dqn_agents, maze_map, n_agents, 'DQN')
    evaluate_and_visualize(ql_agents, maze_map, n_agents, 'Q-Learning')
    
    # 性能对比
    compare_performance(
        (dqn_rewards, dqn_steps, dqn_success),
        (ql_rewards, ql_steps, ql_success)
    )
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print("=" * 60)
    print(f"DQN:")
    print(f"  Final Avg Reward (last 50): {np.mean(dqn_rewards[-50:]):.2f}")
    print(f"  Final Success Rate (last 50): {np.mean(dqn_success[-50:]):.2%}")
    print(f"  Best Success Rate: {max(dqn_success):.2%}")
    print(f"\nQ-Learning:")
    print(f"  Final Avg Reward (last 50): {np.mean(ql_rewards[-50:]):.2f}")
    print(f"  Final Success Rate (last 50): {np.mean(ql_success[-50:]):.2%}")
    print(f"  Best Success Rate: {max(ql_success):.2%}")
    print("=" * 60)

if __name__ == '__main__':
    main()