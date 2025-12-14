import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
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
        self.initial_positions = []

    def reset(self):
        """重置环境，确保智能体不重叠"""
        self.agent_positions = []
        for j in range(self.n_agents):
            for _ in range(100000):
                x = random.randint(1, self.height - 2)
                y = random.randint(1, self.width - 2)
                if (self.maze_map[x][y] == 0 and 
                    [x, y] not in self.agent_positions):
                    self.agent_positions.append([x, y])
                    break
        self.initial_positions = [pos.copy() for pos in self.agent_positions]
        return self.agent_positions

    def step(self, agents, actions, dones):
        """执行动作，返回新状态、奖励和完成标志"""
        new_positions = []
        rewards = [0] * self.n_agents
        
        for i, action in enumerate(actions):
            if dones[i] == 1:
                new_positions.append(self.agent_positions[i])
                rewards[i] = 0
                continue
                
            x, y = self.agent_positions[i]
            new_x, new_y = x, y
            
            # 执行动作：0-左, 1-右, 2-上, 3-下
            if action == 0 and y > 0:
                new_y = y - 1
            elif action == 1 and y < self.width - 1:
                new_y = y + 1
            elif action == 2 and x > 0:
                new_x = x - 1
            elif action == 3 and x < self.height - 1:
                new_x = x + 1
            
            # 基础移动奖励
            rewards[i] = -1
            
            # 检查墙壁碰撞
            if self.maze_map[new_x][new_y] == 1:
                rewards[i] = -10
                new_x, new_y = x, y  # 保持原位
            else:
                # 检查与其他智能体碰撞
                collision = False
                for j in range(self.n_agents):
                    if (i != j and not dones[j] and 
                        [new_x, new_y] == self.agent_positions[j]):
                        collision = True
                        break
                
                if collision:
                    rewards[i] = -5
                    new_x, new_y = x, y  # 保持原位
            
            # 距离奖励（引导向目标移动）
            old_dist = abs(x - agents[i].goal[0]) + abs(y - agents[i].goal[1])
            new_dist = abs(new_x - agents[i].goal[0]) + abs(new_y - agents[i].goal[1])
            
            if new_dist < old_dist:
                rewards[i] += 3  # 靠近目标奖励
            elif new_dist > old_dist:
                rewards[i] -= 2  # 远离目标惩罚
            
            # 到达目标
            if (new_x, new_y) == agents[i].goal:
                rewards[i] = 100
                dones[i] = 1
            
            new_positions.append([new_x, new_y])
        
        self.agent_positions = new_positions
        return new_positions, rewards, dones

    def step_eval(self, agents, actions, dones):
        """评估模式的步进"""
        new_positions = []
        
        for i, action in enumerate(actions):
            if dones[i] == 1:
                new_positions.append(self.agent_positions[i])
                continue
                
            x, y = self.agent_positions[i]
            new_x, new_y = x, y
            
            if action == 0 and y > 0:
                new_y = y - 1
            elif action == 1 and y < self.width - 1:
                new_y = y + 1
            elif action == 2 and x > 0:
                new_x = x - 1
            elif action == 3 and x < self.height - 1:
                new_x = x + 1
            
            # 检查墙壁
            if self.maze_map[new_x][new_y] == 1:
                new_x, new_y = x, y
            
            # 检查目标
            if (new_x, new_y) == agents[i].goal:
                dones[i] = 1
            
            new_positions.append([new_x, new_y])
        
        self.agent_positions = new_positions
        return new_positions, [0] * self.n_agents, dones

    def get_state_representation(self, agent_idx):
        """获取富状态表示（位置 + 周围环境 + 目标方向）"""
        x, y = self.agent_positions[agent_idx]
        state = []
        
        # 归一化当前位置
        state.extend([x / self.height, y / self.width])
        
        # 周围8个方向的环境信息
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width:
                state.append(float(self.maze_map[nx][ny]))
            else:
                state.append(1.0)  # 边界视为墙
        
        return np.array(state, dtype=np.float32)

# ==================== 改进的DQN网络 ====================
class ImprovedDQN(nn.Module):
    """深度Q网络"""
    def __init__(self, state_dim, action_dim):
        super(ImprovedDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# ==================== DQN智能体 ====================
class DQNAgent:
    """DQN智能体"""
    def __init__(self, state_dim, action_size, maze_map, seed=0):
        self.state_dim = state_dim
        self.action_size = action_size
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Q网络
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.qnetwork_local = ImprovedDQN(state_dim, action_size).to(self.device)
        self.qnetwork_target = ImprovedDQN(state_dim, action_size).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.0005)
        
        # 经验回放
        self.memory = deque(maxlen=50000)
        
        # 超参数（关键优化点）
        self.gamma = 0.99           # 提高折扣因子，关注长期回报
        self.epsilon = 1.0          # 初始探索率
        self.epsilon_min = 0.01     # 最小探索率
        self.epsilon_decay = 0.995  # 探索衰减率
        self.tau = 0.005            # 软更新参数
        self.batch_size = 128       # 增大批次大小
        self.update_every = 4       # 每4步更新一次
        self.learn_step = 0
        
        # 随机生成目标
        self.goal = self._generate_goal(maze_map)
        print(f"Agent goal: {self.goal}")

    def _generate_goal(self, maze_map):
        """生成随机目标位置"""
        height, width = len(maze_map), len(maze_map[0])
        for _ in range(100000):
            x = random.randint(1, height - 2)
            y = random.randint(1, width - 2)
            if maze_map[x][y] == 0:
                return (x, y)
        return (1, 1)

    def act(self, state, training=True):
        """ε-贪婪策略选择动作"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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
        
        # 采样
        experiences = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.qnetwork_local(next_states).argmax(1, keepdim=True)
            next_q = self.qnetwork_target(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        current_q = self.qnetwork_local(states).gather(1, actions)
        
        # Huber损失
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()
        
        # 软更新目标网络
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

# ==================== 训练函数 ====================
def train_dqn(env, agents, episodes=2000, max_steps=200):
    """训练DQN智能体"""
    n_agents = len(agents)
    episode_rewards = []
    episode_steps = []
    success_rates = []
    
    for episode in range(episodes):
        states = env.reset()
        states_encoded = [env.get_state_representation(i) for i in range(n_agents)]
        dones = [0] * n_agents
        
        total_reward = 0
        steps = 0
        
        while not all(dones) and steps < max_steps:
            # 选择动作
            actions = [agent.act(state) for agent, state in 
                      zip(agents, states_encoded)]
            
            # 执行动作
            next_states, rewards, dones = env.step(agents, actions, dones)
            next_states_encoded = [env.get_state_representation(i) 
                                  for i in range(n_agents)]
            
            # 存储经验并学习
            for i, agent in enumerate(agents):
                if not dones[i] or rewards[i] > 0:
                    agent.remember(states_encoded[i], actions[i], 
                                 rewards[i], next_states_encoded[i], dones[i])
                    agent.learn()
            
            states_encoded = next_states_encoded
            total_reward += sum(rewards)
            steps += 1
        
        # 衰减探索率
        for agent in agents:
            agent.decay_epsilon()
        
        # 记录指标
        episode_rewards.append(total_reward / n_agents)
        episode_steps.append(steps)
        success_rates.append(sum(dones) / n_agents)
        
        # 输出进度
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else episode_rewards[-1]
            avg_success = np.mean(success_rates[-100:]) if len(success_rates) >= 100 else success_rates[-1]
            print(f"Episode {episode}/{episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Success Rate: {avg_success:.2%} | "
                  f"Epsilon: {agents[0].epsilon:.4f} | "
                  f"Steps: {steps}")
    
    print("Training finished!")
    return episode_rewards, episode_steps, success_rates

# ==================== 评估函数 ====================
def evaluate_dqn(env, agents, max_steps=200):
    """评估DQN智能体"""
    states = env.reset()
    states_encoded = [env.get_state_representation(i) for i in range(len(agents))]
    dones = [0] * len(agents)
    
    pathall = [states.copy()]
    steps = 0
    
    while not all(dones) and steps < max_steps:
        actions = [agent.act(state, training=False) for agent, state in 
                  zip(agents, states_encoded)]
        next_states, _, dones = env.step_eval(agents, actions, dones)
        pathall.append(next_states.copy())
        next_states_encoded = [env.get_state_representation(i) 
                              for i in range(len(agents))]
        states_encoded = next_states_encoded
        steps += 1
    
    print(f"\nEvaluation Results:")
    print(f"  Total steps: {steps}")
    print(f"  Agents reached goal: {sum(dones)}/{len(agents)}")
    for i, agent in enumerate(agents):
        final_pos = pathall[-1][i]
        print(f"  Agent {i}: Start {env.initial_positions[i]} -> "
              f"End {final_pos}, Goal {agent.goal}")
    
    return pathall

# ==================== 可视化函数 ====================
def plot_paths(maze_map, pathall, agents, title="DQN Path Visualization"):
    """绘制路径"""
    maze = np.array(maze_map)
    n_agents = len(agents)
    
    # 转置路径
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
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    for i, path in enumerate(paths_by_agent):
        color = colors[i]
        
        # 绘制轨迹
        for j in range(len(path) - 1):
            ax.plot([path[j][1], path[j+1][1]], 
                   [path[j][0], path[j+1][0]], 
                   color=color, linewidth=2.5, alpha=0.7)
        
        # 标记起点
        ax.scatter(path[0][1], path[0][0], 
                  color='red', marker='o', s=150, zorder=5,
                  edgecolors='black', linewidths=2)
        
        # 标记终点
        ax.scatter(path[-1][1], path[-1][0], 
                  color='lime', marker='s', s=150, zorder=5,
                  edgecolors='black', linewidths=2)
        
        # 标记目标
        ax.scatter(agents[i].goal[1], agents[i].goal[0], 
                  color='gold', marker='*', s=300, zorder=5,
                  edgecolors='black', linewidths=2)
    
    ax.set_xlim(-0.5, len(maze[0]) - 0.5)
    ax.set_ylim(-0.5, len(maze) - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(f'{title}\n(Steps: {len(pathall)}, Agents: {n_agents})', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_training_metrics(episode_rewards, episode_steps, success_rates):
    """绘制训练指标"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    window = 50
    
    def smooth(data, window_size):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # 奖励曲线
    smoothed = smooth(episode_rewards, window)
    axes[0].plot(smoothed, linewidth=2, color='blue')
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Average Reward', fontsize=12)
    axes[0].set_title(f'Training Reward (Smoothed)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 步数曲线
    smoothed = smooth(episode_steps, window)
    axes[1].plot(smoothed, linewidth=2, color='green')
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Steps per Episode', fontsize=12)
    axes[1].set_title(f'Steps per Episode (Smoothed)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # 成功率曲线
    smoothed = smooth(success_rates, window)
    axes[2].plot(smoothed, linewidth=2, color='orange')
    axes[2].set_xlabel('Episode', fontsize=12)
    axes[2].set_ylabel('Success Rate', fontsize=12)
    axes[2].set_title(f'Success Rate (Smoothed)', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 1.1])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==================== 主程序 ====================
def main():
    """主函数"""
    print("=" * 60)
    print("DQN Multi-Agent Path Planning")
    print("=" * 60)
    
    # 定义迷宫
    maze_map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    
    # 参数设置
    n_agents = 20
    state_dim = 10  # 2(位置) + 8(周围环境)
    action_size = 4
    episodes = 2000
    
    # 创建环境和智能体
    print(f"\nCreating environment with {n_agents} agents...")
    env = MultiAgentMaze(maze_map, n_agents)
    agents = [DQNAgent(state_dim, action_size, maze_map, seed=i) 
              for i in range(n_agents)]
    
    # 训练
    print("\n" + "=" * 60)
    print("Training agents...")
    print("=" * 60)
    episode_rewards, episode_steps, success_rates = train_dqn(
        env, agents, episodes=episodes
    )
    
    # 评估
    print("\n" + "=" * 60)
    print("Evaluating agents...")
    print("=" * 60)
    pathall = evaluate_dqn(env, agents)
    
    # 可视化
    print("\n" + "=" * 60)
    print("Visualizing results...")
    print("=" * 60)
    plot_paths(maze_map, pathall, agents)
    plot_training_metrics(episode_rewards, episode_steps, success_rates)
    
    # 统计信息
    print("\n" + "=" * 60)
    print("Training Summary:")
    print("=" * 60)
    print(f"Final average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final success rate (last 100): {np.mean(success_rates[-100:]):.2%}")
    print(f"Best success rate: {max(success_rates):.2%}")
    print("=" * 60)

if __name__ == '__main__':
    main()