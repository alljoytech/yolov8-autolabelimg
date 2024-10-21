import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义环境
class SimpleDroneDeliveryEnv:
    def __init__(self, num_drones, task_points, priorities):
        self.num_drones = num_drones
        self.task_points = task_points
        self.priorities = priorities  # 新增的任务优先级
        self.obs_dim = 3  # 每个无人机的状态维度 (x, y, 电量)
        self.act_dim = 2  # 每个无人机的动作维度 (dx, dy)
        self.state = np.random.rand(num_drones, self.obs_dim) * 30  # 初始化无人机状态，范围在 [0, 30] 之间
        self.done = [False] * num_drones
        self.trajectories = [[] for _ in range(num_drones)]  # 记录每个无人机的轨迹

    def reset(self):
        self.state = np.random.rand(self.num_drones, self.obs_dim) * 30
        self.done = [False] * self.num_drones
        self.trajectories = [[] for _ in range(self.num_drones)]
        return self.state

    def step(self, actions):
        next_state = self.state.copy()
        rewards = np.zeros(self.num_drones)

        for i in range(self.num_drones):
            if not self.done[i]:
                next_state[i, :2] += actions[i]  # 更新位置
                next_state[i, 2] -= 0.1  # 消耗电量

                # 确保无人机位置在立方体内部
                next_state[i, :2] = np.clip(next_state[i, :2], 0, 30)

                # 记录轨迹
                self.trajectories[i].append(next_state[i, :2].copy())

                distance_to_task = np.linalg.norm(next_state[i, :2] - self.task_points[i])
                rewards[i] = -distance_to_task

                if distance_to_task < 0.1:  # 任务完成
                    rewards[i] += 10 * self.priorities[i]  # 根据任务优先级调整奖励
                    self.done[i] = True

                if next_state[i, 2] <= 0:  # 电量耗尽
                    rewards[i] -= 10
                    self.done[i] = True

        return next_state, rewards, all(self.done)


# 自适应任务分配策略，考虑优先级和合作因素
def adaptive_task_allocation(env, drone_states):
    for i in range(env.num_drones):
        if not env.done[i]:
            # 基于电量、距离和优先级的加权任务分配策略
            weights = []
            for j, task in enumerate(env.task_points):
                distance = np.linalg.norm(drone_states[i, :2] - task)
                battery_factor = drone_states[i, 2]
                priority_factor = env.priorities[j]
                weight = (distance / battery_factor) / priority_factor
                weights.append(weight)

            # 分配到加权值最小的任务点
            min_weight_idx = np.argmin(weights)
            env.task_points[i] = env.task_points[min_weight_idx]

            # 合作机制：如果一个无人机任务过重或电量不足，可以请求其他无人机协助
            if weights[min_weight_idx] > 5:  # 假设5是一个负载过重的阈值
                for j in range(env.num_drones):
                    if j != i and not env.done[j]:
                        other_distance = np.linalg.norm(drone_states[j, :2] - env.task_points[min_weight_idx])
                        if other_distance < distance and drone_states[j, 2] > drone_states[i, 2]:
                            # 交换任务点
                            env.task_points[i], env.task_points[j] = env.task_points[j], env.task_points[i]


# 经验回放池
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, num_agents):
        self.obs_buf = np.zeros((size, num_agents, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, num_agents, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, num_agents, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, num_agents), dtype=np.float32)
        self.done_buf = np.zeros((size, num_agents), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}


# Actor 和 Critic 网络
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * num_agents + act_dim * num_agents, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, acts):
        combined = torch.cat([obs.view(obs.size(0), -1), acts.view(acts.size(0), -1)], dim=-1)
        return self.net(combined)


# MADDPG 类
class MADDPG:
    def __init__(self, obs_dim, act_dim, act_limit, num_agents, gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3):
        self.num_agents = num_agents
        self.gamma = gamma
        self.polyak = polyak

        self.actors = [Actor(obs_dim, act_dim, act_limit).to(device) for _ in range(num_agents)]
        self.critics = [Critic(obs_dim, act_dim, num_agents).to(device) for _ in range(num_agents)]

        self.target_actors = deepcopy(self.actors)
        self.target_critics = deepcopy(self.critics)

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=pi_lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=q_lr) for critic in self.critics]

    def get_action(self, obs, noise_scale):
        actions = []
        for i in range(self.num_agents):
            obs_i = obs[i].to(device)
            act = self.actors[i](obs_i).detach().cpu().numpy()
            act += noise_scale * np.random.randn(act.size)
            actions.append(np.clip(act, -1, 1))
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        return actions

    def update(self, batch):
        for i in range(self.num_agents):
            obs, act, rew, next_obs, done = batch['obs'], batch['act'], batch['rew'], batch['next_obs'], batch['done']

            with torch.no_grad():
                next_act = torch.cat([self.target_actors[j](next_obs[:, j, :]) for j in range(self.num_agents)], dim=-1)
                target_q = rew[:, i].unsqueeze(-1) + self.gamma * (1 - done[:, i].unsqueeze(-1)) * self.target_critics[
                    i](next_obs, next_act)

            current_q = self.critics[i](obs, act)
            critic_loss = nn.MSELoss()(current_q, target_q)
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            for p in self.critics[i].parameters():
                p.requires_grad = False

            pi_loss = -self.critics[i](obs, torch.cat([self.actors[j](obs[:, j, :]) for j in range(self.num_agents)],
                                                      dim=-1)).mean()
            self.actor_optimizers[i].zero_grad()
            pi_loss.backward()
            self.actor_optimizers[i].step()

            for p in self.critics[i].parameters():
                p.requires_grad = True

            with torch.no_grad():
                for p, p_targ in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

                for p, p_targ in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)


# 初始化环境和MADDPG模型
num_drones = 2
task_points = np.random.rand(num_drones, 2) * 5  # 任务点
priorities = np.random.rand(num_drones) * 10 + 1  # 随机分配的任务优先级
env = SimpleDroneDeliveryEnv(num_drones, task_points, priorities)
maddpg = MADDPG(env.obs_dim, env.act_dim, act_limit=1.0, num_agents=num_drones)

# 经验回放池
replay_buffer = ReplayBuffer(obs_dim=env.obs_dim, act_dim=env.act_dim, size=100000, num_agents=num_drones)

# 训练过程
num_episodes = 100
batch_size = 64

for episode in range(num_episodes):
    obs = env.reset()
    print(episode)
    done = False
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        actions = maddpg.get_action(obs_tensor, noise_scale=0.1).cpu().numpy()

        next_obs, rewards, done = env.step(actions)

        # 自适应任务分配策略
        adaptive_task_allocation(env, next_obs)

        # 存储经验
        replay_buffer.store(obs, actions, rewards, next_obs, done)
        obs = next_obs

        # 更新网络
        if replay_buffer.size >= batch_size:
            batch = replay_buffer.sample_batch(batch_size)
            maddpg.update(batch)

    print(f"Episode {episode + 1} completed")

# 绘制无人机轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_zlim(0, 30)

colors = ['r', 'g', 'b']
for i in range(num_drones):
    trajectory = np.array(env.trajectories[i])
    ax.plot(trajectory[:, 0], trajectory[:, 1], np.linspace(30, 0, len(trajectory)), c=colors[i],
            label=f'Drone {i + 1}')

# 绘制任务点
for i, task_point in enumerate(task_points):
    ax.scatter(task_point[0], task_point[1], 0, c=colors[i], marker='x', s=100)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.legend()
plt.show()
