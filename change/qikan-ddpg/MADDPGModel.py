import numpy as np
from copy import deepcopy
import torch.optim as optim
import torch
import torch.nn as nn
import core as core
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, SpectralClustering,DBSCAN
import math
import seaborn as sns
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, obs_dim, act_dim, act_bound, num_agents, actor_critic=core.MLPActorCritic, seed=0,
                 replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, act_noise=0.1):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_bound = act_bound
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create actor-critic and target networks for each agent
        self.ac = [actor_critic(obs_dim, act_dim, act_limit=self.act_bound).to(device) for _ in range(num_agents)]
        self.ac_targ = [deepcopy(ac).to(device) for ac in self.ac]

        self.pi_optimizers = [optim.Adam(ac.pi.parameters(), lr=pi_lr) for ac in self.ac]
        self.q_optimizers = [optim.Adam(ac.q.parameters(), lr=q_lr) for ac in self.ac]

        for i in range(num_agents):
            for p in self.ac_targ[i].parameters():
                p.requires_grad = False

        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, num_agents=num_agents)
        self.update_num = 0

    def compute_loss_q(self, data, agent_idx):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.ac[agent_idx].q(torch.cat([o[:, agent_idx, :], a.view(o.size(0), -1)], dim=-1))

        with torch.no_grad():
            # Get target actions from all agents at next state
            target_actions = [self.ac_targ[i].pi(o2[:, i, :]) for i in range(self.num_agents)]
            target_actions = torch.cat(target_actions, dim=-1)
            q_pi_targ = self.ac_targ[agent_idx].q(torch.cat([o2[:, agent_idx, :], target_actions], dim=-1))
            backup = r[:, agent_idx].unsqueeze(1) + self.gamma * (1 - d[:, agent_idx].unsqueeze(1)) * q_pi_targ

        loss_q = ((q - backup) ** 2).mean()
        return loss_q

    def compute_loss_pi(self, data, agent_idx):
        o = data['obs']
        pi = self.ac[agent_idx].pi(o[:, agent_idx, :])
        actions = [pi if i == agent_idx else self.ac[i].pi(o[:, i, :]).detach() for i in range(self.num_agents)]
        actions = torch.cat(actions, dim=-1)
        q_pi = self.ac[agent_idx].q(torch.cat([o[:, agent_idx, :], actions], dim=-1))
        return -q_pi.mean()

    def update(self, data):
        for agent_idx in range(self.num_agents):
            # Update Q-network
            self.q_optimizers[agent_idx].zero_grad()
            loss_q = self.compute_loss_q(data, agent_idx)
            loss_q.backward()
            self.q_optimizers[agent_idx].step()

            # Freeze Q-network so you don't waste computational effort
            for p in self.ac[agent_idx].q.parameters():
                p.requires_grad = False

            # Update policy network
            self.pi_optimizers[agent_idx].zero_grad()
            loss_pi = self.compute_loss_pi(data, agent_idx)
            loss_pi.backward()
            self.pi_optimizers[agent_idx].step()

            # Unfreeze Q-network
            for p in self.ac[agent_idx].q.parameters():
                p.requires_grad = True

            # Update target networks by polyak averaging
            with torch.no_grad():
                for p, p_targ in zip(self.ac[agent_idx].parameters(), self.ac_targ[agent_idx].parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, obs, noise_scale):
        actions = []
        for agent_idx in range(self.num_agents):
            a = self.ac[agent_idx].act(torch.as_tensor(obs[agent_idx], dtype=torch.float32, device=device))
            a += noise_scale * np.random.randn(self.act_dim)
            actions.append(np.clip(a, -self.act_bound, self.act_bound))
        return np.array(actions)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, num_agents):
        self.num_agents = num_agents
        self.obs_buf = np.zeros((size, num_agents, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, num_agents, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, num_agents, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, num_agents), dtype=np.float32)
        self.done_buf = np.zeros((size, num_agents), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}
if __name__ == '__main__':
    obs_dim = 10  # 假设每个智能体的状态维度为10
    act_dim = 2  # 假设每个智能体的动作维度为2
    act_bound = 1.0  # 动作范围为[-1, 1]
    num_agents = 5  # 假设有5个智能体

    maddpg = MADDPG(obs_dim, act_dim, act_bound, num_agents)
    a = maddpg
    print(a)