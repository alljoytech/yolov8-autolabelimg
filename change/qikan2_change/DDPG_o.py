import numpy as np
from copy import deepcopy
from torch.optim import Adam
import torch
import core as core
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, SpectralClustering,DBSCAN
import math
import torch.nn as nn
import torch.nn.functional as F
import random
import seaborn as sns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        attention_weights = self.softmax(self.attention(x))
        out = x * attention_weights
        return out


class SharedNetwork(nn.Module):
    def __init__(self, state_dim):
        super(SharedNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.attention1 = Attention(400)
        self.l2 = nn.Linear(400, 300)
        self.attention2 = Attention(300)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.attention1(x)
        x = F.relu(self.l2(x))
        x = self.attention2(x)
        return x

# Actor Network with Attention
class Actor(nn.Module):
    def __init__(self, shared_network, act_dim, act_limit,high_level_task_dim):
        super(Actor, self).__init__()
        self.shared_network = shared_network
        self.l3 = nn.Linear(300+ high_level_task_dim, act_dim)
        self.act_limit = act_limit

    def forward(self, obs, high_level_task):
        x = self.shared_network(obs)
        x = torch.cat([x, high_level_task], dim=-1)  # 合并高层次任务决策
        x = torch.tanh(self.l3(x))

        return (x+1)/2*(self.act_limit[1] - self.act_limit[0])+ self.act_limit[0]

class Critic(nn.Module):
    def __init__(self, shared_network, state_dim, action_dim,high_level_task_dim):
        super(Critic, self).__init__()
        self.shared_network = shared_network
        self.l3 = nn.Linear(300 + action_dim+high_level_task_dim, 300)
        self.l4 = nn.Linear(300, 1)

    def forward(self, obs, act,high_level_task):
        x = self.shared_network(obs)
        xu = torch.cat([x, act,high_level_task], dim=-1)
        q = F.relu(self.l3(xu))
        q = self.l4(q)
        return q

# Define the HighLevelPlanner module
class HighLevelPlanner(nn.Module):
    def __init__(self, obs_dim, task_dim, hidden_sizes=(128, 64), activation=nn.ReLU):
        super().__init__()
        self.l1 = nn.Linear(obs_dim + task_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], task_dim)

    def forward(self, obs, task):
        # Ensure task is 2D by adding an extra dimension if needed
        # Ensure task is 2D
        if obs.dim() == 3:
            task_expanded = task.unsqueeze(1).expand(-1, obs.size(1),-1)
            x = torch.cat([obs, task_expanded], dim=-1)
            # x = torch.cat([obs.unsqueeze(0), task.unsqueeze(0)], dim=-1)
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
        if obs.dim() == 1:
            obs = obs.expand(task.size(0), -1)
            # Concatenate along the last dimension
            x = torch.cat([obs, task], dim=-1)
            # x = torch.cat([obs.unsqueeze(0), task.unsqueeze(0)], dim=-1)
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
        if obs.dim() == 2:
            task_expanded = task.expand(obs.size(0), -1)
            x = torch.cat([obs, task_expanded], dim=-1)
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x))

# Define the MLPActorCritic module with hierarchical structure
class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit,
                 high_level_task_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        self.shared_network = SharedNetwork(obs_dim)
        self.high_level_task_dim = high_level_task_dim
        self.high_level_planner = HighLevelPlanner(obs_dim, high_level_task_dim)

        self.pi = Actor(self.shared_network, act_dim, act_limit, high_level_task_dim)
        self.q = Critic(self.shared_network, obs_dim, act_dim, high_level_task_dim)

    def act(self, obs):
        with torch.no_grad():
            high_level_task = self.high_level_planner(obs, torch.zeros(1, self.high_level_task_dim, device=device))  # Assuming task is a placeholder
            high_level_task  = high_level_task.squeeze()
            return self.pi(obs,high_level_task), high_level_task

    def forward(self, obs, act):
        high_level_task = self.high_level_planner(obs, torch.zeros(1, self.high_level_task_dim, device=device))  # Assuming task is a placeholder
        high_level_task  = high_level_task.squeeze()
        q_value = self.q(obs, act, high_level_task)
        return q_value


class ReplayBuffer:  # 采样后输出为tensor
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.k = 4

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
    def sample_batch_others(self, batch_size=32):
        """
         此版本采用聚类后抽样，保证每次样本的充足性
        """
        obs = self.obs_buf[0:self.size]
        obs2 = self.obs2_buf[0:self.size]
        act = self.act_buf[0:self.size]
        rew = self.rew_buf[0:self.size]
        done = self.done_buf[0:self.size]
        all_data = np.concatenate((obs,obs2,act,np.array(rew).reshape(-1, 1),np.array(done).reshape(-1, 1)),axis=1)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(all_data)
        np.savetxt('./data_csv_1/x_pca.csv', np.array(X_pca), delimiter=',')
        """可视化"""
        # 创建一个三维绘图对象
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2], c='b', marker='o')
        plt.show()
        ################################################################
        """抽样后的数据"""
        idxs = np.random.randint(0, self.size, size=batch_size)
        X_pca_1 = X_pca[idxs]
        np.savetxt('./data_csv_1/random_x.csv', np.array(X_pca_1), delimiter=',')
        """可视化"""
        # 创建一个三维绘图对象
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca_1[:, 0], X_pca_1[:, 1], X_pca_1[:, 2], c='b', marker='o')
        plt.show()
        ####################################################
        """K均值聚类"""
        k=5
        kmeans = KMeans(n_clusters=k,random_state=42) # 五类
        kmeans.fit(all_data)
        labels_k = kmeans.labels_
        centers_k = kmeans.cluster_centers_
        choose_num = []
        for i in range(k):
            element = i
            positions = np.array([index for index, value in enumerate(labels_k) if value == element])
            idxs_r = np.random.randint(0, len(positions), size=int(batch_size/k))
            choose_num.append(positions[idxs_r])
        X_pca_2 = X_pca[np.concatenate(choose_num)]
        np.savetxt('./data_csv_1/knn_x.csv', np.array(X_pca_2), delimiter=',')
        # 创建一个三维绘图对象
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca_2[:, 0], X_pca_2[:, 1], X_pca_2[:, 2], c='b', marker='o')
        plt.show()
        batch = dict(obs=self.obs_buf[X_pca_2],
                     obs2=self.obs2_buf[X_pca_2],
                     act=self.act_buf[X_pca_2],
                     rew=self.rew_buf[X_pca_2],
                     done=self.done_buf[X_pca_2])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

    def KNN_my(self):
        obs = self.obs_buf[0:self.size]
        obs2 = self.obs2_buf[0:self.size]
        act = self.act_buf[0:self.size]
        rew = self.rew_buf[0:self.size]
        done = self.done_buf[0:self.size]
        all_data = np.concatenate((obs, obs2, act, np.array(rew).reshape(-1, 1), np.array(done).reshape(-1, 1)), axis=1)
        """K均值聚类"""
        kmeans = KMeans(n_clusters=self.k, random_state=42)  # 五类
        kmeans.fit(all_data)
        labels_k = kmeans.labels_
        centers_k = kmeans.cluster_centers_
        size_k = self.size
        positive=[]
        for i in range(self.k):
            positive.append(np.ones(np.sum(labels_k==i),dtype=np.float32)/np.sum(labels_k == i))
        return labels_k,centers_k,size_k,positive

    def KNN_my_2(self,centers_k,size_k,labels_k):
        obs = self.obs_buf[size_k:self.size]
        obs2 = self.obs2_buf[size_k:self.size]
        act = self.act_buf[size_k:self.size]
        rew = self.rew_buf[size_k:self.size]
        done = self.done_buf[size_k:self.size]
        all_data = np.concatenate((obs, obs2, act, np.array(rew).reshape(-1, 1), np.array(done).reshape(-1, 1)), axis=1)
        """计算欧式距离"""
        distens = []
        for i in range(np.size(centers_k,0)):
            dis_i = np.linalg.norm(all_data-centers_k[i],axis = 1)
            distens.append(dis_i)
        min_indices = np.argmin(distens, axis=0)
        labels_k_2 = np.hstack((labels_k, min_indices))
        size_k = self.size
        return labels_k_2,size_k

    def KNN_my_3(self,centers_k,size_k,labels_k,positive):
        obs = self.obs_buf[size_k:self.size]
        obs2 = self.obs2_buf[size_k:self.size]
        act = self.act_buf[size_k:self.size]
        rew = self.rew_buf[size_k:self.size]
        done = self.done_buf[size_k:self.size]
        all_data = np.concatenate((obs, obs2, act, np.array(rew).reshape(-1, 1), np.array(done).reshape(-1, 1)), axis=1)
        """计算欧式距离"""
        distens = []
        for i in range(np.size(centers_k,0)):
            dis_i = np.linalg.norm(all_data-centers_k[i],axis = 1)
            distens.append(dis_i)
        min_indices = np.argmin(distens, axis=0)
        for i in range(np.size(centers_k,0)):
            index_p = np.where(min_indices == i)
            if len(index_p[0])==0:continue
            new_in = np.ones(len(index_p[0]),dtype=np.float32)/len(positive[i])
            positive[i] = np.hstack((positive[i], new_in))
            total = sum(positive[i])
            positive[i] = [p / total for p in positive[i]]
        labels_k_2 = np.hstack((labels_k, min_indices))
        size_k = self.size
        return labels_k_2,size_k,positive

    def noise_add(self,num):
        """add reward"""
        rew_1 = self.rew_buf[self.size-num :self.size]
        # 最大最小归一化
        min_val, max_val = min(rew_1), max(rew_1)
        normalized_data = [(x - min_val) / (max_val - min_val) for x in rew_1]
        R2 = -math.log(np.var(np.array(normalized_data)),10) # computer 方差 -ln(R)
        # print('R2 :%f' % R2)
        return R2

    def sample_batch_other_1(self,labels_k=[1], batch_size=32,positive=[]):
         choose_num = []
         for i in range(self.k):
             element = i
             positions = np.array([index for index, value in enumerate(labels_k) if value == element])
             idxs_r = np.random.randint(0, len(positions), size=int(batch_size / self.k))
             choose_num.append(positions[idxs_r])
         np.concatenate(choose_num)
         positive_1 = positive
         batch = dict(obs=self.obs_buf[choose_num],
                      obs2=self.obs2_buf[choose_num],
                      act=self.act_buf[choose_num],
                      rew=self.rew_buf[choose_num],
                      done=self.done_buf[choose_num])
         return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

    def sample_batch_other_2(self, labels_k=[1], batch_size=32,positive=[]):
        """
        此版本经验优先值计算 根据reward大小计算经验获取
        """
        choose_num = []
        kk=0
        for i in range(self.k):
            element = i
            positions = np.array([index for index, value in enumerate(labels_k) if value == element])
            result = random.choices(positions, weights=positive[i], k=int(batch_size / self.k))
            choose_num.append(result)
            # 减少抽样到数据的概率
            indic=[]
            for item in result:
                indices = np.argwhere(positions == item).flatten()
                indic.append(int(indices))
            posi = np.array(positive[i])
            posi[indic] *= 0.5   #采样后数据降低一半概率
            positive[i] = list(posi)
        np.concatenate(choose_num)
        choose_num = np.array(choose_num).reshape(1,-1)
        batch = dict(obs=self.obs_buf[choose_num],
                     obs2=self.obs2_buf[choose_num],
                     act=self.act_buf[choose_num],
                     rew=self.rew_buf[choose_num],
                     done=self.done_buf[choose_num])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()},positive






class DDPG:
    def __init__(self, obs_dim, act_dim, act_limit,high_level_task_dim, seed=0, replay_size=int(1e6),
                 gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, act_noise=0.1):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.high_level_task_dim = high_level_task_dim
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.ac = MLPActorCritic(obs_dim, act_dim, act_limit, high_level_task_dim).to(device)
        self.ac_targ = deepcopy(self.ac).to(device)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        high_level_task = self.ac.high_level_planner(o, torch.zeros(1, self.high_level_task_dim, device=device))
        q = self.ac.q(o, a,high_level_task)

        with torch.no_grad():
            high_level_task_targ = self.ac_targ.high_level_planner(o2, torch.zeros(1, self.high_level_task_dim,
                                                                                   device=device))
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2, high_level_task_targ), high_level_task_targ)
            # q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        loss_q = ((q - backup) ** 2).mean()
        return loss_q

    def compute_loss_pi(self, data):
        o = data['obs']
        high_level_task = self.ac.high_level_planner(o, torch.zeros(1, self.high_level_task_dim, device=device))
        q_pi = self.ac.q(o, self.ac.pi(o, high_level_task), high_level_task)
        # q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    def update(self, data):
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        for p in self.ac.q.parameters():
            p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        for p in self.ac.q.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, noise_scale):
        a, high_level_task = self.ac.act(torch.as_tensor(o, dtype=torch.float32, device=device))
        a = a.cpu().numpy()
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, self.act_limit[0], self.act_limit[1]), high_level_task

if __name__ == "__main__":
    obs_dim = 60
    act_dim = 30
    act_limit = [0.1,3]
    high_level_task_dim = 5

    aa = DDPG(obs_dim, act_dim, act_limit, high_level_task_dim)
    obs = torch.randn(1, obs_dim, device=device)  # Example observation
    act, high_level_task = aa.get_action(obs, noise_scale=0.1)







