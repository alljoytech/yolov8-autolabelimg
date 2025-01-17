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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        """可视化"""
        # 创建一个三维绘图对象
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2], c='b', marker='o')
        plt.show()
        """抽样后的数据"""
        idxs = np.random.randint(0, self.size, size=batch_size)
        X_pca_1 = X_pca[idxs]
        """可视化"""
        # 创建一个三维绘图对象
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca_1[:, 0], X_pca_1[:, 1], X_pca_1[:, 2], c='b', marker='o')
        plt.show()
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
        return labels_k

    def noise_add(self,num):
        """add reward"""
        rew_1 = self.rew_buf[self.size-num :self.size]
        # 最大最小归一化
        min_val, max_val = min(rew_1), max(rew_1)
        normalized_data = [(x - min_val) / (max_val - min_val) for x in rew_1]
        R2 = -math.log(np.var(np.array(normalized_data)),10) # computer 方差 -ln(R)
        # print('R2 :%f' % R2)
        return R2

    def sample_batch_other_1(self,labels_k=[1], batch_size=32):
         choose_num = []
         for i in range(self.k):
             element = i
             positions = np.array([index for index, value in enumerate(labels_k) if value == element])
             idxs_r = np.random.randint(0, len(positions), size=int(batch_size / self.k))
             choose_num.append(positions[idxs_r])
         np.concatenate(choose_num)
         batch = dict(obs=self.obs_buf[choose_num],
                      obs2=self.obs2_buf[choose_num],
                      act=self.act_buf[choose_num],
                      rew=self.rew_buf[choose_num],
                      done=self.done_buf[choose_num])
         return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

    def sample_batch_other_2(self, batch_size=32):
        """
        此版本经验优先值计算 根据reward大小计算经验获取
        """
        obs = self.obs_buf[0:self.size]
        obs2 = self.obs2_buf[0:self.size]
        act = self.act_buf[0:self.size]
        rew = self.rew_buf[0:self.size] #reward
        done = self.done_buf[0:self.size]


        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}


class DDPG:
    def __init__(self, obs_dim, act_dim, act_bound, actor_critic=core.MLPActorCritic, seed=0,
                 replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, act_noise=0.1):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_bound = act_bound
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.ac = actor_critic(obs_dim, act_dim, act_limit=self.act_bound).to(device)
        self.ac_targ = deepcopy(self.ac).to(device)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def compute_loss_q(self, data):  # 返回(q网络loss, q网络输出的状态动作值即Q值)
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        return loss_q  # 这里的loss_q没加负号说明是最小化，很好理解，TD正是用函数逼近器去逼近backup，误差自然越小越好

    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()  # 这里的负号表明是最大化q_pi,即最大化在当前state策略做出的action的Q值

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, noise_scale):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32, device=device))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, self.act_bound[0], self.act_bound[1])
