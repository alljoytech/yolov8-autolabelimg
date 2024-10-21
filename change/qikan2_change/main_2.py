from DDPG_o import DDPG
from env_static_dy import limition
from method import getReward, transformAction, setup_seed
import random
import numpy as np
import matplotlib.pyplot as py
import os
import torch
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    setup_seed(11)  # 设置随机数种子
    limition = limition()
    obs_dim = 6 * (limition.numberOfSphere + limition.numberOfCylinder + limition.numberOfCone) + 9
    act_dim = 1 * (limition.numberOfSphere + limition.numberOfCylinder + limition.numberOfCone) + 3
    act_bound = [0.1, 3]
    act_dim_dy = 3
    actionBound = [[0.1, 3], [0.1, 3], [0.1, 3]] # 连续空间
    high_level_task_dim = 5 # 高层决策
    dynamicController = DDPG(obs_dim, act_dim, act_bound,high_level_task_dim)
    sumyes = 0
    MAX_EPISODE = 300
    MAX_STEP = 1500
    update_every = 50
    batch_size = 128
    noise = 0.3
    update_cnt = 0
    rewardList = []
    reward_all = []
    noise_all = []
    dissum = []
    maxReward = -np.inf
    minReward = np.inf
    z=0
    for episode in range(MAX_EPISODE):
        q = limition.x0
        limition.reset()
        rewardSum = 0
        qBefore = [None, None, None]
        for j in range(MAX_STEP):
            dic = limition.updateObs()
            vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
            obsDicq = limition.calculateDynamicState(q)
            obs_sphere, obs_cylinder, obs_cone = obsDicq['sphere'], obsDicq['cylinder'], obsDicq['cone']
            obs_mix = obs_sphere + obs_cylinder + obs_cone
            obs_dy = limition.calDynamicState_dy(q,obsCenter)
            obs_sta = np.array([]) # 中心控制器接受所有状态集合
            for k in range(len(obs_mix)):
                obs_sta = np.hstack((obs_sta, obs_mix[k])) # 拼接状态为一个1*n向量
            obs = np.hstack((obs_sta,obs_dy))
            if episode > 50:
                noise *= 0.99995
                if noise <= 0.1: noise = 0.1
                action,high_task = dynamicController.get_action(obs, noise_scale=noise)
                # 分解动作向量
                action_sphere = action[0:limition.numberOfSphere]
                action_cylinder = action[limition.numberOfSphere:limition.numberOfSphere + limition.numberOfCylinder]
                action_cone = action[limition.numberOfSphere + limition.numberOfCylinder:limition.numberOfSphere + \
                                                                                         limition.numberOfCylinder + limition.numberOfCone]
                action_dym = action[limition.numberOfSphere +limition.numberOfCylinder + limition.numberOfCone:]
            else:
                action_sphere = [random.uniform(act_bound[0], act_bound[1]) for k in range(limition.numberOfSphere)]
                action_cylinder = [random.uniform(act_bound[0], act_bound[1]) for k in range(limition.numberOfCylinder)]
                action_cone = [random.uniform(act_bound[0], act_bound[1]) for k in range(limition.numberOfCone)]
                action_dym = [random.uniform(-1, 1) for i in range(act_dim_dy)]
                action = action_sphere + action_cylinder + action_cone + action_dym

            # 与环境交互
            # if episode ==51:
            #     actionAfter = transformAction(action_dym, actionBound, act_dim_dy)
            actionAfter = transformAction(action_dym, actionBound, act_dim_dy)
            qNext = limition.getqNext_sta_dy_change(limition.epsilon0, action_sphere, action_cylinder, action_cone, q, qBefore,
                                             obsCenter, vObs, actionAfter[0], actionAfter[1], actionAfter[2])
            obsDicqNext = limition.calculateDynamicState(qNext)
            obs_sphere_next, obs_cylinder_next, obs_cone_next = obsDicqNext['sphere'], obsDicqNext['cylinder'], \
                                                                obsDicqNext['cone']
            obs_mix_next = obs_sphere_next + obs_cylinder_next + obs_cone_next
            obs_sta = np.array([])
            for k in range(len(obs_mix_next)):
                obs_sta = np.hstack((obs_sta, obs_mix_next[k]))
            obs_next_dy = limition.calDynamicState_dy(qNext, obsCenterNext)
            obs_next = np.hstack((obs_sta, obs_next_dy))
            flag = limition.checkCollision(qNext,obsCenterNext)
            reward = getReward(flag, limition, qBefore, q, qNext, obsCenterNext)
            rewardSum += reward
            done = True if limition.distanceCost(limition.qgoal, qNext) < limition.threshold else False
            dynamicController.replay_buffer.store(obs, action, reward, obs_next, done)
            if episode >= 30 and j % update_every == 0:
                if dynamicController.replay_buffer.size >= batch_size:
                    update_cnt += update_every
                    if z == 1:
                        labels_k, size_k,positive_choose = dynamicController.replay_buffer.KNN_my_3(centers_k, size_k, labels_k,positive_choose)
                    if z == 0:
                        labels_k, centers_k, size_k,positive_choose = dynamicController.replay_buffer.KNN_my()
                        z += 1
                    for _ in range(update_every):
                        batch,positive_choose = dynamicController.replay_buffer.sample_batch_other_2(labels_k, batch_size,positive_choose)
                        dynamicController.update(data=batch)
            if done: break
            qBefore = q
            q = qNext
        reward_all.append(rewardSum)
        # 选择测试其他环境 此处先忽略 查看此环境是否可以
        print('Episode:', episode, 'Reward:%f' % rewardSum, 'noise:%f' % noise, 'update_cnt:%d' % update_cnt)
        if episode > MAX_EPISODE * 2 / 3:
            if rewardSum > maxReward:
                print('reward大于历史最优，已保存模型！')
                maxReward = rewardSum
                torch.save(dynamicController.ac.pi, 'TrainedModel/centralizedActor.pkl')
                limition.saveCSV()
            if rewardSum < minReward:
                print('reward大于历史最优，已保存模型！')
                minReward = rewardSum
                limition.saveCSV_1()
    py.plot(range(len(reward_all)), reward_all, marker='o', linestyle='-')
    py.show()
    np.savetxt('./data_csv/reward_2.csv', np.array(reward_all), delimiter=',')














