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
    obs_dim = 6 * (limition.numberOfbuild) + 9
    act_dim = 1 * (limition.numberOfbuild) + 3
    act_bound = [0.1, 3]
    act_dim_dy = 3
    actionBound = [[0.1, 3], [0.1, 3], [0.1, 3]] # 连续空间
    high_level_task_dim = 5 # 高层决策
    dynamicController = DDPG(obs_dim, act_dim, act_bound,high_level_task_dim)
    sumyes = 0
    MAX_EPISODE = 500
    MAX_STEP = 3000
    update_every = 50
    batch_size = 128
    noise = 0.3
    update_cnt = 0
    rewardList = []
    reward_all = []
    noise_all = []
    dissum = []
    maxReward = -np.inf
    for episode in range(MAX_EPISODE):
        q = limition.x0
        limition.reset()
        rewardSum = 0
        qBefore = [None, None, None]
        for j in range(MAX_STEP):
            dic = limition.updateObs()
            vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
            obsDicq = limition.calculateDynameicState_build(q)
            obs_mix = obsDicq['building']
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
                action_build = action[0:limition.numberOfbuild]
                action_dym = action[limition.numberOfbuild:]
            else:
                action_build = [random.uniform(act_bound[0], act_bound[1]) for k in range(limition.numberOfbuild)]
                action_dym = [random.uniform(-1, 1) for i in range(act_dim_dy)]
                action = action_build + action_dym

            # 与环境交互
            # if episode ==51:
            #     actionAfter = transformAction(action_dym, actionBound, act_dim_dy)
            actionAfter = transformAction(action_dym, actionBound, act_dim_dy)
            qNext = limition.getqNext_sta_dy_change(limition.epsilon0, action_build, q, qBefore,
                                             obsCenter, vObs, actionAfter[0], actionAfter[1], actionAfter[2])
            obsDicqNext = limition.calculateDynameicState_build(qNext)
            obs_min_next = obsDicqNext['building']
            obs_sta = np.array([])
            for k in range(len(obs_min_next)):
                obs_sta = np.hstack((obs_sta, obs_min_next[k]))
            obs_next_dy = limition.calDynamicState_dy(qNext, obsCenterNext)
            obs_next = np.hstack((obs_sta, obs_next_dy))
            flag = limition.checkCollision(qNext,obsCenterNext)
            reward = getReward(flag, limition, qBefore, q, qNext, obsCenterNext)
            rewardSum += reward
            done = True if limition.distanceCost(limition.qgoal, qNext) < limition.threshold else False
            dynamicController.replay_buffer.store(obs, action, reward, obs_next, done)
            if episode >= 50 and j % update_every == 0:
                if dynamicController.replay_buffer.size >= batch_size:
                    update_cnt += update_every
                    for _ in range(update_every):
                        batch = dynamicController.replay_buffer.sample_batch(batch_size)
                        dynamicController.update(data=batch)
            if done: break
            qBefore = q
            q = qNext
        reward_all.append(rewardSum)
        # 选择测试其他环境 此处先忽略 查看此环境是否可以
        print('Episode:', episode, 'Reward:%f' % rewardSum, 'noise:%f' % noise, 'update_cnt:%d' % update_cnt)
        if episode > MAX_EPISODE * 2 / 3:
            if rewardSum > maxReward:
                limition.saveCSV()
                print('reward大于历史最优，已保存模型！')
                maxReward = rewardSum
                torch.save(dynamicController.ac.pi, 'TrainedModel/centralizedActor.pkl')
    limition.saveCSV_1()
    # py.plot(range(len(reward_all)), reward_all, marker='o', linestyle='-')
    # py.show()
    # limition.saveCSV()















