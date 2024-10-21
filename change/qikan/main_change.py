from DDPGModel import DDPG
from env_al import limition
from method import getReward, setup_seed
import random
import numpy as np
import torch
import matplotlib.pyplot as py

if __name__ == '__main__':
    setup_seed(11)   # 设置随机数种子

    limition = limition()
    obs_dim = 6 * (limition.numberOfSphere + limition.numberOfCylinder + limition.numberOfCone)
    act_dim = 1 * (limition.numberOfSphere + limition.numberOfCylinder + limition.numberOfCone)
    act_bound = [0.1, 3]

    centralizedContriller = DDPG(obs_dim, act_dim, act_bound)
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
    z = 0
    for episode in range(MAX_EPISODE):
        q = limition.x0
        limition.reset()
        rewardSum = 0
        qBefore = [None, None, None]
        for j in range(MAX_STEP):
            obsDicq = limition.calculateDynamicState(q)
            obs_sphere, obs_cylinder, obs_cone = obsDicq['sphere'], obsDicq['cylinder'], obsDicq['cone']
            obs_mix = obs_sphere + obs_cylinder + obs_cone
            obs = np.array([]) # 中心控制器接受所有状态集合
            for k in range(len(obs_mix)):
                obs = np.hstack((obs, obs_mix[k])) # 拼接状态为一个1*n向量
            if episode > 50:
                noise *= 0.99995
                if noise <= 0.1: noise = 0.1
                action = centralizedContriller.get_action(obs, noise_scale=noise)
                # 分解动作向量
                action_sphere = action[0:limition.numberOfSphere]
                action_cylinder = action[limition.numberOfSphere:limition.numberOfSphere + limition.numberOfCylinder]
                action_cone = action[limition.numberOfSphere + limition.numberOfCylinder:limition.numberOfSphere +\
                                          limition.numberOfCylinder + limition.numberOfCone]
            else:
                action_sphere = [random.uniform(act_bound[0],act_bound[1]) for k in range(limition.numberOfSphere)]
                action_cylinder = [random.uniform(act_bound[0],act_bound[1]) for k in range(limition.numberOfCylinder)]
                action_cone = [random.uniform(act_bound[0],act_bound[1]) for k in range(limition.numberOfCone)]
                action = action_sphere + action_cylinder + action_cone

            # 与环境交互
            qNext = limition.getqNext(limition.epsilon0, action_sphere, action_cylinder, action_cone, q, qBefore)
            obsDicqNext = limition.calculateDynamicState(qNext)
            obs_sphere_next, obs_cylinder_next, obs_cone_next = obsDicqNext['sphere'], obsDicqNext['cylinder'], obsDicqNext['cone']
            obs_mix_next = obs_sphere_next + obs_cylinder_next + obs_cone_next
            obs_next = np.array([])
            for k in range(len(obs_mix_next)):
                obs_next = np.hstack((obs_next, obs_mix_next[k]))
            flag = limition.checkCollision(qNext)
            reward = getReward(flag, limition, qBefore, q, qNext)
            rewardSum += reward

            done = True if limition.distanceCost(limition.qgoal, qNext) < limition.threshold else False
            centralizedContriller.replay_buffer.store(obs, action, reward, obs_next, done)

            if episode >= 50 and j % update_every == 0:
                if centralizedContriller.replay_buffer.size >= batch_size:
                    update_cnt += update_every
                    if z == 1:
                        labels_k, size_k,positive_choose = centralizedContriller.replay_buffer.KNN_my_3(centers_k, size_k, labels_k,positive_choose)
                    if z == 0:
                        labels_k, centers_k, size_k,positive_choose = centralizedContriller.replay_buffer.KNN_my()
                        z += 1
                    for _ in range(update_every):
                        batch,positive_choose = centralizedContriller.replay_buffer.sample_batch_other_2(labels_k, batch_size,positive_choose)
                        centralizedContriller.update(data=batch)
            if done:
                sumyes += 1
                break
            qBefore = q
            q = qNext
        dissum.append(limition.discost())
        noise_all.append(noise)
        reward_all.append(rewardSum)
        print('Episode:', episode, 'Reward:%f' % rewardSum, 'noise:%f' % noise, 'update_cnt:%d' %update_cnt)
        rewardList.append(round(rewardSum,2))
        if episode > MAX_EPISODE*2/3:
            if rewardSum > maxReward:
                print('reward大于历史最优，已保存模型！')
                maxReward = rewardSum
                torch.save(centralizedContriller.ac.pi, 'TrainedModel/centralizedActor_change.pkl')
    limition.saveCSV_1()
    np.savetxt('./data_csv_1/reward_2.csv', np.array(reward_all), delimiter=',')
    np.savetxt('./data_csv_1/discost.csv', np.array(dissum), delimiter=',')
    print(sumyes)