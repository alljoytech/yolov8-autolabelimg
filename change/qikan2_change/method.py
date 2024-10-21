import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from scipy.spatial import ConvexHull
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def point_to_polyhedron_distance(point, polyhedron_points):
    """
    计算点到多面体的最小距离
    :param point: 三维点
    :param polyhedron_points: 多面体的顶点坐标数组，形状为 (N, 3)
    :return: 点到多面体的最小距离
    """
    hull = ConvexHull(polyhedron_points)
    min_distance = float('inf')

    for simplex in hull.simplices:
        triangle = polyhedron_points[simplex]
        dist = distance_to_triangle(point, triangle)
        min_distance = min(min_distance, dist)

    return min_distance


def distance_to_triangle(point, triangle):
    """
    计算点到三角形的最短距离
    :param point: 三维点
    :param triangle: 三角形的三个顶点，形状为 (3, 3)
    :return: 最短距离
    """
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    v0 = triangle[0] - point

    a = np.dot(edge1, edge1)
    b = np.dot(edge1, edge2)
    c = np.dot(edge2, edge2)
    d = np.dot(edge1, v0)
    e = np.dot(edge2, v0)
    f = np.dot(v0, v0)

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    if s + t <= det:
        if s < 0:
            if t < 0:
                if d < 0:
                    s = clamp(-d / a, 0, 1)
                    t = 0
                else:
                    s = 0
                    t = clamp(-e / c, 0, 1)
            else:
                s = 0
                t = clamp(-e / c, 0, 1)
        elif t < 0:
            s = clamp(-d / a, 0, 1)
            t = 0
        else:
            inv_det = 1 / det
            s *= inv_det
            t *= inv_det
    else:
        if s < 0:
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2 * b + c
                s = clamp(numer / denom, 0, 1)
                t = 1 - s
            else:
                t = clamp(-e / c, 0, 1)
                s = 0
        elif t < 0:
            if b + e > a + d:
                numer = c + e - b - d
                denom = a - 2 * b + c
                s = clamp(numer / denom, 0, 1)
                t = 1 - s
            else:
                s = clamp(-d / a, 0, 1)
                t = 0
        else:
            numer = c + e - b - d
            denom = a - 2 * b + c
            s = clamp(numer / denom, 0, 1)
            t = 1 - s

    closest_point = triangle[0] + s * edge1 + t * edge2
    return np.linalg.norm(closest_point - point)


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def compute_reference_radius(vertices):
    """
    计算多边体的参考半径（最小包围球的半径）
    :param vertices: 多边体的顶点坐标数组，形状为 (N, 3)
    :return: 多边体的参考半径
    """
    center = np.mean(vertices, axis=0)
    return np.max([np.linalg.norm(v - center) for v in vertices])

def add_height_to_polygon(polygon_2d, height):
    # Create the bottom face with z=0
    bottom_face = np.hstack((polygon_2d, np.zeros((polygon_2d.shape[0], 1))))
    # Create the top face with z=height
    top_face = np.hstack((polygon_2d, np.full((polygon_2d.shape[0], 1), height)))
    # Combine bottom and top faces
    return np.vstack((bottom_face, top_face))

def getReward(flag,limition_my, qBefore, q,qNext,obsCenter):           #计算reward函数
    reward = 0
    if qNext[-1] >= max(limition_my.maxminhigh):
        reward -= (qNext[-1] - max(limition_my.maxminhigh)) * 10
    elif qNext[-1] < min(limition_my.maxminhigh):
        reward -= (min(limition_my.maxminhigh) - qNext[-1]) * 10
    # 动态reward
    distance_dy = limition_my.distanceCost(qNext, obsCenter)
    if distance_dy <= limition_my.obsR:
        reward += (distance_dy - limition_my.obsR) / limition_my.obsR - 1
    else:
        if distance_dy < limition_my.obsR + 0.4:  # 威胁区
            tempR = limition_my.obsR + 0.4
            reward += (distance_dy - tempR) / tempR - 0.3
    # Reward_col
    if flag[0] == 0:
        if flag[1] == 0:
            point = add_height_to_polygon(limition_my.building[0, flag[2]], limition_my.buildingH)
            distance = point_to_polyhedron_distance(qNext, point)
            r = compute_reference_radius(point)
            reward += (distance - r) / r - 3
    else:
        distance1 = limition_my.distanceCost(qNext, limition_my.qgoal)
        distance2 = limition_my.distanceCost(limition_my.x0, limition_my.qgoal)
        if distance1 > limition_my.threshold:
            reward += -distance1/distance2
        else:
            reward += -distance1/distance2 + 3
    return reward


def choose_action(ActorList, s):
    actionList = []
    for i in range(len(ActorList)):
        state = s[i]
        state = torch.as_tensor(state, dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        a = ActorList[i](state).cpu().detach().numpy()
        actionList.append(a[0])
    return actionList

# def drawActionCurve(actionCurveList, obstacleName):
#     plt.figure()
#     for i in range(actionCurveList.shape[1]):
#         array = actionCurveList[:,i]
#         plt.plot(np.arange(array.shape[0]), array, linewidth = 2, label = 'Rep%d curve'%i)
#     plt.title('Variation diagram of repulsion factor of %s' %obstacleName)
#     plt.grid()
#     plt.xlabel('time')
#     plt.ylabel('value')
#     # plt.legend(loc='best')

def checkCollision(limition_my, path):  # 检查轨迹是否与障碍物碰撞
    for i in range(path.shape[0]):
        if limition_my.checkCollision(path[i,:])[0] == 0:
            return 0
    return 1

def checkPath(limition_my, path):
    sum = 0  # 轨迹距离初始化
    for i in range(path.shape[0] - 1):
        sum += limition_my.distanceCost(path[i, :], path[i + 1, :])
    if checkCollision(limition_my, path) == 1:
        print('与障碍物无交点，轨迹距离为：', sum)
    else:
        print('与障碍物有交点，轨迹距离为：', sum)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def transformAction(actionBefore, actionBound, actionDim):
    actionAfter = []
    for i in range(actionDim):
        action_i = actionBefore[i]
        action_bound_i = actionBound[i]
        actionAfter.append((action_i+1)/2*(action_bound_i[1] - action_bound_i[0]) + action_bound_i[0])
    return actionAfter

class Arguments:
    def __init__(self, limition_my):
        self.obs_dim = 6 * (limition_my.numberOfSphere + limition_my.numberOfCylinder + limition_my.numberOfCone)
        self.act_dim = 1 * (limition_my.numberOfSphere + limition_my.numberOfCylinder + limition_my.numberOfCone)
        self.act_bound = [0.1, 3]


def test(iifds, pi, conf):
    iifds.reset()    # 重置环境
    q = iifds.start
    qBefore = [None, None, None]
    rewardSum = 0
    for i in range(500):
        dic = iifds.updateObs(if_test=True)
        vObs, obsCenter, obsCenterNext = dic['v'], dic['obsCenter'], dic['obsCenterNext']
        obs = iifds.calDynamicState(q, obsCenter)
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        action = pi(obs).cpu().detach().numpy()
        action = transformAction(action, conf.actionBound, conf.act_dim)
        # 与环境交互
        qNext = iifds.getqNext(q, obsCenter, vObs, action[0], action[1], action[2], qBefore)
        rewardSum += getReward(obsCenterNext, qNext, q, qBefore, iifds)

        qBefore = q
        q = qNext

        if iifds.distanceCost(q, iifds.goal) < iifds.threshold:
            break
    return rewardSum


# def test_multiple(pi, conf):
#     """动态多障碍环境测试模型效果"""
#     reward_list = []
#     for index in range(1,7):      # 从1-6一共6个测试环境，遍历测试
#         env = Environment(index)
#         env.reset()
#         q = env.start
#         qBefore = [None, None, None]
#         rewardSum = 0
#         for i in range(500):
#             data_dic = env.update_obs_pos(q)
#             v_obs, obs_center, obs_R = data_dic['v'], data_dic['obsCenter'], data_dic['obs_r']
#             state = env.calDynamicState(q, obs_center, obs_R, v_obs)
#             state = torch.as_tensor(state, dtype=torch.float, device=device)
#             action = pi(state).cpu().detach().numpy()
#             a = transformAction(action, conf.actionBound, conf.act_dim)
#             qNext = env.getqNext(q, obs_center, v_obs, obs_R, a[0], a[1], a[2], qBefore)
#             rewardSum += get_reward_multiple(env,qNext,data_dic)
#             qBefore = q
#             q = qNext
#             if env.distanceCost(q, env.goal) < env.threshold:
#                 break
#         reward_list.append(rewardSum)
#     return reward_list







