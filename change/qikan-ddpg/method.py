import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def getReward(flag,apf,qNext):
    #计算reward函数
    reward = 0
    # Reward_col
    if flag[0] == 0:
        if flag[1] == 0:
            distance = apf.distanceCost(qNext, apf.obstacle[flag[2],:])
            reward += (distance - apf.Robstacle[flag[2]])/apf.Robstacle[flag[2]] -1
        if flag[1] == 1:
            distance = apf.distanceCost(qNext[0:2], apf.cylinder[flag[2],:])
            reward += (distance - apf.cylinderR[flag[2]])/apf.cylinderR[flag[2]] -1
        if flag[1] == 2:
            distance = apf.distanceCost(qNext[0:2], apf.cone[flag[2],:])
            r = apf.coneR[flag[2]] - qNext[2] * apf.coneR[flag[2]] / apf.coneH[flag[2]]
            reward += (distance - r)/r - 1
    else:
        # Reward_len
        distance1 = apf.distanceCost(qNext, apf.qgoal)
        distance2 = apf.distanceCost(apf.x0, apf.qgoal)
        if distance1 > apf.threshold:
            reward += -distance1/distance2
        else:
            reward += -distance1/distance2 + 3
    return reward


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

def getReward_change(flag, apf, qNext):
    reward = 0

    # Reward_col
    if flag[0] == 0:
        # 如果没有发生碰撞，计算与多边体的距离
        if flag[1] == 0:
            point = add_height_to_polygon(apf.building[0,flag[2]],apf.buildingH)
            distance = point_to_polyhedron_distance(qNext, point)
            r = compute_reference_radius(point)
            reward += (distance - r) / r - 3

    else:
        # Reward_len：计算与目标点的距离
        distance1 = apf.distanceCost(qNext, apf.qgoal)
        distance2 = apf.distanceCost(apf.x0, apf.qgoal)
        if distance1 > apf.threshold:
            reward += -distance1 / distance2
        else:
            reward += -distance1 / distance2 + 3

    return reward


def choose_action(ActorList, s):
    """
    :param ActorList: actor网络列表
    :param s: 每个agent的state append形成的列表
    :return: 每个actor给每个对应的state进行动作输出的值append形成的列表
    """
    actionList = []
    for i in range(len(ActorList)):
        state = s[i]
        state = torch.as_tensor(state, dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        a = ActorList[i](state).cpu().detach().numpy()
        actionList.append(a[0])
    return actionList

def drawActionCurve(actionCurveList, obstacleName):
    """
    :param actionCurveList: 动作值列表
    :return: None 绘制图像
    """
    plt.figure()
    for i in range(actionCurveList.shape[1]):
        array = actionCurveList[:,i]
        plt.plot(np.arange(array.shape[0]), array, linewidth = 2, label = 'Rep%d curve'%i)
    plt.title('Variation diagram of repulsion factor of %s' %obstacleName)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('value')
    # plt.legend(loc='best')

def checkCollision(apf, path):  # 检查轨迹是否与障碍物碰撞
    """
    :param apf: 环境
    :param path: 一个路径形成的列表
    :return: 1代表无碰撞 0代表碰撞
    """
    for i in range(path.shape[0]):
        if apf.checkCollision(path[i,:])[0] == 0:
            return 0
    return 1

def checkPath(apf, path):
    """
    :param apf: 环境
    :param path: 路径形成的列表
    :return: None 打印是否与障碍物有交点以及path的总距离
    """
    sum = 0  # 轨迹距离初始化
    for i in range(path.shape[0] - 1):
        sum += apf.distanceCost(path[i, :], path[i + 1, :])
    if checkCollision(apf, path) == 1:
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
        actionAfter.append((action_i+1)/2*(actionBound[1] - actionBound[0]) + actionBound[0])
    return actionAfter

class Arguments:
    def __init__(self, apf):
        self.obs_dim = 6 * (apf.numberOfSphere + apf.numberOfCylinder + apf.numberOfCone)
        self.act_dim = 1 * (apf.numberOfSphere + apf.numberOfCylinder + apf.numberOfCone)
        self.act_bound = [0.1, 3]













