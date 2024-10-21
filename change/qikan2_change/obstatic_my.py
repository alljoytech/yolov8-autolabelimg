import numpy as np
import scipy.io as scio



class Obstacle1:
    def __init__(self):
        self.obstacle = np.array([[3, 2, 1],
                                  [3, 4, 2],
                                  [5, 4, 2],
                                  [6, 2, 1]
                                  ], dtype=float)  # 球障碍物坐标
        self.Robstacle = np.array([1, 2, 2, 1], dtype=float)  # 球半径
        self.cylinder = np.array([[8, 3]
                                  ], dtype=float)  # 圆柱体障碍物坐标（只需要给定圆形的x,y即可，无顶盖）
        self.cylinderR = np.array([0.5], dtype=float)  # 圆柱体障碍物半径
        self.cylinderH = np.array([8], dtype=float)  # 圆柱体高度
        self.cone = np.array([[4, 0]
                              ], dtype=float)  # 圆锥底面中心坐标
        self.coneR = np.array([1], dtype=float)  # 圆锥底面圆半径
        self.coneH = np.array([3], dtype=float)  # 圆锥高度
        self.qgoal = np.array([8, 6, 1.2], dtype=float)  # 目标点
        self.x0 = np.array([0, 0, 2], dtype=float)  # 轨迹起始点


class Obstacle2:
    def __init__(self):
        self.obstacle = np.array([[2, 3, 2],
                                  [4, 7, 1],
                                  [7, 7, 2]
                                  ], dtype=float)  # 球障碍物坐标
        self.Robstacle = np.array([2, 1, 2], dtype=float)  # 球半径
        self.cylinder = np.array([[5, 5],
                                  [8, 4]
                                  ], dtype=float)  # 圆柱体障碍物坐标（只需要给定圆形的x,y即可，无顶盖）
        self.cylinderR = np.array([1, 1], dtype=float)  # 圆柱体障碍物半径
        self.cylinderH = np.array([7, 6], dtype=float)  # 圆柱体高度
        self.cone = np.array([[5, 2]
                              ], dtype=float)  # 圆锥底面中心坐标
        self.coneR = np.array([2], dtype=float)  # 圆锥底面圆半径
        self.coneH = np.array([3], dtype=float)  # 圆锥高度

        self.qgoal = np.array([10, 7, 1], dtype=float)  # 目标点
        self.x0 = np.array([-2, 1, 3], dtype=float)  # 轨迹起始点


class Obstacle3:
    def __init__(self):
        self.obstacle = np.array([[2, 2, 1],
                                  [2, 4, 1],
                                  [4, 2, 1],
                                  [4, 4, 1],
                                  [6, 2, 1],
                                  [6, 4, 1]
                                  ], dtype=float)  # 球障碍物坐标
        self.Robstacle = np.array([1, 1, 1, 1, 1, 1], dtype=float)  # 球半径
        self.cylinder = np.array([
        ], dtype=float)  # 圆柱体障碍物坐标（只需要给定圆形的x,y即可，无顶盖）
        self.cylinderR = np.array([], dtype=float)  # 圆柱体障碍物半径
        self.cylinderH = np.array([], dtype=float)  # 圆柱体高度
        self.cone = np.array([
        ], dtype=float)  # 圆锥底面中心坐标
        self.coneR = np.array([], dtype=float)  # 圆锥底面圆半径
        self.coneH = np.array([], dtype=float)  # 圆锥高度

        self.qgoal = np.array([8, 6, 1.5], dtype=float)  # 目标点
        self.x0 = np.array([0, 0, 1], dtype=float)  # 轨迹起始点


class Obstacle4:
    def __init__(self):
        self.obstacle = np.array([[2, 5, 2],
                                  [4, 3, 2],
                                  [8, 8, 2]
                                  ], dtype=float)  # 球障碍物坐标
        self.Robstacle = np.array([2, 2, 2], dtype=float)  # 球半径
        self.cylinder = np.array([[4, 7]
                                  ], dtype=float)  # 圆柱体障碍物坐标（只需要给定圆形的x,y即可，无顶盖）
        self.cylinderR = np.array([1], dtype=float)  # 圆柱体障碍物半径
        self.cylinderH = np.array([6], dtype=float)  # 圆柱体高度
        self.cone = np.array([[8, 5]
                              ], dtype=float)  # 圆锥底面中心坐标
        self.coneR = np.array([2], dtype=float)  # 圆锥底面圆半径
        self.coneH = np.array([4], dtype=float)  # 圆锥高度

        self.qgoal = np.array([10, 11, 2], dtype=float)  # 目标点
        self.x0 = np.array([0, 0, 3], dtype=float)  # 轨迹起始点


class Obstacle5:
    def __init__(self):
        self.obstacle = np.array([[2, 5, 2],
                                  [4, 3, 2],
                                  [8, 8, 2],
                                  [12, 15, 4]
                                  ], dtype=float)  # 球障碍物坐标
        self.Robstacle = np.array([2, 2, 2, 4], dtype=float)  # 球半径
        self.cylinder = np.array([[4, 7], [12, 27],
                                  [8, 15], [23, 28],
                                  [10, 9], [22, 25],
                                  [6, 14], [11, 14],
                                  [14, 14], [20, 30],
                                  [11, 17], [20, 19],
                                  [8, 17], [25, 29],
                                  [11, 11], [14, 17],
                                  [8, 16], [14, 12],
                                  [17, 28], [23, 31],
                                  [25, 26], [17, 30],
                                  [16, 25], [18, 21],
                                  [14, 28], [18, 32],
                                  [15, 31], [13, 24],
                                  [19, 24], [21, 21]
                                  ], dtype=float)
        self.cylinderR = np.array(
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3],
            dtype=float)
        self.cylinderH = np.array(
            [6, 1, 5, 10, 8, 13, 10, 3, 4, 6, 8, 9, 13, 2, 4, 3, 6, 8, 4, 18, 1, 4, 6, 8, 5, 3, 4, 2, 1, 10],
            dtype=float)
        self.cone = np.array([[8, 5],
                              [25, 25]
                              ], dtype=float)  # 圆锥底面中心坐标
        self.coneR = np.array([2, 6], dtype=float)  # 圆锥底面圆半径
        self.coneH = np.array([4, 10], dtype=float)  # 圆锥高度

        self.qgoal = np.array([30, 35, 5], dtype=float)  # 目标点
        self.x0 = np.array([0, 0, 3], dtype=float)  # 轨迹起始点


class Obstacle6:
    def __init__(self):
        self.choose_building_point = scio.loadmat('choose_building_face.mat')
        self.point = self.choose_building_point['choose_building_face']
        self.obsnum = len(self.point[0])
        self.buildingHigh = 20
        self.qgoal = np.array([21, 22, 5], dtype=float)  # 目标点
        self.x0 = np.array([15, 12, 5], dtype=float)  # 轨迹起始点


Obstacle = {"Obstacle1": Obstacle1(), "Obstacle2": Obstacle2(), "Obstacle3": Obstacle3(), "Obstacle4": Obstacle4(),
            "Obstacle5": Obstacle5(),'Obstacle6':Obstacle6()}



