import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from obstatic_my import Obstacle                   # 几个静态障碍物环境坐标
from scipy.spatial import ConvexHull


class limition:
    def __init__(self):
        #-------------------障碍物-------------------#
        '''
        障碍物坐标在z>=0范围
        圆柱和圆锥从z=0开始
        '''
        env = 'Obstacle6'
        self.building = Obstacle[env].point      # 圆柱体障碍物坐标
        self.buildingH = Obstacle[env].buildingHigh   # 圆柱体高度
        self.numberOfbuild = Obstacle[env].obsnum            # 圆锥障碍物的数量
        self.qgoal = Obstacle[env].qgoal          # 目标点
        self.x0 = Obstacle[env].x0                # 轨迹起始点
        self.stepSize = 0.2                               # 物体移动的固定步长
        self.dgoal = 5                                    # 当q与qgoal距离超过它时将衰减一部分引力
        self.r0 = 5                                       # 斥力超过这个范围后将不复存在
        self.threshold = 0.2                              # q与qgoal距离小于它时终止训练或者仿真
        #------------运动学约束------------#
        self.xmax = 10/180 * np.pi                        # 偏航角速度最大值  每个步长允许变化的角度
        self.gammax = 10/180 * np.pi                      # 爬升角速度最大值  每个步长允许变化的角度
        self.maximumClimbingAngle = 100/180 * np.pi       # 最大爬升角
        self.maximumSubductionAngle = - 75 / 180 * np.pi  # 最大俯冲角
        self.maxminhigh = [3, 25]                         # 最大最小高度
        #-------------路径（每次getqNext会自动往path添加路径）---------#
        self.path = self.x0.copy()
        self.path = self.path[np.newaxis, :]              # 增加一个维度

        # -------------force load---------# need save  the code not write
        self.attaction_load = [0, 0, 0]
        self.repulsion_load = [0, 0, 0]
        self.comfore = [0,0,0]
        self.unitforce = [0,0,0]
        #--------------多无人机参数----------#
        self.numberOfuav = 2                               #无人机数量
        #-------------一些参考参数可选择使用-------------#
        self.epsilon0 = 0.8
        self.eta0 = 0.5

    def reset(self):        # 重置环境
        self.path = self.x0.copy()
        self.path = self.path[np.newaxis, :]

    # def calculateDynamicState(self, q):
    #     dic = {'building':[]}
    #     sAll = self.qgoal - q
    #     for i in range(self.numberOfbuild):
    #         s1 = self.obstacle[i,:] - q
    #         dic['sphere'].append(np.hstack((s1,sAll)))
    #     return dic


    def calculateDynameicState_build(self, q):
        """
        :d_CP 中心点到点的距离
        :L, W, H 长方体的长度、宽度和高度
        :distance_to_face 中心点到面的距离
        :d_final 除去中心点到面的距离后的点到长方体的距离
        """
        dic = {'building':[]}
        sAll = self.qgoal - q
        for i in range(self.numberOfbuild):
            # 计算长方体的中心点和尺寸
            C = np.append(self.calculate_centroid(self.building[0,i]),self.buildingH/2)
            ##
            # dimensions = self.calculate_dimensions(self.building[i])
            # L, W, H = dimensions
            # # 计算距离
            # d_CP, distance_to_face, d_final = self.calculate_distances(C, q, L, W, H)
            #d_CP 中心点到点的距离
            s1 = C - q
            dic['building'].append(np.hstack((s1, sAll)))
        return dic


    def calculate_distances(self,C, P, L, W, H):
        # 计算中心点到点的距离
        d_CP = np.linalg.norm(np.array(P) - np.array(C))

        # 找到与某点连线所穿过的面
        intersect_point, distance_to_face = self.find_intersecting_plane(C, P, L, W, H)

        # 计算除去中心点到面的距离之后的距离
        d_final = d_CP - distance_to_face

        return d_CP, distance_to_face, d_final


    @staticmethod
    def calculate_centroid(vertices):
        # 计算中心点
        centroid = np.mean(vertices, axis=0)
        return centroid

    @staticmethod
    def calculate_dimensions(vertices):
        # 计算长方体的长度、宽度和高度
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        dimensions = max_coords - min_coords
        return dimensions

    @staticmethod
    def find_intersecting_plane(C, P, L, W, H):
        # 计算与连线所穿过的面
        dir_vector = np.array(P) - np.array(C)

        # 判断与哪个面相交，根据方向向量的符号和大小来判断
        t_values = []
        if dir_vector[0] != 0:
            t_values.append(abs((L / 2) / dir_vector[0]))
        if dir_vector[1] != 0:
            t_values.append(abs((W / 2) / dir_vector[1]))
        if dir_vector[2] != 0:
            t_values.append(abs((H / 2) / dir_vector[2]))

        t_min = min(t_values)
        intersect_point = C + t_min * dir_vector
        return intersect_point, t_min * np.linalg.norm(dir_vector)

    def pointToRectDist(self, q, qobs_min, qobs_max):
        # Calculate the minimum distance from a point to a rectangular box
        dx = max(qobs_min[0] - q[0], 0, q[0] - qobs_max[0])
        dy = max(qobs_min[1] - q[1], 0, q[1] - qobs_max[1])
        dz = max(qobs_min[2] - q[2], 0, q[2] - qobs_max[2])
        return np.array([dx, dy, dz])

    def inRepulsionArea(self, q):  # 计算一个点位r0半径范围内的障碍物索引, 返回字典{'sphere':[1,2,..],'cylinder':[0,1,..]}  2021.1.6
        """:return 返回字典{'building':[1,2,..]}"""
        dic = {'building':[]}
        for i in range(self.numberOfbuild):
            C = self.calculate_centroid(self.building[i])
            dimensions = self.calculate_dimensions(self.building[i])
            L, W, H = dimensions
            # 计算距离
            d_CP, distance_to_face, d_final = self.calculate_distances(C, q, L, W, H)
            #d_CP 中心点到点的距离
            if d_final < self.r0:
                dic['building'].append(i)
        return dic


    def attraction(self, q, epsilon):
        # """ no change"""
        r = self.distanceCost(q, self.qgoal)
        if r <= self.dgoal:
            fx = epsilon * (self.qgoal[0] - q[0])
            fy = epsilon * (self.qgoal[1] - q[1])
            fz = epsilon * (self.qgoal[2] - q[2])
        else:
            fx = self.dgoal * epsilon * (self.qgoal[0] - q[0]) / r
            fy = self.dgoal * epsilon * (self.qgoal[1] - q[1]) / r
            fz = self.dgoal * epsilon * (self.qgoal[2] - q[2]) / r
        return np.array([fx, fy, fz])

    # 计算长方体的中心点和尺寸
    @staticmethod
    def calculate_rect_properties(p1, p2, p3, p4, height):
        # 计算底面中心点
        bottom_center = (p1 + p2 + p3 + p4) / 4
        # 高度方向的偏移
        offset = np.array([0, 0, height / 2])
        # 长方体的中心点
        center = bottom_center + offset

        # 计算长度和宽度
        length = np.linalg.norm(p1 - p2)
        width = np.linalg.norm(p2 - p3)
        size = np.array([length, width, height])

        return center, size

    @staticmethod
    def add_height_to_polygon(polygon_2d, height):
        # Create the bottom face with z=0
        bottom_face = np.hstack((polygon_2d, np.zeros((polygon_2d.shape[0], 1))))
        # Create the top face with z=height
        top_face = np.hstack((polygon_2d, np.full((polygon_2d.shape[0], 1), height)))
        # Combine bottom and top faces
        return np.vstack((bottom_face, top_face))

    def pointToSegmentDist(self, p, v, w):
        # Calculate the minimum distance from point p to line segment vw
        l2 = np.sum((w - v)**2)
        if l2 == 0.0:
            return self.distanceCost(p, v), v  # v == w case
        t = max(0, min(1, np.dot(p - v, w - v) / l2))
        projection = v + t * (w - v)
        return self.distanceCost(p, projection), projection

    def pointToPolygonDist(self, q, polygon_3d):
        # Calculate the minimum distance from point q to a 3D polygon (a polyhedron)
        min_dist = float('inf')
        closest_point = None

        num_vertices = polygon_3d.shape[0] // 2
        bottom_vertices = polygon_3d[:num_vertices]
        top_vertices = polygon_3d[num_vertices:]

        # Check distance to bottom and top faces
        for i in range(num_vertices - 1):
            dist, projection = self.pointToSegmentDist(q, bottom_vertices[i], bottom_vertices[i + 1])
            if dist < min_dist:
                min_dist = dist
                closest_point = projection

            dist, projection = self.pointToSegmentDist(q, top_vertices[i], top_vertices[i + 1])
            if dist < min_dist:
                min_dist = dist
                closest_point = projection

        # Check distance to vertical edges
        for i in range(num_vertices):
            dist, projection = self.pointToSegmentDist(q, bottom_vertices[i], top_vertices[i])
            if dist < min_dist:
                min_dist = dist
                closest_point = projection

        return min_dist, closest_point

    def repulsion(self, q, action):
        f0 = np.array([0, 0, 0])
        for i in range(self.numberOfbuild):
            buildingaddH = self.add_height_to_polygon(self.building[0][i],self.buildingH)
            min_r, closest_point = self.pointToPolygonDist(q, buildingaddH)
            Rq2qgoal = self.distanceCost(q, self.qgoal)
            if min_r <= self.r0:
                dist_vec = q - closest_point
                tempfvec = action[i] * (1 / min_r  - 1 / self.r0) * Rq2qgoal ** 2 / min_r  ** 2 * dist_vec \
                           + action[i] * (1 / min_r  - 1 / self.r0) ** 2 * Rq2qgoal * self.differential(q, self.qgoal)
                f0 = f0 + tempfvec
            else:
                tempfvec = np.array([0.0, 0.0,0.0])
                f0 = f0 + tempfvec

        return f0  #这个版本  #这个

    # def repulsionForOneObstacle(self, q, eta, qobs): #这个版本的斥力计算函数计算的是一个障碍物的斥力 2020.12.24
    #     f0 = np.array([0, 0, 0])  # 初始化斥力的合力
    #     Rq2qgoal = self.distanceCost(q, self.qgoal)
    #     r = self.distanceCost(q, qobs)
    #     if r <= self.r0:
    #         tempfvec = eta * (1 / r - 1 / self.r0) * Rq2qgoal ** 2 / r ** 2 * self.differential(q, qobs) \
    #                    + eta * (1 / r - 1 / self.r0) ** 2 * Rq2qgoal * self.differential(q, self.qgoal)
    #         f0 = f0 + tempfvec
    #     else:
    #         tempfvec = np.array([0, 0, 0])
    #         f0 = f0 + tempfvec
    #     return f0
    #
    # def dynamicRepulsion(self, q):   #动态障碍物的斥力
    #     f0 = np.array([0, 0, 0])  # 初始化斥力的合力
    #     Rq2qgoal = self.distanceCost(q, self.qgoal)
    #     r = self.distanceCost(q, self.dynamicSphereXYZ)
    #     if r <= self.dynamicSpherer0:
    #         tempfvec = self.dynamicSphereEta * (1 / r - 1 / self.dynamicSpherer0) * Rq2qgoal ** 2 / r ** 2 * self.differential(q,self.dynamicSphereXYZ) \
    #                    + self.dynamicSphereEta * (1 / r - 1 / self.dynamicSpherer0) ** 2 * Rq2qgoal * self.differential(q, self.qgoal)
    #         f0 = f0 + tempfvec
    #     else:
    #         tempfvec = np.array([0, 0, 0])
    #         f0 = f0 + tempfvec
    #     return f0

    def differential(self, q, other):   #向量微分
        output1 = (q[0] - other[0]) / self.distanceCost(q, other)
        output2 = (q[1] - other[1]) / self.distanceCost(q, other)
        output3 = (q[2] - other[2]) / self.distanceCost(q, other)
        return np.array([output1, output2, output3])

    def getqNext(self, epsilon,action, q, qBefore):   #eta和epsilon需要外部提供，eta1List为球的斥力列表，eta2List为圆柱体的斥力表 fix:2021.2.9
        """
        当qBefore为[None, None, None]时，意味着q是航迹的起始点，下一位置不需要做运动学约束，否则进行运动学约束
        """
        qBefore = np.array(qBefore)
        if qBefore[0] is None:
            unitCompositeForce = self.getUnitCompositeForce(q,action,epsilon)
            qNext = q + self.stepSize * unitCompositeForce  # 计算下一位置
        else:
            unitCompositeForce = self.getUnitCompositeForce(q,action,epsilon)
            qNext = q + self.stepSize * unitCompositeForce  # 计算下一位置
            _, _, _, _, qNext = self.kinematicConstrant(q, qBefore, qNext)
        self.path = np.vstack((self.path, qNext))  # 记录轨迹
        return qNext

    def getUnitCompositeForce(self, q,action, epsilon):
        Attraction = self.attraction(q, epsilon)  # 计算引力
        self.attaction_load = np.vstack((self.attaction_load, Attraction))
        Repulsion = self.repulsion(q,action)
        self.repulsion_load = np.vstack((self.repulsion_load, Repulsion))
        compositeForce = Attraction + Repulsion  # 合力 = 引力 + 斥力
        self.comfore = np.vstack((self.comfore,compositeForce))
        unitCompositeForce = self.getUnitVec(compositeForce)  # 力单位化，apf中力只用来指示移动方向
        self.unitforce = np.vstack((self.unitforce,unitCompositeForce))
        return unitCompositeForce

    def kinematicConstrant(self, q, qBefore, qNext):
        """
        运动学约束函数 返回(上一时刻航迹角，上一时刻爬升角，约束后航迹角，约束后爬升角，约束后下一位置qNext)
        """
        # 计算qBefore到q航迹角x1,gam1
        qBefore2q = q - qBefore
        if qBefore2q[0] != 0 or qBefore2q[1] != 0:
            x1 = np.arcsin(np.abs(qBefore2q[1] / np.sqrt(qBefore2q[0] ** 2 + qBefore2q[1] ** 2)))  # 这里计算的角限定在了第一象限的角 0-pi/2
            gam1 = np.arcsin(qBefore2q[2] / np.sqrt(np.sum(qBefore2q ** 2)))
        else:
            return None, None, None, None, qNext
        # 计算q到qNext航迹角x2,gam2
        q2qNext = qNext - q
        x2 = np.arcsin(np.abs(q2qNext[1] / np.sqrt(q2qNext[0] ** 2 + q2qNext[1] ** 2)))  # 这里同理计算第一象限的角度
        gam2 = np.arcsin(q2qNext[2] / np.sqrt(np.sum(q2qNext ** 2)))

        # 根据不同象限计算矢量相对于x正半轴的角度 0-2 * pi
        if qBefore2q[0] > 0 and qBefore2q[1] > 0:
            x1 = x1
        if qBefore2q[0] < 0 and qBefore2q[1] > 0:
            x1 = np.pi - x1
        if qBefore2q[0] < 0 and qBefore2q[1] < 0:
            x1 = np.pi + x1
        if qBefore2q[0] > 0 and qBefore2q[1] < 0:
            x1 = 2 * np.pi - x1
        if qBefore2q[0] > 0 and qBefore2q[1] == 0:
            x1 = 0
        if qBefore2q[0] == 0 and qBefore2q[1] > 0:
            x1 = np.pi / 2
        if qBefore2q[0] < 0 and qBefore2q[1] == 0:
            x1 = np.pi
        if qBefore2q[0] == 0 and qBefore2q[1] < 0:
            x1 = np.pi * 3 / 2


        # 根据不同象限计算与x正半轴的角度
        if q2qNext[0] > 0 and q2qNext[1] > 0:
            x2 = x2
        if q2qNext[0] < 0 and q2qNext[1] > 0:
            x2 = np.pi - x2
        if q2qNext[0] < 0 and q2qNext[1] < 0:
            x2 = np.pi + x2
        if q2qNext[0] > 0 and q2qNext[1] < 0:
            x2 = 2 * np.pi - x2
        if q2qNext[0] > 0 and q2qNext[1] == 0:
            x2 = 0
        if q2qNext[0] == 0 and q2qNext[1] > 0:
            x2 = np.pi / 2
        if q2qNext[0] < 0 and q2qNext[1] == 0:
            x2 = np.pi
        if q2qNext[0] == 0 and q2qNext[1] < 0:
            x2 = np.pi * 3 / 2

        # 约束航迹角x   xres为约束后的航迹角
        deltax1x2 = self.angleVec(q2qNext[0:2], qBefore2q[0:2])  # 利用点乘除以模长乘积求xoy平面投影的夹角
        if deltax1x2 < self.xmax:
            xres = x2
        elif x1 - x2 > 0 and x1 - x2 < np.pi:  # 注意这几个逻辑
            xres = x1 - self.xmax
        elif x1 - x2 > 0 and x1 - x2 > np.pi:
            xres = x1 + self.xmax
        elif x1 - x2 < 0 and x2 - x1 < np.pi:
            xres = x1 + self.xmax
        else:
            xres = x1 - self.xmax

        # 约束爬升角gam   注意：爬升角只用讨论在-pi/2到pi/2区间，这恰好与arcsin的值域相同。  gamres为约束后的爬升角
        if np.abs(gam1 - gam2) <= self.gammax:
            gamres = gam2
        elif gam2 > gam1:
            gamres = gam1 + self.gammax
        else:
            gamres = gam1 - self.gammax
        if gamres > self.maximumClimbingAngle:
            gamres = self.maximumClimbingAngle
        if gamres < self.maximumSubductionAngle:
            gamres = self.maximumSubductionAngle

        # 计算约束过后下一个点qNext的坐标
        Rq2qNext = self.distanceCost(q, qNext)
        deltax = Rq2qNext * np.cos(gamres) * np.cos(xres)
        deltay = Rq2qNext * np.cos(gamres) * np.sin(xres)
        deltaz = Rq2qNext * np.sin(gamres)

        qNext = q + np.array([deltax, deltay, deltaz])
        return x1, gam1, xres, gamres, qNext

    # def checkCollision(self, q):#change
    #     """
    #     检查一个位置是否碰撞障碍物,如果碰撞返回[0,障碍物类型index, 碰撞障碍物index]，如果没有碰撞返回[1,-1, -1]
    #     """
    #     for i in range(self.numberOfSphere):#球的检测碰撞
    #         if self.distanceCost(q, self.obstacle[i, :]) <= self.Robstacle[i]:
    #             return np.array([0,0,i])
    #     for i in range(self.numberOfCylinder): #圆柱体的检测碰撞
    #         if 0 <= q[2] <= self.cylinderH[i] and self.distanceCost(q[0:2], self.cylinder[i, :]) <= self.cylinderR[i]:
    #             return np.array([0,1,i])
    #     for i in range(self.numberOfCone):
    #         if q[2] >= 0 and self.distanceCost(q[0:2], self.cone[i,:]) <= self.coneR[i] - q[2] * self.coneR[i] / self.coneH[i]:
    #             return np.array([0,2,i])
    #     return np.array([1,-1, -1])   #不撞的编码

    @staticmethod
    def point_to_plane_distance(point, plane_points):
        """
        计算点到平面的距离
        :param point: 需要检测的点，形状为 (3,)
        :param plane_points: 定义平面的三个点，形状为 (3, 3)
        :return: 点到平面的距离
        """
        vec1 = plane_points[1] - plane_points[0]
        vec2 = plane_points[2] - plane_points[0]
        normal = np.cross(vec1, vec2)
        distance = np.dot(normal, point - plane_points[0]) / np.linalg.norm(normal)
        return np.abs(distance)

    def is_point_in_polyhedron(self,point, vertices, tolerance=1e-6):
        """
        检测点是否在多边体内部或与多边体的面重合
        :param point: 需要检测的点，形状为 (3,)
        :param vertices: 多边体的顶点，形状为 (N, 3)
        :param tolerance: 允许的误差范围，默认值为1e-6
        :return: 如果点在多边体内部或与多边体的面重合，返回True，否则返回False
        """
        hull = ConvexHull(vertices)
        if_in = 0
        # 检查点是否在多边体内部或与面的距离为零
        for simplex in hull.simplices:
            face = vertices[simplex]
            distance = self.point_to_plane_distance(point, face)
            if distance < tolerance:
                if_in = 1
                return if_in

        # 使用多面体的凸包检查点是否在多面体内部
        if np.all(np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1] <= tolerance):
            if_in = 1
            return if_in

        return if_in

    def checkCollision(self,q):
        """
        检查一个位置是否碰撞障碍物,如果碰撞返回[0,障碍物类型index, 碰撞障碍物index]，如果没有碰撞返回[1,-1, -1]
        """
        for i in range(self.numberOfbuild):
            buildingaddH = self.add_height_to_polygon(self.building[0][i], self.buildingH)
            result = self.is_point_in_polyhedron(q, buildingaddH)
            if result==1:return np.array([0,0,i])
        return np.array([1,-1,-1])


    @staticmethod
    def distanceCost(point1, point2):  # 求两点之间的距离函数
        return np.sqrt(np.sum((point1 - point2) ** 2))

    @staticmethod
    def angleVec(vec1, vec2):  # 计算两个向量之间的夹角
        temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
        temp = np.clip(temp,-1,1)  # 可能存在精度误差导致上一步的temp略大于1，因此clip
        theta = np.arccos(temp)
        return theta

    @staticmethod
    def getUnitVec(vec):   #单位化向量方法
        unitVec = vec / np.sqrt(np.sum(vec ** 2))
        return unitVec

    def calculateLength(self):
        """
        对类中自带的path进行距离计算，path会在getqNext函数中自动添加位置
        """
        sum = 0  # 轨迹距离初始化
        for i in range(self.path.shape[0] - 1):
            sum += self.distanceCost(self.path[i, :], self.path[i + 1, :])
        return sum

    # def drawEnv(self):    #绘制环境方法，matplotlib渲染属实不行，这里只是测试
    #     fig = plt.figure()
    #     self.ax=Axes3D(fig)
    #     plt.grid(True)  # 添加网格
    #     self.ax.scatter3D(self.qgoal[0], self.qgoal[1], self.qgoal[2], marker='o', color='red', s=100, label='Goal')
    #     self.ax.scatter3D(self.x0[0], self.x0[1], self.x0[2], marker='o', color='blue', s=100, label='Start')
    #     for i in range(self.Robstacle.shape[0]): #绘制球
    #         self.drawSphere(self.obstacle[i, :], self.Robstacle[i])
    #     for i in range(self.cylinder.shape[0]):  #绘制圆柱体
    #         self.drawCylinder(self.cylinder[i,:],self.cylinderR[i], self.cylinderH[i])
    #     plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置
    #     plt.grid()
    #     self.ax.set_xlim3d(left = 0, right = 10)
    #     self.ax.set_ylim3d(bottom=0, top=10)
    #     self.ax.set_zlim3d(bottom=0, top=10)
    #
    # def drawSphere(self, center, radius):   #绘制球函数
    #     u = np.linspace(0, 2 * np.pi, 40)
    #     v = np.linspace(0, np.pi, 40)
    #     x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    #     y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    #     z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    #     h = self.ax.plot_wireframe(x, y, z, cstride=4, color='b')
    #     return h
    #
    # def drawCylinder(self, center, radius, height):  #绘制圆柱体函数
    #     u = np.linspace(0, 2 * np.pi, 30)  # 把圆分按角度为50等分
    #     h = np.linspace(0, height, 20)  # 把高度均分为20份
    #     x = np.outer(center[0] + radius * np.sin(u), np.ones(len(h)))  # x值重复20次
    #     y = np.outer(center[1] + radius * np.cos(u), np.ones(len(h)))  # y值重复20次
    #     z = np.outer(np.ones(len(u)), h)  # x，y 对应的高度
    #     # Plot the surface
    #     self.ax.plot_surface(x, y, z)  #也可以plot_wireframe

    def saveCSV(self):   #保存数据方便matlab绘图(受不了matplotlib三维了)
        np.savetxt('./data_csv/pathMatrix.csv', self.path, delimiter=',')
        np.savetxt('./data_csv/obstacleMatrix.csv', self.obstacle, delimiter=',')
        np.savetxt('./data_csv/RobstacleMatrix.csv', self.Robstacle, delimiter=',')
        np.savetxt('./data_csv/cylinderMatrix.csv', self.cylinder, delimiter=',')
        np.savetxt('./data_csv/cylinderRMatrix.csv', self.cylinderR, delimiter=',')
        np.savetxt('./data_csv/cylinderHMatrix.csv', self.cylinderH, delimiter=',')
        np.savetxt('./data_csv/coneMatrix.csv', self.cone, delimiter=',')
        np.savetxt('./data_csv/coneRMatrix.csv', self.coneR, delimiter=',')
        np.savetxt('./data_csv/coneHMatrix.csv', self.coneH, delimiter=',')
        np.savetxt('./data_csv/start.csv', self.x0, delimiter=',')
        np.savetxt('./data_csv/goal.csv', self.qgoal, delimiter=',')

    def saveCSV_1(self):   #保存数据方便matlab绘图(受不了matplotlib三维了)
        np.savetxt('./data_csv_1/pathMatrix.csv', self.path, delimiter=',')
        # np.savetxt('./data_csv_1/obstacleMatrix.csv', self.obstacle, delimiter=',')
        # np.savetxt('./data_csv_1/RobstacleMatrix.csv', self.Robstacle, delimiter=',')
        # np.savetxt('./data_csv_1/cylinderMatrix.csv', self.cylinder, delimiter=',')
        # np.savetxt('./data_csv_1/cylinderRMatrix.csv', self.cylinderR, delimiter=',')
        # np.savetxt('./data_csv_1/cylinderHMatrix.csv', self.cylinderH, delimiter=',')
        # np.savetxt('./data_csv_1/coneMatrix.csv', self.cone, delimiter=',')
        # np.savetxt('./data_csv_1/coneRMatrix.csv', self.coneR, delimiter=',')
        # np.savetxt('./data_csv_1/coneHMatrix.csv', self.coneH, delimiter=',')
        np.savetxt('./data_csv_1/start.csv', self.x0, delimiter=',')
        np.savetxt('./data_csv_1/goal.csv', self.qgoal, delimiter=',')

    def drawPath(self):   #绘制path变量
        self.ax.plot3D(self.path[:,0],self.path[:,1],self.path[:,2],color="deeppink",linewidth=2,label = 'UAV path')


    def discost(self):
        a = self.path
        distances = np.sqrt(np.sum(np.diff(a, axis=0) ** 2, axis=1))
        # 计算总距离
        total_distance = np.sum(distances)
        return total_distance



    #--------测试用方法---------#
    def loop(self):             #循环仿真
        q = self.x0.copy()
        qBefore = [None, None, None]
        eta1List = [0.2 for i in range(self.obstacle.shape[0])]
        eta2List = [0.2 for i in range(self.cylinder.shape[0])]
        eta3List = [0.2 for i in range(self.cone.shape[0])]
        for i in range(500):
            qNext = self.getqNext(self.epsilon0,eta1List,eta2List,eta3List, q,qBefore)  # qBefore是上一个点
            qBefore = q

            # self.ax.plot3D([q[0], qNext[0]], [q[1], qNext[1]], [q[2], qNext[2]], color="k",linewidth=2)  # 绘制上一位置和这一位置

            q = qNext

            if self.distanceCost(qNext,self.qgoal) < self.threshold:   #当与goal之间距离小于threshold时结束仿真，并将goal的坐标放入path
                self.path = np.vstack((self.path,self.qgoal))
                # self.ax.plot3D([qNext[0], self.qgoal[0]], [qNext[1], self.qgoal[1]], [qNext[2], self.qgoal[2]], color="k",linewidth=2)  # 绘制上一位置和这一位置
                break
            # plt.pause(0.001)

if __name__ == '__main__':
    limt = limition()
    q = [0,0,0]
    limt.repulsion(q)
    limt.getUnitCompositeForce(q,0.8)






