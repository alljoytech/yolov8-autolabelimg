import numpy as np
import random

"""提供多个单障碍物动态环境用于训练UAV"""


def obstacle1(time_now, time_step):
    obs_ref = np.array([5, 5, 5], dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0] + 2 * np.cos(time_now),
                          obs_ref[1] + 2 * np.sin(time_now),
                          obs_ref[2]], dtype=float)
    vObs = np.array([-2 * np.sin(time_now),
                     2 * np.cos(time_now), 0])
    time_now += time_step

    obsCenterNext = np.array([obs_ref[0] + 2 * np.cos(time_now),
                              obs_ref[1] + 2 * np.sin(time_now),
                              obs_ref[2]], dtype=float)
    vObsNext = np.array([-2 * np.sin(time_now), 2 * np.cos(time_now), 0])
    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic


def obstacle2(time_now, time_step):
    obs_ref = np.array([9, 9, 5.5], dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0] - 0.5 * time_now + random.uniform(-1, 1),
                          obs_ref[1] - 0.5 * time_now + random.uniform(-1, 1),
                          obs_ref[2] + np.sin(2 * time_now)], dtype=float)
    vObs = np.array([-0.5, -0.5, 2 * np.cos(2 * time_now)])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0] - 0.5 * time_now + random.uniform(-1, 1),
                              obs_ref[1] - 0.5 * time_now + random.uniform(-1, 1),
                              obs_ref[2] + np.sin(2 * time_now)], dtype=float)
    vObsNext = np.array([-0.5, -0.5, 2 * np.cos(2 * time_now)])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic


def obstacle3(time_now, time_step):
    obs_ref = np.array([5, 10, 5.5], dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0],
                          obs_ref[1] - time_now,
                          obs_ref[2] + np.sin(2 * time_now)], dtype=float)
    vObs = np.array([0, -1, 2 * np.cos(2 * time_now)])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0],
                              obs_ref[1] - time_now,
                              obs_ref[2] + np.sin(2 * time_now)], dtype=float)
    vObsNext = np.array([0, -1, 2 * np.cos(2 * time_now)])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic


def obstacle4(time_now, time_step):
    obs_ref = np.array([6, 6, 5], dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0] + 3 * np.cos(0.5 * time_now),
                          obs_ref[1] + 3 * np.sin(0.5 * time_now),
                          obs_ref[2] + np.sin(2 * time_now)], dtype=float)
    vObs = np.array([-1.5 * np.sin(0.5 * time_now), 1.5 * np.cos(0.5 * time_now), 2 * np.cos(2 * time_now)])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0] + 3 * np.cos(0.5 * time_now),
                              obs_ref[1] + 3 * np.sin(0.5 * time_now),
                              obs_ref[2] + np.sin(2 * time_now)], dtype=float)
    vObsNext = np.array([-1.5 * np.sin(0.5 * time_now), 1.5 * np.cos(0.5 * time_now), 2 * np.cos(2 * time_now)])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic


def obstacle5(time_now, time_step):
    obs_ref = np.array([5, 8, 5], dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0] + 3 * np.sin(0.5 * time_now),
                          obs_ref[1] + 3 * np.cos(0.5 * time_now),
                          obs_ref[2] + np.sin(0.5 * time_now)], dtype=float)
    vObs = np.array([1.5 * np.cos(0.5 * time_now), -1.5 * np.sin(0.5 * time_now), 0.5 * np.cos(0.5 * time_now)])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0] + 3 * np.sin(0.5 * time_now),
                              obs_ref[1] + 3 * np.cos(0.5 * time_now),
                              obs_ref[2] + np.sin(0.5 * time_now)], dtype=float)
    vObsNext = np.array([1.5 * np.cos(0.5 * time_now), -1.5 * np.sin(0.5 * time_now), 0.5 * np.cos(0.5 * time_now)])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic


def obstacle6(time_now, time_step):
    obs_ref = np.array([5, 6, 5], dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0] + 3 * np.sin(0.5 * time_now) + random.uniform(-1, 1),
                          obs_ref[1] + np.cos(0.5 * time_now) + random.uniform(-1, 1),
                          obs_ref[2] + np.sin(0.5 * time_now)], dtype=float)
    vObs = np.array([1.5 * np.cos(0.5 * time_now), -0.5 * np.sin(0.5 * time_now), 0.5 * np.cos(0.5 * time_now)])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0] + 3 * np.sin(0.5 * time_now) + random.uniform(-1, 1),
                              obs_ref[1] + np.cos(0.5 * time_now) + random.uniform(-1, 1),
                              obs_ref[2] + np.sin(0.5 * time_now)], dtype=float)
    vObsNext = np.array([1.5 * np.cos(0.5 * time_now), -0.5 * np.sin(0.5 * time_now), 0.5 * np.cos(0.5 * time_now)])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic


def obstacle7(time_now, time_step):
    obs_ref = np.array([10, 10, 5], dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0] - time_now,
                          obs_ref[1] - time_now,
                          obs_ref[2]], dtype=float)
    vObs = np.array([-1, -1, 0])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0] - time_now,
                              obs_ref[1] - time_now,
                              obs_ref[2]], dtype=float)
    vObsNext = np.array([-1, -1, 0])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic


def obstacle8(time_now, time_step):
    obs_ref = np.array([3, 10, 5], dtype=float)
    dic = {}
    time_thre = 8
    if time_now < time_thre:
        obsCenter = np.array([obs_ref[0] + 5 * np.sin(np.pi / 2 + 0.3 * time_now),
                              obs_ref[1] + 5 * np.cos(np.pi / 2 + 0.3 * time_now),
                              obs_ref[2]], dtype=float)
        vObs = np.array([1.5 * np.cos(np.pi / 2 + 0.3 * time_now), -1.5 * np.sin(np.pi / 2 + 0.3 * time_now), 0])
        time_now += time_step
        obsCenterNext = np.array([obs_ref[0] + 5 * np.sin(np.pi / 2 + 0.3 * time_now),
                                  obs_ref[1] + 5 * np.cos(np.pi / 2 + 0.3 * time_now),
                                  obs_ref[2]], dtype=float)
        vObsNext = np.array([1.5 * np.cos(np.pi / 2 + 0.3 * time_now), -1.5 * np.sin(np.pi / 2 + 0.3 * time_now), 0])
    else:
        delta_time = time_now - time_thre
        time_cal = time_thre - delta_time
        obsCenter = np.array([obs_ref[0] + 5 * np.sin(np.pi / 2 + 0.3 * time_cal),
                              obs_ref[1] + 5 * np.cos(np.pi / 2 + 0.3 * time_cal),
                              obs_ref[2]], dtype=float)
        vObs = np.array([1.5 * np.cos(np.pi / 2 + 0.3 * time_cal), -1.5 * np.sin(np.pi / 2 + 0.3 * time_cal), 0])
        time_now += time_step
        time_cal -= time_step
        obsCenterNext = np.array([obs_ref[0] + 5 * np.sin(np.pi / 2 + 0.3 * time_cal),
                                  obs_ref[1] + 5 * np.cos(np.pi / 2 + 0.3 * time_cal),
                                  obs_ref[2]], dtype=float)
        vObsNext = np.array([1.5 * np.cos(np.pi / 2 + 0.3 * time_cal), -1.5 * np.sin(np.pi / 2 + 0.3 * time_cal), 0])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic


def obstacle9(time_now, time_step):
    obs_ref = np.array([5, 5, 5], dtype=float)
    dic = {}
    obsCenter = np.array([obs_ref[0] + random.uniform(-1, 1),
                          obs_ref[1] + random.uniform(-1, 1),
                          obs_ref[2]], dtype=float)
    vObs = np.array([-1, -1, 0])
    time_now += time_step
    obsCenterNext = np.array([obs_ref[0] + random.uniform(-1, 1),
                              obs_ref[1] + random.uniform(-1, 1),
                              obs_ref[2]], dtype=float)
    vObsNext = np.array([-1, -1, 0])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic


def obstacle10(time_now, time_step):
    # 初始位置（中心点）
    obs_ref = np.array([16, 10, 5], dtype=float)
    final_pos = np.array([16, 20, 8], dtype=float)
    dic = {}
    # 计算震荡和扰动
    y_offset = 0.5 * np.sin(time_now)  # y方向的sin震荡，震荡幅度可以调整
    z_offset = 3 * np.sin(time_now)  # z方向的sin震荡，震荡幅度可以调整
    perturbation = np.array([0, random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])

    # 计算当前位置
    obsCenter = obs_ref + np.array([0, y_offset, z_offset]) + perturbation

    # 更新速度矢量
    vObs = np.array([0, 1, 0])  # y方向上逐渐向正方向移动

    # 更新时间
    time_now += time_step

    # 计算下一个位置
    y_offset_next = 0.5 * np.sin(time_now)
    z_offset_next = 3 * np.sin(time_now)
    perturbation_next = np.array([0, random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])
    obsCenterNext = obs_ref + np.array([0, y_offset_next, z_offset_next]) + perturbation_next

    vObsNext = np.array([0, 1, 0])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic


def obstacle11(time_now, time_step):
    obs_ref = np.array([10, 15, 8], dtype=float)  # 初始参考位置
    dic = {}
    time_thre = 10  # 时间阈值

    if time_now < time_thre:
        # 在时间阈值之前的运动路径
        obsCenter = np.array([
            obs_ref[0] + 2 * np.sin(np.pi / 3 + 0.2 * time_now) + random.uniform(0.1, 0.2),
            obs_ref[1] + 2 * np.cos(np.pi / 3 + 0.2 * time_now) + random.uniform(0.1, 0.2),
            obs_ref[2] + 3 * np.sin(0.5 * time_now) + random.uniform(0.1, 0.2)
        ], dtype=float)

        vObs = np.array([
            0.4 * np.cos(np.pi / 3 + 0.2 * time_now),
            -0.4 * np.sin(np.pi / 3 + 0.2 * time_now),
            0.6 * np.cos(0.5 * time_now)
        ])

        time_now += time_step

        obsCenterNext = np.array([
            obs_ref[0] + 2 * np.sin(np.pi / 3 + 0.2 * time_now) + random.uniform(0.1, 0.2),
            obs_ref[1] + 2 * np.cos(np.pi / 3 + 0.2 * time_now) + random.uniform(0.1, 0.2),
            obs_ref[2] + 3 * np.sin(0.5 * time_now) + random.uniform(0.1, 0.2)
        ], dtype=float)

        vObsNext = np.array([
            0.4 * np.cos(np.pi / 3 + 0.2 * time_now),
            -0.4 * np.sin(np.pi / 3 + 0.2 * time_now),
            0.6 * np.cos(0.5 * time_now)
        ])

    else:
        # 在时间阈值之后的运动路径
        delta_time = time_now - time_thre
        time_cal = time_thre - delta_time

        obsCenter = np.array([
            obs_ref[0] + 2 * np.sin(np.pi / 3 + 0.2 * time_cal) + random.uniform(0.1, 0.2),
            obs_ref[1] + 2 * np.cos(np.pi / 3 + 0.2 * time_cal) + random.uniform(0.1, 0.2),
            obs_ref[2] + 3 * np.sin(0.5 * time_cal) + random.uniform(0.1, 0.2)
        ], dtype=float)

        vObs = np.array([
            0.4 * np.cos(np.pi / 3 + 0.2 * time_cal),
            -0.4 * np.sin(np.pi / 3 + 0.2 * time_cal),
            0.6 * np.cos(0.5 * time_cal)
        ])

        time_now += time_step
        time_cal -= time_step

        obsCenterNext = np.array([
            obs_ref[0] + 2 * np.sin(np.pi / 3 + 0.2 * time_cal) + random.uniform(0.1, 0.2),
            obs_ref[1] + 2 * np.cos(np.pi / 3 + 0.2 * time_cal) + random.uniform(0.1, 0.2),
            obs_ref[2] + 3 * np.sin(0.5 * time_cal) + random.uniform(0.1, 0.2)
        ], dtype=float)

        vObsNext = np.array([
            0.4 * np.cos(np.pi / 3 + 0.2 * time_cal),
            -0.4 * np.sin(np.pi / 3 + 0.2 * time_cal),
            0.6 * np.cos(0.5 * time_cal)
        ])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic


def obstacle12(time_now, time_step):
    start_pos = np.array([12, 10, 5], dtype=float)
    end_pos = np.array([12, 20, 8], dtype=float)
    dic = {}
    total_time = 20  # 总时间
    progress = min(time_now / total_time, 1.0)  # 计算当前的进度

    # 在路径中间转圈的复杂运动
    radius = 2  # 转圈的半径
    angle = 2 * np.pi * progress * 3  # 通过进度控制角度，乘以3是为了多转几圈
    z_offset = (end_pos[2] - start_pos[2]) * progress  # z方向的线性上升

    obsCenter = np.array([
        start_pos[0] + radius * np.cos(angle),  # x方向的转圈运动
        start_pos[1] + radius * np.sin(angle) + 10 * progress,  # y方向的转圈运动，叠加一个线性上升
        start_pos[2] + z_offset  # z方向的上升
    ], dtype=float)

    # 添加扰动
    obsCenter += np.array([
        random.uniform(0.1, 0.2),  # x方向扰动
        random.uniform(0.1, 0.2),  # y方向扰动
        random.uniform(0.1, 0.2)  # z方向扰动
    ])

    # 计算速度矢量
    vObs = np.array([
        -radius * np.sin(angle) * 2 * np.pi * 3 / total_time,  # x方向速度
        radius * np.cos(angle) * 2 * np.pi * 3 / total_time + (end_pos[1] - start_pos[1]) / total_time,  # y方向速度
        (end_pos[2] - start_pos[2]) / total_time  # z方向速度
    ])

    time_now += time_step
    progress_next = min(time_now / total_time, 1.0)

    # 下一时刻的位置
    angle_next = 2 * np.pi * progress_next * 3
    z_offset_next = (end_pos[2] - start_pos[2]) * progress_next

    obsCenterNext = np.array([
        start_pos[0] + radius * np.cos(angle_next),
        start_pos[1] + radius * np.sin(angle_next) + 10 * progress_next,
        start_pos[2] + z_offset_next
    ], dtype=float)

    # 添加扰动
    obsCenterNext += np.array([
        random.uniform(0.1, 0.2),  # x方向扰动
        random.uniform(0.1, 0.2),  # y方向扰动
        random.uniform(0.1, 0.2)  # z方向扰动
    ])

    vObsNext = np.array([
        -radius * np.sin(angle_next) * 2 * np.pi * 3 / total_time,
        radius * np.cos(angle_next) * 2 * np.pi * 3 / total_time + (end_pos[1] - start_pos[1]) / total_time,
        (end_pos[2] - start_pos[2]) / total_time
    ])

    dic['v'] = vObs
    dic['obsCenter'] = obsCenter
    dic['vNext'] = vObsNext
    dic['obsCenterNext'] = obsCenterNext

    return time_now, dic

"""生成一个函数列表"""
obs_list = [obstacle1, obstacle2, obstacle3, obstacle4,
            obstacle5, obstacle6, obstacle7, obstacle8,
            obstacle9, obstacle10,obstacle11,obstacle12]
