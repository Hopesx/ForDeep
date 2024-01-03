import numpy as np

def are_rays_intersecting(P1, D1, P2, D2):
    # 计算参数方程中的系数矩阵
    A = np.vstack((D1, -D2)).T
    B = P2 - P1

    # 解方程得到参数 t1 和 t2
    try:
        t1, t2 = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        # 矩阵 A 不可逆，无法求解，射线平行或重合
        return False

    # 检查参数范围
    if t1 >= 0 and t2 >= 0:
        return True
    else:
        return False


def check_grid_overlap(ray1, ray2, threshold):
    # 检查两个网格是否重叠，即两个射线是否有交点
    # 如果有交点且交点与原点距离小于阈值，则返回交点坐标
    # 提取点
    # print("ray1",ray1)
    A1 = ray1[:3]
    A2 = ray1[3:6]
    B1 = ray2[:3]
    B2 = ray2[3:6]
    # 计算两个点之间的射线方向向量
    v1 = A2 - A1
    v2 = B2 - B1

    intersection = np.zeros(3)
    if np.dot(v1, v2) == 1:
        # 两线平行
        return intersection

    startPointSeg = B1 - A1
    vecS1 = np.cross(v1, v2)  # 有向面积1
    vecS2 = np.cross(startPointSeg, v2)  # 有向面积2
    num = np.dot(startPointSeg, vecS1)

    # 判断两这直线是否共面
    if num >= 1e-5 or num <= -1e-5:
        return intersection

    # 有向面积比值，利用点乘是因为结果可能是正数或者负数
    num2 = np.dot(vecS2, vecS1) / np.dot(vecS1, vecS1)
    intersection = A1 + v1 * num2
    if np.dot((intersection - A2), v1) > 0 and np.dot((intersection - B2), v2) > 0:
        # 交点在A2，B2后面，即交点在归一化平面外
        return intersection
    else:
        return np.zeros(3)


def line_line_intersection(p1, v1, p2, v2):
    intersection = np.zeros(3)
    if np.dot(v1, v2) == 1:
        # 两线平行
        return False

    startPointSeg = p2 - p1
    vecS1 = np.cross(v1, v2)  # 有向面积1
    vecS2 = np.cross(startPointSeg, v2)  # 有向面积2
    num = np.dot(startPointSeg, vecS1)

    # 判断两这直线是否共面
    if num >= 1e-5 or num <= -1e-5:
        return False

    # 有向面积比值，利用点乘是因为结果可能是正数或者负数
    num2 = np.dot(vecS2, vecS1) / np.dot(vecS1, vecS1)
    intersection = p1 + v1 * num2
    print("交点", intersection)
    return True






A1 = np.array([1, -1, 1])
A2 = np.array([2, 1, 2.25])
B1 = np.array([-1, 1, 0])
B2 = np.array([0, 2, 1])
v1 = np.array([1, 2, 1.25])
v2 = np.array([1, 1, 1])
print("result", check_grid_overlap(np.append(A1, A2), np.append(B1, B2), 1))
# line_line_intersection(A1, v1, B1, v2)
