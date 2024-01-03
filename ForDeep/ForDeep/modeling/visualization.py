import numpy as np
from deep import Deep
from open3d import*


def visualization():
    print("visualization start work")
    a = Deep()
    deep = a.deep()
    del a
    points_array = np.array(deep)
    print("point_size", points_array.size)
    print("points",points_array)

    del deep
    # gc.collect()  # 回收垃圾
    # deep = []
    # a = [1, 2, 3]
    # b = [3.89, 3.09, 5.9]
    # deep.append(a)
    # deep.append(b)
    # points_array = np.array(deep)
    # 将点坐标转换成 Open3D 的 PointCloud 对象
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points_array)

    # 创建 Visualizer 和 ViewControl
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    # 添加点云到场景中
    vis.add_geometry(pcd)

    # 设置渲染参数并显示点云
    opt = vis.get_render_option()
    opt.point_size = 5  # 点的大小
    opt.background_color = np.asarray([0, 0, 0])  # 背景颜色

    vis.run()
    vis.destroy_window()

visualization()