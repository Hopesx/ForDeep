import open3d
import numpy as np
from deep2 import*
# 创建可视化窗口并添加射线对象
vis = open3d.visualization.Visualizer()
vis.create_window()
point = []
a = Deep()
print(a.data_folder)
center = np.array([0.0, 0.0, 0.0])
for i in range(5):
    rays = a.get_ray(i)
    for ray in rays:
        origin = ray.origin
        direction = ray.direction * 4
        endpoint = origin + direction
        center += endpoint
        # print("origin", origin)
        # print("endpoint", endpoint)
        # 创建Open3D的LineSet对象
        lineset = open3d.geometry.LineSet()
        lineset.points = open3d.utility.Vector3dVector([origin, endpoint])
        lineset.lines = open3d.utility.Vector2iVector([[0, 1]])
        lineset.colors = open3d.utility.Vector3dVector([[i / 10, 0, 0]])
        vis.add_geometry(lineset)

center = center / 27 / 40 / 22
print("center", center)
# 长方体边长
edge_lengths = [5, 5, 5]
# 长方体中心
center = [0.5, 1.1, 1.5]

# 创建坐标系
coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])

# 创建长方体顶点
x, y, z = center
points = np.array([
    [x - edge_lengths[0]/2, y - edge_lengths[1]/2, z - edge_lengths[2]/2],
    [x + edge_lengths[0]/2, y - edge_lengths[1]/2, z - edge_lengths[2]/2],
    [x - edge_lengths[0]/2, y + edge_lengths[1]/2, z - edge_lengths[2]/2],
    [x + edge_lengths[0]/2, y + edge_lengths[1]/2, z - edge_lengths[2]/2],
    [x - edge_lengths[0]/2, y - edge_lengths[1]/2, z + edge_lengths[2]/2],
    [x + edge_lengths[0]/2, y - edge_lengths[1]/2, z + edge_lengths[2]/2],
    [x - edge_lengths[0]/2, y + edge_lengths[1]/2, z + edge_lengths[2]/2],
    [x + edge_lengths[0]/2, y + edge_lengths[1]/2, z + edge_lengths[2]/2]
])

# 创建长方体线段
lines = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
line_set = open3d.geometry.LineSet()
line_set.points = open3d.utility.Vector3dVector(points)
line_set.lines = open3d.utility.Vector2iVector(lines)
line_set.paint_uniform_color([0, 0, 0])  # 设置颜色
vis.add_geometry(line_set)
vis.add_geometry(coord_frame)
# 添加三维坐标系
vis.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(size=1))
# 运行可视化窗口
vis.run()
vis.destroy_window()