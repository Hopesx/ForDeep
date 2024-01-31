import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from PIL import Image
import re
import os
import math
from open3d import *
import gc
import trimesh
from multiprocessing import Pool
from functools import partial
import itertools

from typing import Any, Dict, List, Tuple

from image_enconder import ImageEncoderViT

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #代表只使用第0个gpu


class SpatialGrid:
    #空间网格
    def __init__(self):
        self.features = None
        self.isEntity = False
        self.loc = None
        self.f1 = 100
        self.f2 = 100
        self.f3 = 100
        self.S = np.zeros(1)


    def add_features(self, feature):
        # if np.array_equal(self.features, np.zeros((1, 256))) == True:
        if self.features is None:
            self.features = feature
        else:
            self.features = np.vstack((self.features, feature))

    def local(self, local):
        self.loc = local

    def get_loc(self):
        return self.loc

class Ray:
    #射线
    def __init__(self, origin, direction, num):
        self.origin = origin  # 射线起点坐标 (x, y, z)
        self.direction = direction  # 射线方向向量 (dx, dy, dz)
        self.num = num


class Deep(nn.Module):
    image_format: str = "RGB"
    def __init__(
            self,
            image_encoder: ImageEncoderViT = ImageEncoderViT(720, 720),
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
    )-> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.image_num = 6
        self.image_size = (640, 360)
        self.grid_size = (16, 16)
        self.SpatialGrid_size = (0.04, 0.04, 0.04)
        self.num_grids = 0 #图像网格数
        self.rays = [] #射线list
        self.num_SpatialGrids = (50, 50, 70)  # 空间网格数
        num_SGrids = self.num_SpatialGrids[0] * self.num_SpatialGrids[1] * self.num_SpatialGrids[2]
        self.SpatialGrids = [SpatialGrid() for _ in range(num_SGrids)]
        self.SpatialGrids = [] #空间网格,记录经过射线的特征张量
        # self.overlaps = []
        self.center = (0, 0, 0)
        self.data_folder = "/home/liu/ForDeep/ForDeep/data/duck"  # 数据文件夹路径



    def load(self, image_path):
        #逐张读取图像
        # image_folder = "/home/liu/ForDeep/ForDeep/data/nerf_llff_data/fortress/images_8"  # 图像文件夹路径
        # image_files = os.listdir(image_folder)  # 获取图像文件列表
        image = Image.open(image_path).convert("RGB")  # 读取图像并转换为RGB格式

        # 将图像转换为torch张量，并添加到字典中
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()  # 转换为形状为3xHxW的torch张量
        original_size = image.size  # 获取图像的原始尺寸
        self.image_size = image.size

        # 创建包含图像信息的字典
        image_dict = {
            'image': image_tensor,
            'original_size': original_size
        }
        return image_dict

    def image_features(self, image_path):
        #获取图像特征张量
        print("image_features")
        input = self.load(image_path)
        # input_image = self.preprocess(x["image"])
        input_images = torch.stack([self.preprocess(input["image"])], dim=0)
        image_features = self.image_encoder(input_images)
        print("image_features_size", image_features.size())
        return image_features

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_height - h
        padw = self.image_encoder.img_width - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def get_ray(self, i):
        #获取射线
        print("get_ray start work")
        poses_arr = np.load(os.path.join(self.data_folder, 'poses_bounds.npy'))
        # print("poses_arr",poses_arr.shape)
        # poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        hwf = poses[0, :3, -1]  # 取出前三行最后一列元素
        poses = poses[:, :3, :4]

        H, W, focal = hwf  # 分别赋予 Hieight、Width、focal
        print("H", H)
        print("W", W)
        print("focal", focal)
        K = np.array([
            [focal, 0, 0.5 * self.image_size[0]],
            [0, focal, 0.5 * self.image_size[1]],
            [0, 0, 1]
        ])
        print("K", K)
        # print(hwf)
        print("poses", poses[0])

        rays = self.compute_ray(i, poses, K)

        return rays

    def compute_ray(self, i, poses, K):
        width, height = self.image_size
        num_horizontal_grids = width // self.grid_size[0]
        num_vertical_grids = height // self.grid_size[1]
        # for i in range(self.image_num):
        grids = [(j, k) for j in range(num_vertical_grids) for k in range(num_horizontal_grids)]
        # self.transformed_grids.append([])
        # 构造外参矩阵T
        T = np.zeros((4, 4))
        T[:3, :] = poses[i, :, :]
        # print("poses", poses[i, :, :])
        T[-1, -1] = 1
        print("T:", T)
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]
        # 相机坐标系转换到世界坐标系
        # transformed_camera_o = np.dot(np.linalg.inv(T), np.array([[0], [0], [0], [1]]))
        transformed_camera_o = np.dot(T, np.array([[0], [0], [0], [1]]))
        print("transformed_camera_o", transformed_camera_o[:3])
        print("t", t)
        rays = []  # 射线list
        num = 0
        for grid in grids:
            # print("grid", grid)
            # #像素坐标系转换到世界坐标系，设Z=1
            # x = np.dot(np.linalg.inv(K), np.array([grid[0] * self.grid_size[0], grid[1] * self.grid_size[1], 1]))
            # transformed_grid = np.dot(np.linalg.inv(T), x)
            # 像素坐标系转换到相机坐标系
            u = grid[0] * self.grid_size[0]
            v = grid[1] * self.grid_size[1]

            uv = np.array([[u], [v], [1]])
            # print("uv", uv)
            # camera_uv = np.dot(np.linalg.inv(K), uv)
            # camera_uv = np.vstack((camera_uv, [1]))
            camera_uv = np.array([(u - K[0, 2]) / K[0, 0], -((v - K[1, 2]) / K[0, 0]), -1])
            print("camera_uv", camera_uv)
            # 相机坐标系转换到世界坐标系
            transformed_camera_uv = np.dot(T[:3, :3], camera_uv)
            # transformed_camera_uv = np.dot(np.linalg.inv(R), uv).T + t
            # print("transformed_camera_uv", transformed_camera_uv)
            # print("transformed_camera_o", transformed_camera_o)
            # 构造射线
            # print("transformed_camera_uv", transformed_camera_uv)
            # direction = np.squeeze(transformed_camera_uv[:3] - transformed_camera_o[:3])
            # origin = np.squeeze(transformed_camera_o[:3])
            direction = transformed_camera_uv[:3]
            origin = np.squeeze(transformed_camera_o[:3])

            # print("direction", direction)
            # print("origin", origin)
            ray = Ray(origin, direction, num)
            # print("ray", ray.num)
            # print("o:", origin, " direction:", direction)
            rays.append(ray)
            num = num + 1
        print("射线over")
        return rays

    def get_Spatial(self):
        rays = []
        for i in self.image_num:
            rays.append(self.get_ray(i))



    def get_SpatialGrid_Features(self):
        print("get_SpatialGrid_Features start work")
        # 获取经过空间网格的射线代表的特征张量
        # 图像网格数
        self.num_grids = (self.image_size[0] // self.grid_size[0]) * (self.image_size[1] // self.grid_size[1])
        # 初始化空间网格
        num_SGrids = self.num_SpatialGrids[0] * self.num_SpatialGrids[1] * self.num_SpatialGrids[2]
        self.SpatialGrids = [SpatialGrid() for _ in range(num_SGrids)]
        halflenth = round(self.SpatialGrid_size[0] * self.num_SpatialGrids[0] / 2, 2)
        for k in range(num_SGrids):
            z = k // (self.num_SpatialGrids[0] * self.num_SpatialGrids[1])
            xy = k - z * self.num_SpatialGrids[0] * self.num_SpatialGrids[1]
            x = xy % self.num_SpatialGrids[0]
            y = xy // self.num_SpatialGrids[0]

            x = x + round((self.center[0] - halflenth) / self.SpatialGrid_size[0])
            y = y + round((self.center[1] - halflenth) / self.SpatialGrid_size[1])
            z = z + round((self.center[2] - halflenth) / self.SpatialGrid_size[2])
            local = np.array([x, y, z])
            # print("local", local)
            # print("k", k, " ", local)
            self.SpatialGrids[k].local(local)

        # 获取图像特征
        image_folder = os.path.join(self.data_folder, 'images')
        image_files = os.listdir(image_folder)  # 获取图像文件列表
        sorted_files = sorted(image_files)
        # sorted_files = sorted(image_files, key=get_numeric_part)  # 按照数字顺序排序
        # print("sorted_files", sorted_files)
        for i in range(self.image_num):
            print(i, "图")
            image_file = sorted_files[i]
            image_path = os.path.join(image_folder, image_file)  # 构建图像文件的完整路径
            print("image_path", image_path)
            features = self.image_features(image_path)
            # 获取射线
            rays = self.get_ray(i)
            for ray in rays:
                self.get_SpatialGrid(features, ray)
        # for k in range(num_SGrids):
        #     if self.SpatialGrids[k].loc is not None:
        #         print("k:", k, " ", self.SpatialGrids[k].loc)
        return

    def get_SpatialGrid(self, features, ray, start=None):
        # 获取射线经过的空间网格
        width, height = self.image_size
        num_horizontal_grids = width // self.grid_size[0]
        num_vertical_grids = height // self.grid_size[1]
        direction = ray.direction
        halflenth0 = round(self.SpatialGrid_size[0] * self.num_SpatialGrids[0] / 2, 2)
        halflenth1 = round(self.SpatialGrid_size[1] * self.num_SpatialGrids[1] / 2, 2)
        halflenth2 = round(self.SpatialGrid_size[2] * self.num_SpatialGrids[2] / 2, 2)
        # print("halflenth", halflenth)
        x_min = round((self.center[0] - halflenth0) / self.SpatialGrid_size[0])
        x_max = round((self.center[0] + halflenth0) / self.SpatialGrid_size[0])
        y_min = round((self.center[1] - halflenth1) / self.SpatialGrid_size[1])
        y_max = round((self.center[1] + halflenth1) / self.SpatialGrid_size[1])
        z_min = round((self.center[2] - halflenth2) / self.SpatialGrid_size[2])
        z_max = round((self.center[2] + halflenth2) / self.SpatialGrid_size[2])
        min = np.array([x_min, y_min, z_min])
        max = np.array([x_max - 1, y_max - 1, z_max - 1])
        # print("min", min)
        # print("max", max)
        # print((self.center[2] - halflenth) // self.SpatialGrid_size[2])
        local = np.array([x_min, y_min, z_min])
        if start is None:
            # print("start is none")
            # 找到射线经过的第一个网格
            find = False
            for i in range(3):
                if direction[i] > 0:
                    local[i] = min[i]
                else:
                    local[i] = max[i]
                for local[(i + 1) % 3], local[(i + 2) % 3] in itertools.product(range(min[(i + 1) % 3], max[(i + 1) % 3]), range(min[(i + 2) % 3], max[(i + 2) % 3])):
                    k = int(local[0] - min[0] + (local[1] - min[1]) * self.num_SpatialGrids[0] + (local[2] - min[2]) * self.num_SpatialGrids[0] * self.num_SpatialGrids[1])
                    # self.SpatialGrids[k].loc = local
                    # print("k", k)

                    if self.is_ray_intersect_grid(ray, local):
                        print("loc", local)
                        # 射线经过空间网格
                        find = True
                        print(ray.num, "网格经过", self.SpatialGrids[k].loc[0], ",", self.SpatialGrids[k].loc[1], ",", self.SpatialGrids[k].loc[2], flush=True)
                        feature = features[0, :, ray.num // num_horizontal_grids, ray.num % num_horizontal_grids]
                        # print("feature_size", np.reshape(feature.detach().numpy(), (1, 256)).shape)
                        self.SpatialGrids[k].add_features(np.reshape(feature.detach().numpy(), (1, 256)))
                        self.get_SpatialGrid(features, ray, local)
                        return
                if find:
                    return
        else:
            local = start
            # print("start", start)
            for i in range(3):
                # print("i", i)
                local[i] = local[i] + int(direction[i] / abs(direction[i]))
                # print("x1", x1)

                if local[i] >= min[i] and local[i] <= max[i]:
                    k = int(local[0] - min[0] + (local[1] - min[1]) * self.num_SpatialGrids[0] + (local[2] - min[2]) *
                            self.num_SpatialGrids[0] * self.num_SpatialGrids[1])
                    # print("k", local[0] - min[0] + (local[1] - min[1]) * self.SpatialGrid_size[0] + (local[2] - min[2]) *
                    #         self.SpatialGrid_size[0] * self.SpatialGrid_size[1])
                    # print("loc", self.SpatialGrids[k].loc)
                    # self.SpatialGrids[k].loc = local
                    if self.is_ray_intersect_grid(ray, local):
                        # 射线经过空间网格
                        # print("local", local)
                        # print(ray.num)
                        # print("ray.num % num_vertical_grids", ray.num % num_vertical_grids)
                        # print("ray.num // num_vertical_grids", ray.num // num_vertical_grids)
                        feature = features[0, :, ray.num // num_horizontal_grids, ray.num % num_horizontal_grids]
                        self.SpatialGrids[k].add_features(np.reshape(feature.detach().numpy(), (1, 256)))
                        print(ray.num, "网格经过", self.SpatialGrids[k].loc[0], ",", self.SpatialGrids[k].loc[1], ",", self.SpatialGrids[k].loc[2], flush=True)
                        self.get_SpatialGrid(features, ray, local)
                        return "over"
                    else:
                        local[i] = local[i] - int(direction[i] / abs(direction[i]))
                else:
                    local[i] = local[i] - int(direction[i] / abs(direction[i]))
        return "over"

    def is_ray_intersect_grid(self, ray, loc):
        x = loc[0]
        y = loc[1]
        z = loc[2]

        # 计算网格六个面的边界坐标
        x_min = x * self.SpatialGrid_size[0]
        x_max = (x + 1) * self.SpatialGrid_size[0]
        y_min = y * self.SpatialGrid_size[1]
        y_max = (y + 1) * self.SpatialGrid_size[1]
        z_min = z * self.SpatialGrid_size[2]
        z_max = (z + 1) * self.SpatialGrid_size[2]

        # 计算射线与每个面的交点
        t_xmin = (x_min - ray.origin[0]) / ray.direction[0]
        t_xmax = (x_max - ray.origin[0]) / ray.direction[0]
        t_ymin = (y_min - ray.origin[1]) / ray.direction[1]
        t_ymax = (y_max - ray.origin[1]) / ray.direction[1]
        t_zmin = (z_min - ray.origin[2]) / ray.direction[2]
        t_zmax = (z_max - ray.origin[2]) / ray.direction[2]

        # 找到最小和最大的交点参数值
        t_min = max(min(t_xmin, t_xmax), min(t_ymin, t_ymax), min(t_zmin, t_zmax))
        t_max = min(max(t_xmin, t_xmax), max(t_ymin, t_ymax), max(t_zmin, t_zmax))

        # 如果最小交点参数值小于最大交点参数值，则射线与网格相交
        # if t_min > 0 and t_max > 0:
        #     return t_min < t_max
        # else:
        #     return False
        return t_max > max(t_min, 0.0)

    def deep(self):
        self.get_SpatialGrid_Features()
        deep = []
        num_SpatialGrids = self.num_SpatialGrids[0] * self.num_SpatialGrids[1] * self.num_SpatialGrids[2]
        # print("num_SpatialGrids", num_SpatialGrids)
        for k in range(num_SpatialGrids):
            # print("k", k)
            # print("features_size", self.SpatialGrids[k].features.shape)
            # self.SpatialGrids[k].is_Entity(self.SVD(self.SpatialGrids[k].features)
            # if np.array_equal(self.SpatialGrids[k].features, np.zeros((1, 256))) == False:
            if self.SpatialGrids[k].features.shape[0] > 3:
                # print(k, "Sgrid", self.SpatialGrids[k].features)
                if self.SVD(k, 0.5):
                    # print("loc", self.SpatialGrids[k].loc)
                    deep.append(self.SpatialGrids[k].loc)
        print("deep", deep)
        self.visualization(deep, True, False)
        return deep

    def SVD(self, k, threshold):
        U, S, VT = np.linalg.svd(self.SpatialGrids[k].features)
        # print("U ", U)
        # print("S ", S)
        # print("VT ", VT)
        # if S.shape
        f1 = S[1] / S[0]
        f2 = S[2] / S[1]
        f3 = S[3] / S[2]
        self.SpatialGrids[k].S = S
        self.SpatialGrids[k].f1 = f1
        self.SpatialGrids[k].f2 = f2
        self.SpatialGrids[k].f3 = f3
        if S[-1] < 1:
            self.SpatialGrids[k].isEntity = True
            return True
        else:
            return False

        # if (S[1] / S[0]) < threshold:
        #     self.SpatialGrids[k].isEntity = True
        #     print("f1", f1)
        #     print("f2", f2)
        #     print("f3", f3)
        #     return True
        #     # if(S[2] / S[1]) < 0.5:
        #     #     print("f1", f1)
        #     #     print("f2", f2)
        #     #     return True
        #     # else:
        #     #     return False
        # else:
        #     return False

    def visualization(self, point, iswrite, isread):
        print("visualization start work")
        # deep = self.deep()
        if isread:
            point = self.read()
        if point is None:
            print("empty")
        # print(point)
        points_array = np.array(point)
        # print(points_array)
        print("points", points_array.shape)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points_array)
        # open3d.visualization.draw_geometries([pcd])

        if iswrite:
            self.write(pcd)

        # 创建 Visualizer 和 ViewControl
        vis = open3d.visualization.Visualizer()
        vis.create_window(visible=True)

        # 添加点云到场景中
        vis.add_geometry(pcd)

        # 创建一个自定义的坐标系
        custom_coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=20)

        # 将坐标系沿着需要设置为原点的位置平移
        custom_coordinate_frame.translate(np.array([0, 0, 0]))

        # 添加三维坐标系
        # vis.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(size=20))
        vis.add_geometry(custom_coordinate_frame)

        # 设置渲染参数并显示点云
        opt = vis.get_render_option()
        opt.point_size = 10  # 点的大小
        opt.background_color = np.asarray([0, 0, 0])  # 背景颜色

        vis.run()
        vis.destroy_window()

    def write(self, pcd):
        # 写入文件
        open3d.io.write_point_cloud(self.data_folder + "/point_cloud5_half.ply", pcd)
        # with open(self.data_folder + "/svd5_half.txt", "w") as file:
        #     for grid in self.SpatialGrids:
        #         if grid.isEntity:
        #             file.write(f"{grid.f1:.6f} {grid.f2:.6f} {grid.f3:.6f}\n")
        with open(self.data_folder + "/S.txt", "w") as file:
            for grid in self.SpatialGrids:
                if grid.isEntity:
                    vector_str = ' '.join(str(s) for s in grid.S)
                    file.write(vector_str + '\n')

    def read(self):
        # 从PLY文件加载点云数据
        ply = open3d.io.read_point_cloud(self.data_folder + "/point_cloud9_5>.ply")
        # 将点云数据转换为 NumPy 数组
        points = np.asarray(ply.points)
        i = 0
        rpoint = []
        # 读取文件中的向量

        with open(self.data_folder + "/S.txt", "r") as file:
            # 循环读取文件中的每一行
            for line in file:
                # 将每行的字符串按空格分割成数字，并转换为整数列表
                vector = [float(x) for x in line.split()]
                # vectors.append(vector)
                if vector[-1] < 0.5 :
                    print("f1", vector[-1])
                    print("f2", vector[-2])
                    print("f3", vector[-3])
                    print(points[i])
                    rpoint.append(points[i])
                i = i + 1
        pointcloud = np.asarray(rpoint)
        return pointcloud


if __name__ == "__main__":
    a = Deep()
    # a.get_SpatialGrid_Features()
    # # a.load()
    # # a.image_features()
    # # a.overlap()
    # # a.compareFeatures()
    # # a.sameGrids()
    # a.deep()
    # for i in range(22):
    #     a.get_ray(i)
    a.visualization(None, False, True)
    # a.image_features("/home/liu/ForDeep/ForDeep/data/test_1/images_2/test_1.jpg")
    # a.get_ray(0)
    # 创建可视化窗口并添加射线对象
    # vis = open3d.visualization.Visualizer()
    # vis.create_window()
    # point = []
    # for i in range(15, 26):
    #     rays = a.get_ray(i)
    #     for ray in rays:
    #         origin = ray.origin
    #         direction = ray.direction
    #         endpoint = origin + direction
    #         print("origin", origin)
    #         print("endpoint", endpoint)
    #         # 创建Open3D的LineSet对象
    #         lineset = open3d.geometry.LineSet()
    #         lineset.points = open3d.utility.Vector3dVector([origin, endpoint])
    #         lineset.lines = open3d.utility.Vector2iVector([[0, 1]])
    #         lineset.colors = open3d.utility.Vector3dVector([[i / 30, 0, 0]])
    #         vis.add_geometry(lineset)
    #
    # # 立方体棱长
    # edge_len = 0.5
    # # 立方体中心
    # center = [0, -0.6, -0.4]
    #
    # # 创建坐标系
    # coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    #
    # # 创建立方体线段
    # points = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    #                    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) * edge_len / 2 + center
    # lines = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    # line_set = open3d.geometry.LineSet()
    # line_set.points = open3d.utility.Vector3dVector(points)
    # line_set.lines = open3d.utility.Vector2iVector(lines)
    # line_set.paint_uniform_color([0.8, 0.8, 0.8])  # 设置颜色
    # vis.add_geometry(line_set)
    # vis.add_geometry(coord_frame)
    # # 添加三维坐标系
    # vis.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(size=1))
    # # 运行可视化窗口
    # vis.run()
    # vis.destroy_window()




