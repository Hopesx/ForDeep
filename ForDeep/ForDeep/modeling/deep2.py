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
from sklearn.cluster import KMeans

from typing import Any, Dict, List, Tuple

from image_enconder import ImageEncoderViT
from ForDeep.LinearCorrNet.vae_pytorch import *
from torch.utils.data import Dataset, DataLoader

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #代表只使用第0个gpu


class SpatialGrid:
    #空间网格
    def __init__(self):
        self.features = np.zeros((1, 256))
        self.colors = np.zeros((1, 3))
        self.isEntity = False
        self.loc = None
        self.f1 = 100
        self.f2 = 100
        self.f3 = 100


    def add_features(self, feature):
        if np.array_equal(self.features, np.zeros((1, 256))) == True:
            self.features = feature
        else:
            self.features = np.vstack((self.features, feature))

    def add_colors(self, color):
        if np.array_equal(self.colors, np.zeros((1, 3))) == True:
            self.colors = color
        else:
            self.colors = np.vstack((self.colors, color))

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
        self.image_num = 20
        self.image_size = (720, 720)
        self.grid_size = (16, 16)
        self.SpatialGrid_size = (0.06, 0.06, 0.06)
        self.num_grids = 0 #图像网格数
        self.rays = [] #射线list
        self.num_SpatialGrids = (70, 70, 70)  # 空间网格数
        num_SGrids = self.num_SpatialGrids[0] * self.num_SpatialGrids[1] * self.num_SpatialGrids[2]
        self.SpatialGrids = [SpatialGrid() for _ in range(num_SGrids)]
        self.SpatialGrids = [] #空间网格,记录经过射线的特征张量
        # self.overlaps = []
        self.center = (0.2, 1.3, 0.8) #vase
        # self.center = (0.5, 0, 1.5)  #vase2
        self.data_folder = "/home/liu/ForDeep/ForDeep/data/vase"  # 数据文件夹路径
        self.feature_model = "/home/liu/ForDeep/ForDeep/LinearCorrNet/vae_model4/vae_model4.pth" # 特征处理模型路径



    def load(self, image_path):
        #逐张读取图像
        # image_folder = "/home/liu/ForDeep/ForDeep/data/nerf_llff_data/fortress/images_8"  # 图像文件夹路径
        # image_files = os.listdir(image_folder)  # 获取图像文件列表
        image = Image.open(image_path).convert("RGB")  # 读取图像并转换为RGB格式

        # 读取网格内图像颜色（直接将图像缩小n倍）
        smaller_image = image.resize((image.width // self.grid_size[0], image.height // self.grid_size[0]), Image.LANCZOS)
        colors = torch.tensor(np.array(smaller_image)).permute(2, 0, 1).float()  # 转换为形状为3xHxW的torch张量
        # colors = torch.tensor(np.array(image)).permute(2, 0, 1).float()  # 转换为形状为3xHxW的torch张量

        # 将图像转换为torch张量，并添加到字典中
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()  # 转换为形状为3xHxW的torch张量
        original_size = image.size  # 获取图像的原始尺寸
        self.image_size = image.size

        # 创建包含图像信息的字典
        image_dict = {
            'image': image_tensor,
            'original_size': original_size
        }
        return image_dict, colors

    def image_features(self, image_path):
        # 获取图像特征张量
        print("image_features")
        input, colors = self.load(image_path)
        # input_image = self.preprocess(x["image"])
        input_images = torch.stack([self.preprocess(input["image"])], dim=0)
        image_features = self.image_encoder(input_images)
        print("image_features_size", image_features.size())
        # 特征处理
        reshaped_features = image_features.reshape(256, -1).T
        # reshaped_features = reshaped_features.reshape(image_features.size[2] * image_features.size[3], 256)
        random_index = np.random.randint(0, image_features.shape[2] * image_features.shape[3])
        # 输出原始特征和转换后的某一点特征向量
        # print("原始特征：", image_features[:, :, random_index // image_features.shape[2], random_index % image_features.shape[2]])
        # print("转换后的某一点特征向量：", reshaped_features[random_index])
        #
        # print("reshaped_features.size", reshaped_features.shape)
        sliced_features = []
        for i in range(0, len(reshaped_features), batch_size):
            if reshaped_features[i:i + batch_size].shape[0] < 16:
                padding_rows = 16 - reshaped_features[i:i + batch_size].shape[0]
                padding = torch.ones((padding_rows, reshaped_features[i:i + batch_size].shape[1]))
                reshaped_features = torch.cat((reshaped_features, padding), dim=0)
            sliced_features.append(reshaped_features[i:i + batch_size])
        print("sliced_features.size", sliced_features[-1].shape)
        # sliced_features = reshaped_features.view(-1, 16, 256)
        # VAE模型
        model.load_state_dict(torch.load(self.feature_model))
        model.eval()  # 设置模型为评估模式
        new_features = np.empty((1, 256))
        for i in range(len(sliced_features)):
            with torch.no_grad():
                recon_features, mu, logvar = model(sliced_features[i].reshape(-1, input_dim))
                new_features = np.vstack((new_features, recon_features.reshape(16, 256)))
        print("new_features_size", new_features.shape)
        new_features = torch.tensor(new_features)
        new_features = new_features[:image_features.size(2) * image_features.size(3)]
        new_features = new_features.reshape(image_features.size())

        print("new_features_size", new_features.size())
        # print("color_size", colors.size())
        return new_features, colors

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

    def get_ray(self, i, boundary=False):
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

        rays = self.compute_ray(i, poses, K, boundary)

        return rays

    def compute_ray(self, i, poses, K, boundary):
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
        if boundary:
            grids_boundary = [(0, 0), (0,)]
        else:
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
                # print("camera_uv", camera_uv)
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
        # for i in range(self.image_num):
        for i in range(self.image_num):
            print(i, "图")
            image_file = sorted_files[i]
            image_path = os.path.join(image_folder, image_file)  # 构建图像文件的完整路径
            print("image_path", image_path)
            features, colors = self.image_features(image_path)
            # 获取射线
            rays = self.get_ray(i)
            for ray in rays:
                self.get_SpatialGrid(features, colors, ray)
        # for k in range(num_SGrids):
        #     if self.SpatialGrids[k].loc is not None:
        #         print("k:", k, " ", self.SpatialGrids[k].loc)
        return

    def get_SpatialGrid(self, features, colors, ray, start=None):
        # 获取射线经过的空间网格
        width, height = self.image_size
        num_horizontal_grids = width // self.grid_size[0]
        num_vertical_grids = height // self.grid_size[1]
        direction = ray.direction
        halflenth0 = round(self.SpatialGrid_size[0] * self.num_SpatialGrids[0] / 2, 2)
        halflenth1 = round(self.SpatialGrid_size[1] * self.num_SpatialGrids[1] / 2, 2)
        halflenth2 = round(self.SpatialGrid_size[2] * self.num_SpatialGrids[2] / 2, 2)
        # print("halflenth", halflenth0)
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
                    color = colors[:, ray.num // num_horizontal_grids, ray.num % num_horizontal_grids]

                    if self.is_ray_intersect_grid(ray, local) and not (all(component >= 210 for component in color)):
                        # print("loc", local)
                        # 射线经过空间网格
                        find = True
                        print(ray.num, "网格经过", self.SpatialGrids[k].loc[0], ",", self.SpatialGrids[k].loc[1], ",", self.SpatialGrids[k].loc[2], flush=True)
                        feature = features[0, :, ray.num // num_horizontal_grids, ray.num % num_horizontal_grids]
                        self.SpatialGrids[k].add_features(np.reshape(feature.detach().numpy(), (1, 256)))

                        self.SpatialGrids[k].add_colors(np.reshape(color.detach().numpy(), (1, 3)))
                        self.get_SpatialGrid(features, colors, ray, local)
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
                        feature = features[0, :, ray.num // num_horizontal_grids, ray.num % num_horizontal_grids]
                        self.SpatialGrids[k].add_features(np.reshape(feature.detach().numpy(), (1, 256)))
                        color = colors[:, ray.num // num_horizontal_grids, ray.num % num_horizontal_grids]
                        self.SpatialGrids[k].add_colors(np.reshape(color.detach().numpy(), (1, 3)))
                        print(ray.num, "网格经过", self.SpatialGrids[k].loc[0], ",", self.SpatialGrids[k].loc[1], ",", self.SpatialGrids[k].loc[2], flush=True)
                        self.get_SpatialGrid(features, colors, ray, local)
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

    def get_color(self, colors):
        # 执行k-means聚类
        k = 3  # 选择一个k值，可以根据实际情况调整
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(colors)

        # 分析聚类结果
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        main_color_index = unique[np.argmax(counts)]

        # 提取最主要的颜色
        main_color = kmeans.cluster_centers_[main_color_index]
        color = (main_color[0] / 255, main_color[1] / 255, main_color[2] / 255)

        print(f"最主要的颜色是: {color}")
        return color

    def deep(self):
        self.get_SpatialGrid_Features()
        deep = []
        colors = []
        num_SpatialGrids = self.num_SpatialGrids[0] * self.num_SpatialGrids[1] * self.num_SpatialGrids[2]
        # print("num_SpatialGrids", num_SpatialGrids)
        for k in range(num_SpatialGrids):
            # print("k", k)
            # print("features_size", self.SpatialGrids[k].features.shape)
            # self.SpatialGrids[k].is_Entity(self.SVD(self.SpatialGrids[k].features)
            # if np.array_equal(self.SpatialGrids[k].features, np.zeros((1, 256))) == False:
            if self.SpatialGrids[k].features.shape[0] > 3:
                # print(k, "Sgrid", self.SpatialGrids[k].features)
                if self.SVD(k, 0.9):
                    # print("loc", self.SpatialGrids[k].loc)
                    deep.append(self.SpatialGrids[k].loc)
                    colors.append(self.get_color(self.SpatialGrids[k].colors))
        # print("deep", deep)
        self.visualization(deep, colors, True, False)
        return deep

    def SVD(self, k, threshold):
        f1 = 100
        f2 = 100
        f3 = 100
        try:
            U, S, VT = np.linalg.svd(self.SpatialGrids[k].features)
            f1 = S[1] / S[0]
            f2 = S[2] / S[0]
            f3 = S[3] / S[0]
        except np.linalg.LinAlgError:
            print("SVD did not converge. Skipping...")

        # print("U ", U)
        # print("S ", S)
        # print("VT ", VT)
        # if S.shape

        # try:
        #     f3 = S[3] / S[2]
        #     self.SpatialGrids[k].f3 = f3
        # except IndexError:
        #     print("S[3] 不存在")
        self.SpatialGrids[k].f1 = f1
        self.SpatialGrids[k].f2 = f2
        self.SpatialGrids[k].f3 = f3
        if f1 < threshold:
            self.SpatialGrids[k].isEntity = True
            print("f1", f1)
            print("f2", f2)
            print("f3", f3)
            return True
            # if(S[2] / S[1]) < 0.5:
            #     print("f1", f1)
            #     print("f2", f2)
            #     return True
            # else:
            #     return False
        else:
            return False

    def euclidean_distance(self, color1, color2):
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return np.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)

    def visualization(self, points, colors, iswrite, isread):
        print("visualization start work")
        # deep = self.deep()
        if isread:
            points, colors = self.read()
        if points is None:
            print("empty")
        # print(point)
        points_array = np.array(points)
        colors_array = np.array(colors)
        # points_array = np.empty((1, 3))
        # colors_array = np.empty((1, 3))
        # print(points_array)
        print("points", points_array.shape)

        # for i in range(points.shape[0]):
        #     color1 = (0.7254902, 0.69019608, 0.6627451)
        #     color2 = (0.6483659931257659, 0.62483658136106, 0.6261438107958027)
        #     color3 = (0.81176471, 0.81176471, 0.81176471)
        #     color4 = (0.81960784,0.76862745,0.69411765)
        #
        #     if (self.euclidean_distance(colors[i],color1) >= 0.1 and
        #             self.euclidean_distance(colors[i],color2) >= 0.1 and
        #             self.euclidean_distance(colors[i],color3) >= 0.1 and
        #             self.euclidean_distance(colors[i],color4) >= 0.1):
        #         colors_array = np.vstack([colors_array, colors[i]])
        #         points_array = np.vstack([points_array, points[i]])


        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points_array)
        pcd.colors = open3d.utility.Vector3dVector(colors_array)
        # open3d.io.write_point_cloud(self.data_folder + "/result" + "/duck_big.ply", pcd)

        if iswrite:
            self.write(pcd)

        # 创建 Visualizer 和 ViewControl
        vis = open3d.visualization.Visualizer()
        vis.create_window(visible=True)

        # 添加点云到场景中
        vis.add_geometry(pcd)

        # 创建坐标系
        coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[10, 10, 10])

        # 添加三维坐标系
        # vis.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(size=20))
        vis.add_geometry(coord_frame)

        # 设置渲染参数并显示点云
        opt = vis.get_render_option()
        opt.point_size = 10  # 点的大小
        opt.background_color = np.asarray([0, 0, 0])  # 背景颜色

        vis.run()
        vis.destroy_window()

    def write(self, pcd):
        # 写入文件
        open3d.io.write_point_cloud(self.data_folder + "/result" + "/point_new8.ply", pcd)
        with open(self.data_folder + "/result" + "/svd_new8.txt", "w") as file:
            for grid in self.SpatialGrids:
                if grid.isEntity:
                    file.write(f"{grid.f1:.6f} {grid.f2:.6f} {grid.f3:.6f}\n")

    def read(self):
        # 从PLY文件加载点云数据
        ply = open3d.io.read_point_cloud(self.data_folder + "/result" + "/point_new8.ply")
        # 将点云数据转换为 NumPy 数组
        points = np.asarray(ply.points)
        colors = np.asarray(ply.colors)
        i = 0
        rpoint = []
        rcolor = []
        with open(self.data_folder + "/result" + "/svd_new8.txt", "r") as file:
            # 循环读取文件中的每一行
            for line in file:
                # 处理每一行的数据
                data = line.split()
                f1 = float(data[0])
                f2 = float(data[1])
                f3 = float(data[2])
                if f1 < 0.9:
                    print("f1", f1)
                    print("f2", f2)
                    print("f3", f3)
                    print(points[i])
                    rpoint.append(points[i])
                    if colors.size != 0:
                        rcolor.append(colors[i])
                        print(colors[i])
                i = i + 1
        pointcloud = np.asarray(rpoint)
        colorcloud = np.asarray(rcolor)
        return pointcloud, colorcloud


if __name__ == "__main__":
    a = Deep()
    a.deep()
    # a.image_features('/home/liu/ForDeep/ForDeep/data/duck/images/duck_0.jpg')
    # for i in range(22):
    #     a.get_ray(i)
    # a.visualization(None, None, False, True)




