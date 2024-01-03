import torch
from torch import nn
from torch.nn import functional as F

from PIL import Image
import os
import math
from open3d import*
import gc

from typing import Any, Dict, List, Tuple

from image_enconder import ImageEncoderViT

class Deep(nn.Module):
    image_format: str = "RGB"
    def __init__(
            self,
            image_encoder: ImageEncoderViT = ImageEncoderViT(),
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
    )-> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.image_num = 20
        self.image_size = (0, 0)
        self.grid_size = (16,16,16)
        self.num_grids = 0
        self.rays = []
        # 创建记录重叠情况的二维数组
        self.overlaps = []
        # self.poses =


    def load(self):
        image_folder = "/home/liu/ForDeep/ForDeep/data/nerf_llff_data/fortress/images_8"  # 图像文件夹路径
        image_files = os.listdir(image_folder)  # 获取图像文件列表

        batched_input = []  # 创建空的 batched_input 列表
        i = 0

        for image_file in image_files:
            i = i + 1
            image_path = os.path.join(image_folder, image_file)  # 构建图像文件的完整路径
            # print(image_path)

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
            batched_input.append(image_dict)  # 将字典添加到batched_input中
        self.image_num = i
        self.num_grids = (self.image_size[0] // self.grid_size[0]) * (self.image_size[1] // self.grid_size[1])
        print("图像有", self.image_num)
        print("图像尺寸", self.image_size)
        return batched_input
        # 批量读取完成后，batched_input 中存储了所有图像的信息

    def image_features(self):
        batched_input = self.load()
        image_features = []
        # if not batched_input :
        #     print("empty")
        # else :
        #     print("ok")
        # i = 0
        # for x in batched_input :
        #     i = i + 1
        #     input_image = torch.stack([self.preprocess(x["image"]) ], dim=0)  # 预处理图像
        #     feature = self.image_encoder(input_image)  # 编码图像特征
        #     image_features.append(feature)
        #     # print(feature)
        #     print(i)

        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)

        image_features = self.image_encoder(input_images)
        # print("iamge_features",image_features.shape)

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

    def check_grid_overlap(self, ray1, ray2, threshold):
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
            print("平行")
            return intersection

        startPointSeg = B1 - A1
        vecS1 = np.cross(v1, v2)  # 有向面积1
        vecS2 = np.cross(startPointSeg, v2)  # 有向面积2
        num = np.dot(startPointSeg, vecS1) / (np.linalg.norm(startPointSeg) * np.linalg.norm(vecS1))
        # print("num", num)

        # 判断两这直线是否共面
        if num >= 1e-1 or num <= -1e-1:
            # print("不共面")
            return intersection

        # 有向面积比值，利用点乘是因为结果可能是正数或者负数
        num2 = np.dot(vecS2, vecS1) / np.dot(vecS1, vecS1)
        intersection = A1 + v1 * num2
        if np.dot((intersection - A2), v1) > 0 and np.dot((intersection - B2), v2) > 0:
            # 交点在A2，B2后面，即交点在归一化平面外
            return intersection
        else:
            return np.zeros(3)

    def compute_grid_overlap(self, poses, K, threshold):
        # 计算每个网格与其他网格的重叠情况
        # # 获取图像数量和网格数量
        # num_grids = (self.image_size[0] // self.grid_size[0]) * (self.image_size[1] // self.grid_size[1])
        # print("num_grids", num_grids)

        # 计算每张图像的网格并转换到世界坐标系中
        # transformed_grids = []
        print("compute_grid_overlap image_size", self.image_size)
        width, height = self.image_size
        num_horizontal_grids = width // self.grid_size[0]
        num_vertical_grids = height // self.grid_size[1]
        for i in range(self.image_num):
            grids = [(j, k) for j in range(num_horizontal_grids) for k in range(num_vertical_grids)]
            # self.transformed_grids.append([])
            # 构造外参矩阵T
            T = np.zeros((4, 4))
            T[:3, :] = poses[i]
            T[-1, -1] = 1
            transformed_camera_o = np.dot(np.linalg.inv(T), np.array([0, 0, 0, 1]))
            for grid in grids:
                # #像素坐标系转换到世界坐标系，设Z=1
                # x = np.dot(np.linalg.inv(K), np.array([grid[0] * self.grid_size[0], grid[1] * self.grid_size[1], 1]))
                # transformed_grid = np.dot(np.linalg.inv(T), x)
                # 像素坐标系转换到相机坐标系
                u = grid[0] * self.grid_size[0]
                v = grid[1] * self.grid_size[1]
                print("uv", u, " ", v)
                camera_uv = np.array([u - K[0, 2], v - K[1, 2], K[0, 0], 1])
                print("camera_uv", camera_uv)
                #相机坐标系原点转换到世界坐标系
                transformed_camera_uv = np.dot(np.linalg.inv(T), camera_uv)
                print("transformed_camera_uv", transformed_camera_uv)
                print("transformed_camera_o", transformed_camera_o)
                #构造射线
                ray = np.append(transformed_camera_o[:3], transformed_camera_uv[:3])
                print("ray", ray)
                self.rays.append(ray)
        print("射线over")

        # 计算每个网格与其他网格的重叠情况
        for i in range(1):
            for j in range(self.num_grids):
                for k in range(i + 1, self.image_num):
                    for l in range(self.num_grids):
                        num1 = i * num_horizontal_grids * num_vertical_grids + j
                        num2 = k * num_horizontal_grids * num_vertical_grids + l
                        # print(self.rays[num1]," ",  self.rays[num2])
                        result = self.check_grid_overlap(self.rays[num1], self.rays[num2], threshold)
                        # print("交点：", i, ",", j, ",", k, ",", l, ",", result)
                        if np.array_equal(result, np.zeros(3)) == False:
                            self.overlaps.append(np.concatenate(([i, j, k, l], result), axis=0))
                            print("交点：", i, ",", j, ",", k, ",", l, ",",result)
        print("overlap")
        return

    def overlap(self):
        print("overlap start work")
        #判断网格是否重叠
        poses_arr = np.load(os.path.join("/home/liu/ForDeep/ForDeep/data/nerf_llff_data/fortress", 'poses_bounds.npy'))
        # print("poses_arr",poses_arr.shape)
        # poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        poses = poses[:, :3, :4]
        hwf = poses[0, :3, -1] # 取出前三行最后一列元素
        H, W, focal = hwf   # 分别赋予 Hieight、Width、focal
        K = np.array([
            [focal, 0, 0.5 * self.image_size[0]],
            [0, focal, 0.5 * self.image_size[1]],
            [0, 0, 1]
        ])
        # print(hwf)
        print("poses", poses.shape)

        self. compute_grid_overlap(poses, K, 100)
        # print("overlap", overlap.shape)
        # print(overlap)
        return

    def computeFeatures(self, feature1, feature2):
        #计算特征相似度
        # 计算两个张量的点积
        dot_product = torch.dot(feature1, feature2)

        # 计算两个张量的模长
        norm1 = torch.norm(feature1)
        norm2 = torch.norm(feature2)

        # 计算余弦相似度
        cosine_similarity = dot_product / (norm1 * norm2)
        return cosine_similarity

    def compareFeatures(self):
        print("compareFeatures start work")
        #对比特征
        image_features = self.image_features()
        self.overlap()
        # 创建记录特征对比结果的二维数组
        # num_grids = (self.image_size[0] // self.grid_size[0]) * (self.image_size[1] // self.grid_size[1])
        compareFeatures = np.zeros((self.image_num, self.num_grids, self.image_num, self.num_grids), dtype=float)
        # compareFeatures = np.zeros(self.image_num, num_grids, self.image_num, num_grids)

        for overlap in self.overlaps:
            x = overlap[:4].astype(int)
            # point = overlap[4:7]
            x1 = x[1] // ((image_features.size())[3])
            y1 = x[1] - x1 * ((image_features.size())[3])
            x2 = x[3] // ((image_features.size())[3])
            y2 = x[3] - x2 * ((image_features.size())[3])
            compareFeatures[x[0], x[1], x[2], x[3]] = self.computeFeatures(image_features[x[0], :, x1, y1],
                                                               image_features[x[2], :, x2, y2])
            compareFeatures[x[2], x[3], x[0], x[1]] = compareFeatures[x[0], x[1], x[2], x[3]]
        # for i in range(self.image_num):
        #     for j in range(self.num_grids):
        #         for k in range(i + 1, self.image_num):
        #             for l in range(self.num_grids):
        #                 # print("i,k",i,k)
        #                 # print("x1,y1,x2,y2",x1,y1,x2,y2)
        #                 # if overlap[i, j, k, l, :] != (0, 0, 0):
        #                 if np.array_equal(self.overlaps[i, j, k, l, :], [0, 0, 0]) == False:
        #                     x1 = j // ((image_features.size())[3])
        #                     y1 = j - x1 * ((image_features.size())[3])
        #                     x2 = l // ((image_features.size())[3])
        #                     y2 = l - x2 * ((image_features.size())[3])
        #                     # print("feature_size", image_features[i, :, x1, y1].size())
        #                     compareFeatures[i, j, k, l] = self.computeFeatures(image_features[i, :, x1, y1], image_features[k, :, x2, y2])
        #                     # print("transformed_grids", transformed_grids[i][j])
        #                     # print("compareFeatures",compareFeatures[i,j,k,l])
        #                     compareFeatures[k, l, i, j] = compareFeatures[i, j, k, l]
        return compareFeatures

    def sameGrids(self):
        print("sameGrids start work")
        compareFeatures = self.compareFeatures()
        # 判断每张图每个网格特征比对最相似的网格
        sameGrids = np.zeros((self.image_num, self.num_grids, 2), dtype=int)
        for i in range(self.image_num):
            for j in range(self.num_grids):
                minComFeatures = 0.5   #阈值设为0.5
                for k in range(i + 1, self.image_num):
                    for l in range(self.num_grids):
                        if compareFeatures[i, j, k, l] < minComFeatures and compareFeatures[i, j, k, l] != 0:
                            minComFeatures = compareFeatures[i, j, k, l]
                            sameGrids[i, j] = (k, l)
                            print("sameGrids", i, ",", j, ":", k, ",", l)
        # print("sameGrids", sameGrids)
        return sameGrids

    def deep(self):
        print("deep start work")
        sameGrids = self.sameGrids()
        # overlap = self.overlap()
        deep = []
        for overlap in self.overlaps:
            x = overlap[:4].astype(int)
            point = overlap[4:7]
            if np.array_equal(sameGrids[x[0], x[1]], x[2:4]):
                deep.append(point)
        # for i in range(self.image_num):
        #     for j in range(self.num_grids):
        #         k, l = sameGrids[i,j]
        #         if not k == 0 and not l == 0:
        #             deep.append(self.overlaps[i, j, k, l, :])
        print("deep",deep)
        return deep

    def visualization(self):
        print("visualization start work")
        deep = self.deep()
        points_array = np.array(deep)
        gc.collect() #回收垃圾
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





a = Deep()
# # a.load()
# # a.image_features()
# # a.overlap()
# # a.compareFeatures()
# # a.sameGrids()
# # a.deep()
a.visualization()






