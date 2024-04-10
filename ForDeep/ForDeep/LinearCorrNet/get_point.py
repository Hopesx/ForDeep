import torch
import numpy as np
import os
from torch import nn
from torch.nn import functional as F

from PIL import Image
from image_enconder import ImageEncoderViT
from typing import Any, Dict, List, Tuple

class GetFeature(nn.Module):
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

    def load(self, image_path):
        # 逐张读取图像
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
        # 获取图像特征张量
        print("image_features")
        input= self.load(image_path)
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



if __name__ == "__main__":
    # 获取图像特征
    a = GetFeature()
    image_folder = '/home/liu/ForDeep/ForDeep/data/vase2/images'
    image_files = os.listdir(image_folder)  # 获取图像文件列表
    sorted_files = sorted(image_files)

    # sorted_files = sorted(image_files, key=get_numeric_part)  # 按照数字顺序排序
    # print("sorted_files", sorted_files)
    # features = []
    # for i in range(22):
    #     print(i, "图")
    #     image_file = sorted_files[i]
    #     image_path = os.path.join(image_folder, image_file)  # 构建图像文件的完整路径
    #     print("image_path", image_path)
    #     feature, _ = a.image_features(image_path)
    #     print(feature.shape)
    #     features.append(feature)


    # 打开文件
    with open('/home/liu/ForDeep/ForDeep/data/vase2/images.txt', 'r') as file:
        lines = file.readlines()  # 逐行读取文件内容

    # 初始化存储数据的字典
    point_data = {}

    # 处理文件内容
    for i in range(5, len(lines), 2):  # 从第5行开始，因为前面是图像信息
        image_info = lines[i-1].strip().split(' ')  # 图像信息
        point_info = lines[i].strip().split(' ')  # 像素坐标和3D点ID信息

        # 获取图片ID
        image_id = int(image_info[0])
        for j in range(2, len(point_info), 3):
            # 提取3D点ID
            point3d_id = int(point_info[j])  # 得到3D点ID
            x = float(point_info[j - 2])  # 获取x坐标
            y = float(point_info[j - 1])  # 获取y坐标

            # 存储数据
            if point3d_id != -1:
                if image_id not in point_data:
                    point_data[image_id] = {'point_ids': [point3d_id], 'pixel_coords': [(x, y)]}
                else:
                    point_data[image_id]['point_ids'].append(point3d_id)
                    point_data[image_id]['pixel_coords'].append((x, y))

    print(len(point_data))

    # 初始化存储结果的字典
    result = {}

    # 遍历 point_data 字典
    for image_id, data in point_data.items():
        #提取特征
        print("image_id", image_id - 1)
        print(image_id - 1, "图")
        image_file = sorted_files[image_id - 1]
        image_path = os.path.join(image_folder, image_file)  # 构建图像文件的完整路径
        print("image_path", image_path)
        feature= a.image_features(image_path)
        print(feature.shape)
        # print(feature)
        # f = np.zeros((1, 256))
        # print("Point3D_ID:", point3d_id)
        # print("Image IDs:", data['image_ids'])
        # print("Pixel Coordinates:")
        for point_id, coord in zip(data['point_ids'], data['pixel_coords']):
            # print(coord)
            x = int(coord[0] / 16)
            y = int(coord[1] / 16)
            # print(coord)
            # print(x, ' ', y)
            # feature = features[image_id - 1]
            # if np.array_equal(f, np.zeros((1, 256))) == True:
            #     f = np.reshape(feature[0, :, y, x].detach().numpy(), (1, 256))
            # else:
            #     f = np.vstack((f, np.reshape(feature[0, :, y, x].detach().numpy(), (1, 256))))
            if x < feature.shape[3] and y < feature.shape[2]:
                if point_id not in result:
                    result[point_id] = {'features': [np.reshape(feature[0, :, y, x].detach().numpy(), (1, 256))]}
                else:
                    result[point_id]['features'].append(np.reshape(feature[0, :, y, x].detach().numpy(), (1, 256)))

    #写入文件
    # 打开文件以写入数据
    with open('/home/liu/ForDeep/ForDeep/data/point_lin/features_data_vase2.txt', 'w') as file:
        for point_id, data in result.items():
            for vector in data['features']:
                # print("vector.shape", vector.shape)
                line = ' '.join(str(num) for num in vector[0])  # 将向量转换为字符串并用空格分隔
                file.write(line + ' ')  # 写入每个向量，并在末尾加上空格
            file.write(line + '\n')  # 写入每个向量，并在末尾加上换行符

    print("文件已成功写入.")






    # 打印结果
    # for point_id, data in point_data.items():
    #     print(f"3D Point ID: {point_id}")
    #     for image_id, (x, y) in zip(data['image_ids'], data['pixel_coords']):
    #         print(f"   Image ID: {image_id}, Pixel coordinates - ({x}, {y})")
    # 打印point3d_id为1597的数据信息
    # i = 6775
    # for image_id, data in point_data.items():
    #     for point_id, (x, y) in zip(data['point_ids'], data['pixel_coords']):
    #         if point_id == i:
    #             print(f"   Image ID: {image_id}, Pixel coordinates - ({x}, {y})")
    # else:
    #     print("No data found for 3D Point ID: {i}")