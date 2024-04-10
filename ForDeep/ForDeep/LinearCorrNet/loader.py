# import torch
# from torch.utils.data import Dataset, DataLoader
#
# class MatrixDataset(Dataset):
#     def __init__(self, file_path):
#         self.matrices = []
#         with open(file_path, 'r') as file:
#             lines = file.readlines()
#             for line in lines:
#                 matrix_data = [float(x) for x in line.split()]
#                 matrix = torch.tensor(matrix_data).reshape(-1, 256)
#                 if matrix.shape[0] >= 4:
#                     self.matrices.append(matrix[:4])
#
#     def __len__(self):
#         return len(self.matrices)
#
#     def __getitem__(self, idx):
#         return self.matrices[idx]
# def custom_collate_fn(batch):
#     # 在collate_fn函数中处理不同大小的数据，不进行填充
#     return batch
#
# def get_loader(batch_size, data_dir, num_workers=16):
#     # 读取数据集
#     dataset = MatrixDataset(data_dir)
#
#     # 创建数据加载器
#
#     # 创建数据加载器，并指定collate_fn参数为自定义的collate函数
#     data_loader = DataLoader(dataset,
#                              batch_size=batch_size,
#                              collate_fn=custom_collate_fn,
#                              shuffle=False,
#                              num_workers=num_workers)
#
#     # # 使用数据加载器进行迭代
#     # for batch in data_loader:
#     #     for item in batch:
#     #         print(item.shape)  # 打印每个样本的张量形状
#
#     return data_loader
#
# # 定义文件路径
# file_path = '/home/liu/ForDeep/ForDeep/data/point_lin/features_data_duck.txt'
#
# # 创建自定义数据集实例
# dataset = MatrixDataset(file_path)
#
# # 创建 DataLoader，设置 batch_size=1，即每个矩阵作为一个 batch
# data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
#
# # 遍历 DataLoader，每次取出一个矩阵作为一个 batch
# print(len(data_loader))
# for batch in data_loader:
#     print(batch.shape)  # 输出当前 batch 的形状


import torch
from torch.utils.data.dataloader import Dataset, DataLoader
import numpy
import random

class MatrixDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip().split()
                data_row = [list(map(float, line[i:i+256]))[:256] for i in range(0, len(line), 256)]
                if len(data_row) >= 4:
                    self.data.extend(data_row[:4])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx])

    def shuffle_data(self):
        # 将 self.data 划分为连续的四个元素为一组的子列表
        grouped_data = [self.data[i:i + 4] for i in range(0, len(self.data), 4)]

        # 随机打乱这些子列表
        random.shuffle(grouped_data)

        # 将打乱后的子列表重新组合成一个新的列表
        shuffled_data = [item for sublist in grouped_data for item in sublist]

        # 更新 self.data
        self.data = shuffled_data

def get_loader(batch_size, data_dir, num_workers=16):
    # 读取数据集
    dataset = MatrixDataset(data_dir)
    dataset.shuffle_data()
    remaining_data = len(dataset) % batch_size
    if remaining_data > 0:
        new_len = len(dataset) - remaining_data
        dataset = dataset[:new_len]  # 舍去多余的数据

    # 创建数据加载器

    # 创建数据加载器，并指定collate_fn参数为自定义的collate函数
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)

    # 使用数据加载器进行迭代
    # for batch in data_loader:
    #     print(batch.shape)  # 打印每个样本的张量形状
        # print(batch)

    return data_loader

# file_path = '/home/liu/ForDeep/ForDeep/data/point_lin/features_data_duck.txt'
# train_loader = get_loader(16, file_path)
# for batch_idx, data in enumerate(train_loader):
#     print(data.shape)
#     reshaped_x_numpy = data.detach().numpy()
#
#     # 对 reshaped_data 进行 SVD 分解
#     U, S, V = numpy.linalg.svd(reshaped_x_numpy, full_matrices=False)
#     U1, S1, V1 = torch.svd(data)
#     f1 = S[1] / S[0]
#     f2 = S[2] / S[0]
#     f3 = S[3] / S[0]
#     f11 = S1[1] / S1[0]
#     f21 = S1[2] / S1[0]
#     f31 = S1[3] / S1[0]
#     print(f1)
#     print(f11)
#     print(f2)
#     print(f21)
