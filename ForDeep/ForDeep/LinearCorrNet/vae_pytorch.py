import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from ForDeep.LinearCorrNet.loader import get_loader
import torch.nn.functional as F
import numpy
import random
import matplotlib.pyplot as plt
import time
import os


# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出均值和方差
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # 对应数据范围 [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        # 检查生成的矩阵是否包含非有限值，进行处理
        if torch.isnan(recon_x).any() or torch.isinf(recon_x).any():
            recon_x = torch.nan_to_num(recon_x)  # 将NaN替换为0，将无穷大替换为很大或很小的有限值

        return recon_x, mu, logvar


# 参数设置
input_dim = 16 * 256
hidden_dim = 256
latent_dim = 64
batch_size = 16
num_epochs = 20
learning_rate = 3e-4
num_workers = 16
lambda_linde = 1000

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor()
])
train_data_dir = '/home/liu/ForDeep/ForDeep/data/point_lin/combined.txt'
test_data_dir = '/home/liu/ForDeep/ForDeep/data/point_lin/features_data_Cerealboxes2.txt'
train_loader = get_loader(batch_size, train_data_dir, num_workers=num_workers)
test_loader = get_loader(batch_size, test_data_dir, num_workers=num_workers)

# 初始化模型和优化器
model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 定义损失函数
def loss_function(recon_x, x, mu, logvar):

    # 数据线性相关性
    # 将变量 reshape 为 n*256 的二维矩阵
    reshaped_x = recon_x.view(16, 256)
    # print(recon_x)

    # x1 = reshaped_x[:4]
    # x2 = reshaped_x[4:8]
    # x3 = reshaped_x[8:12]
    # x4 = reshaaped_x[-4:]
    # y1 = torch.stack([x1[0], x2[0], x3[0], x4[0]])
    # y2 = torch.stack([x1[1], x2[1], x3[1], x4[1]])
    # y3 = torch.stack([x1[2], x2[2], x3[2], x4[2]])
    # y4 = torch.stack([x1[3], x2[3], x3[3], x4[3]])
    x_slices = [reshaped_x[i * 4:(i + 1) * 4] for i in range(4)]

    # 初始化存储结果的列表
    y_slices = []

    # 使用循环拼接张量
    for i in range(4):
        y_i = torch.stack([x[i] for x in x_slices])
        y_slices.append(y_i)

    fx = 0
    for x in x_slices:
        _, S, _ = torch.svd(x)
        f1 = S[1] / S[0]
        fx += f1

    fy = 0
    for y in y_slices:
        _, S, _ = torch.svd(y)
        f1 = S[1] / S[0]
        fy += f1

    fx = fx / 4
    fy = fy / 4

    fxy = 0
    for i in range(4):
        renum = [num for num in range(4) if num != i]
        rannum = [random.randint(0, 3) for _ in range(3)]
        z = torch.stack(
            (x_slices[renum[0]][rannum[0]], x_slices[renum[0]][rannum[1]], x_slices[renum[1]][rannum[0]], x_slices[renum[1]][rannum[1]], x_slices[renum[2]][rannum[1]], x_slices[renum[2]][rannum[2]]))
        xy = torch.cat((x_slices[i], z), dim=0)
        _, S, _ = torch.svd(xy)
        f1 = S[1] / S[0]
        fxy += f1
        # for x in range(4):
        #     for y in range(4):
        #         for z in range(4):
        #             z = torch.stack((x_slices[renum[0]][x], x_slices[renum[1]][y], x_slices[renum[2]][z]))
        #             xy = torch.cat((x_slices[i], z), dim=0)
        #             _, S, _ = torch.svd(xy)
        #             f1 = S[1] / S[0]
        #             fxy += f1
    fxy = fxy / 4
    # if torch.isnan(fxy):
    #     print("S[0]", S[0])
    #     print("fxy", fxy)

    # 对 reshaped_data 进行 SVD 分解
    # U, S, V = numpy.linalg.svd(reshaped_x_numpy, full_matrices=False)
    # _, S1, _ = torch.svd(x1)
    # _, S2, _ = torch.svd(x2)
    # _, S3, _ = torch.svd(x3)
    # _, S4, _ = torch.svd(x4)
    # _, Sy1, _ = torch.svd(y1)
    # _, Sy2, _ = torch.svd(y2)
    # _, Sy3, _ = torch.svd(y3)
    # _, Sy4, _ = torch.svd(y4)


    # f1 = S1[1] / S1[0]
    # f2 = S1[2] / S1[0]
    # f3 = S1[3] / S1[0]
    # linde_loss = f1
    # 计算线性相关性
    # f1 = S[:, 1] / S[:, 0]
    # f2 = S[:, 2] / S[:, 0]
    # f3 = S[:, 3] / S[:, 0]

    # 求平均线性相关性作为损失
    # linde_loss = torch.mean(f1 + f2 + f3)
    return fx, fy, fxy

def draw(loss_w, linde_w):
    x1 = range(0, 20)
    x2 = range(0, 20)
    y1 = loss_w
    y2 = linde_w
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Train loss vs. epoches')
    plt.ylabel('Train loss')
    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2, '.-')
    # plt.xlabel('Train linde vs. epoches')
    # plt.ylabel('Train linde')
    plt.show()
    plt.savefig("linde_loss.jpg")


if __name__ == "__main__":
    train = True
    test = True
    if train:
        print("train")
        loss_w = []
        linde_w = []
        if os.path.exists('vae_model4/vae_model_epoch10.pth'):  # 检查是否存在检查点文件
            # 加载检查点
            model.load_state_dict(torch.load('vae_model4/vae_model_epoch10.pth'))
            print("Checkpoint loaded!")
        # 训练模型
        for epoch in range(10, num_epochs):
            model.train()
            total_loss = 0
            total_fx = 0
            total_fy = 0
            total_fxy = 0
            # total_kl = 0
            length = 0
            for batch_idx, data in enumerate(train_loader):
                data = data.view(-1, input_dim)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                fx, fy, fxy = loss_function(recon_batch, data, mu, logvar)
                # print("fx:", fx)
                # print("fy:", fy)
                # print("fxy:", fxy)
                # linde_tensor = torch.tensor(linde, dtype=torch.float64, requires_grad=True)
                # print('MSE', MSE)
                # print('KLD', KLD)
                # print('linde', linde)
                # loss = torch.sum(KLD) + torch.sum(MSE)
                # loss = torch.abs(linde_tensor)  # 使用这个比例的差的绝对值作为损失
                if not (torch.isnan(fx) and torch.isnan(fy) and torch.isnan(fxy)):
                    loss = fx - fy + fxy
                    loss.backward()
                    total_loss += loss.item()

                    total_fx += fx
                    total_fy += fy
                    total_fxy += fxy
                    length += 1
                    optimizer.step()
                # print(
                #     'Epoch {}, Loss: {:.4f}, MSE: {:.4f}, linde: {:.4f}'.format(epoch + 1, loss.item(), MSE,
                #                                                                 linde))
            # loss_w.append(total_loss / length)
            # linde_w.append(total_linde / len(train_loader))
            print('Epoch {}, Loss: {:.4f}, fx: {:.4f}, fy: {:.4f}, fxy: {:.4f}'.format(epoch + 1, total_loss / length,
                                                                          total_fx / length, total_fy / length,
                                                                                       total_fxy / length))
            if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次
                torch.save(model.state_dict(), 'vae_model4/vae_model_epoch{}.pth'.format(epoch + 1))
            time.sleep(0.003)
        torch.save(model.state_dict(), 'vae_model4/vae_model4.pth')
        # draw(loss_w, linde_w)

    if test:
        print("test")
        # 加载训练好的模型参数
        model.load_state_dict(torch.load('vae_model4/vae_model4.pth'))

        model.eval()  # 将模型设置为评估模式

        total_test_loss = 0
        total_test_fx = 0
        total_test_fy = 0
        total_test_fxy = 0
        length = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = data.view(-1, input_dim)
                recon_batch, mu, logvar = model(data)
                test_fx, test_fy, test_fxy = loss_function(recon_batch, data, mu, logvar)
                if not(torch.isnan(test_fx) and torch.isnan(test_fy) and torch.isnan(test_fxy)):
                    test_loss = test_fx - test_fy + test_fxy
                    total_test_loss += test_loss.item()
                    total_test_fx += test_fx
                    total_test_fy += test_fy
                    total_test_fxy += test_fxy
                    print("fx:", test_fx)
                    print("fy:", test_fy)
                    print("fxy:", test_fxy)
                    length = length + 1

        print('Test set: Average loss: {:.4f}, fx: {:.4f}, fy: {:.4f}, fxy: {:.4f}'.format(total_test_loss / length,
                                                                      total_test_fx / length, total_test_fy / length,
                                                                                          total_test_fxy / length))
        print("len", length)
        print(len(test_loader))
