import matplotlib.pyplot as plt
import numpy as np

# 读取你的图像文件
image = plt.imread('/home/liu/ForDeep/ForDeep/data/duck/images/duck_14.jpg')

# 创建子图
fig, ax = plt.subplots()

# 绘制图像
ax.imshow(image)

# 绘制网格线
for i in range(0, 720, 16):
    ax.axhline(i, color='w', linestyle='--')
    ax.axvline(i, color='w', linestyle='--')

# 显示图像
plt.show()