from PIL import Image
import os

# 指定输入和输出文件夹路径
input_folder = "/home/liu/ForDeep/ForDeep/data/test_1/images"
output_folder = "/home/liu/ForDeep/ForDeep/data/test_1/images_2"

# 指定目标宽度和高度
target_width = 640
target_height = 360

# 循环处理每个图像文件
for filename in os.listdir(input_folder):
    # 检查文件是否为图像文件
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 打开图像文件
        image = Image.open(os.path.join(input_folder, filename))
        # 调整图像大小
        resized_image = image.resize((target_width, target_height))
        # 保存调整后的图像到输出文件夹
        resized_image.save(os.path.join(output_folder, filename))