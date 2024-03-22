import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_dice(image1, image2):
    """
    计算两个图像之间的Dice系数。
    """
    intersection = np.logical_and(image1, image2)
    dice = 2 * np.sum(intersection) / (np.sum(image1) + np.sum(image2))
    return dice

def read_and_preprocess_image(filepath):
    """
    读取和预处理图像。
    """
    image = Image.open(filepath)
    image = image.convert('1')  # 转换为二值图像
    image_array = np.array(image)
    return image_array

# 指定包含所有子文件夹的文件夹路径
parent_folder_path = r'./LIDC-IDRI'

# 遍历每个子文件夹
total_dice_matrix = None
num_folders = 0

for folder in os.listdir(parent_folder_path):
    folder_path = os.path.join(parent_folder_path, folder)
    print(folder_path)
    if os.path.isdir(folder_path):
        # 读取子文件夹中的所有图像
        image_files = os.listdir(folder_path)
        image_files.sort()
        expert_segmentations = [read_and_preprocess_image(os.path.join(folder_path, file)) for file in image_files]

        # 计算子文件夹的Dice系数矩阵
        dice_matrix = np.zeros((len(expert_segmentations), len(expert_segmentations)))

        for i in range(len(expert_segmentations)):
            for j in range(len(expert_segmentations)):
                if i == j:
                    # 相同标注图像的Dice系数为1
                    dice_matrix[i, j] = 1.0
                else:
                    dice = calculate_dice(expert_segmentations[i], expert_segmentations[j])
                    dice_matrix[i, j] = dice

        # 累加Dice系数矩阵
        if total_dice_matrix is None:
            total_dice_matrix = dice_matrix
        else:
            total_dice_matrix += dice_matrix

        num_folders += 1

# 计算平均Dice系数矩阵
average_dice_matrix = total_dice_matrix / num_folders

# 确定最佳一致性
average_dice = np.mean(average_dice_matrix, axis=1)
best_segmentation_index = np.argmax(average_dice)


# 生成专家标签
expert_labels = ["R1", "R2", "R3", "R4"]

plt.figure(figsize=(10, 8))
sns.heatmap(average_dice_matrix, annot=True, cmap='coolwarm', vmin=0.4, vmax=1,
            xticklabels=expert_labels, yticklabels=expert_labels)
plt.title('Average Dice Matrix Heatmap')
plt.xlabel('Expert Index')
plt.ylabel('Expert Index')


heatmap_path = r'./average_dice_heatmap_lung.png'
plt.savefig(heatmap_path)

print("Average Dice Matrix:")
print(average_dice_matrix)
print("Index of the best segmentation based on highest average Dice:")
print(best_segmentation_index)
print(f"Heatmap saved at: {heatmap_path}")
