# 导入数学运算模块
import math
# 导入操作系统接口模块
import os

# 导入JSON处理模块
import json
# 导入随机数生成模块
import random
# 导入NumPy数值计算库
import numpy as np
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的数据集基类
from torch.utils.data import Dataset
# 导入PIL图像处理库
import PIL
from PIL import Image


# 基于JSON配置文件的数据集基类
# 用于加载通过JSON文件定义数据划分的数据集
class BaseJsonDataset(Dataset):
    # 初始化数据集
    # 参数:
    #   image_path: 图像文件所在的根目录路径
    #   json_path: JSON配置文件路径，包含数据划分信息
    #   mode: 数据集模式，'train'（训练）或'test'（测试），默认为'train'
    #   n_shot: 少样本学习中每类的样本数，None表示使用该模式下的全部数据
    #   transform: 图像变换/增强操作，默认为None
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform  # 保存图像变换操作
        self.image_path = image_path  # 保存图像根目录
        self.split_json = json_path  # 保存JSON配置文件路径
        self.mode = mode  # 保存数据集模式
        self.image_list = []  # 初始化图像路径列表
        self.label_list = []  # 初始化标签列表
        # 从JSON文件中读取数据划分信息
        with open(self.split_json) as fp:
            splits = json.load(fp)  # 加载JSON文件
            samples = splits[self.mode]  # 获取当前模式（train/test）的样本
            # 遍历样本，提取图像路径和标签
            for s in samples:
                self.image_list.append(s[0])  # s[0]是图像相对路径
                self.label_list.append(s[1])  # s[1]是类别标签
    
        # 如果指定了少样本数量，则进行少样本采样
        if n_shot is not None:
            few_shot_samples = []  # 存储少样本的索引
            c_range = max(self.label_list) + 1  # 计算类别总数
            # 对每个类别进行采样
            for c in range(c_range):
                # 找出属于类别c的所有样本索引
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)  # 设置随机种子以保证可重复性
                # 从该类别中随机采样n_shot个样本
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            # 根据采样的索引更新图像列表和标签列表
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    # 返回数据集的大小
    # 返回:
    #   int: 数据集中样本的总数
    def __len__(self):
        return len(self.image_list)

    # 获取指定索引的数据样本
    # 参数:
    #   idx: 样本索引
    # 返回:
    #   tuple: (image, label) 图像张量和标签张量
    def __getitem__(self, idx):
        # 构建完整的图像路径
        image_path = os.path.join(self.image_path, self.image_list[idx])
        # 加载图像并转换为RGB格式
        image = Image.open(image_path).convert('RGB')
        # 获取对应的标签
        label = self.label_list[idx]
        # 如果定义了变换操作，则应用到图像上
        if self.transform:
            image = self.transform(image)
        
        # 返回图像和标签（标签转换为长整型张量）
        return image, torch.tensor(label).long()

# 支持少样本学习的数据集列表
# 这些数据集都有预定义的train/test划分，可用于少样本学习实验
fewshot_datasets = ['dtd', 'oxford_flowers', 'food101', 'stanford_cars', 'sun397', 
                'fgvc_aircraft', 'oxford_pets', 'caltech101', 'ucf101', 'eurosat',
                'caltech256', 'cub', 'birdsnap', 'stanford_dogs']  # ========== 新增: Stanford Dogs ==========

# 构建少样本学习数据集的工厂函数
# 参数:
#   set_id: 数据集标识符（如'oxford_flowers', 'food101'等）
#   root: 数据集根目录路径
#   transform: 图像变换/增强操作
#   mode: 数据集模式，'train'或'test'，默认为'train'
#   n_shot: 少样本学习中每类的样本数，None表示使用全部数据
# 返回:
#   Dataset: 构建好的数据集对象
def build_fewshot_dataset(set_id, root, transform, mode='train', n_shot=None):

    # 数据集路径配置字典
    # 格式: "数据集名称": ["图像子目录", "JSON配置文件路径"]
    path_dict = {
        "oxford_flowers": ["jpg", root + "/split_zhou_OxfordFlowers.json"],
        "food101": ["images", root + "/split_zhou_Food101.json"],
        "dtd": ["images", root + "/split_zhou_DescribableTextures.json"],
        "oxford_pets": ["images", root + "/split_zhou_OxfordPets.json"],
        "sun397": ["SUN397", root + "/split_zhou_SUN397.json"],
        "caltech101": ["101_ObjectCategories", root + "/split_zhou_Caltech101.json"],
        "ucf101": ["UCF-101-midframes", root + "/split_zhou_UCF101.json"],
        "stanford_cars": ["", root + "/split_zhou_StanfordCars.json"],
        "eurosat": ["2750", root + "/split_zhou_EuroSAT.json"],
        "cub": ["images", root + "/split_CUB.json"],
        "caltech256": ["256_ObjectCategories", root + "/split_Caltech256.json"],
        "birdsnap": ["images", root + "/split_Birdsnap.json"],
        "stanford_dogs": ["", root + "/split_zhou_StanfordDogs.json"],  # ========== 新增: Stanford Dogs 路径配置 ==========
    }

    # FGVC Aircraft数据集使用特殊的数据集类
    if set_id.lower() == 'fgvc_aircraft':
        return Aircraft(root, mode, n_shot, transform)
    # 获取数据集的路径配置
    path_suffix, json_path = path_dict[set_id.lower()]
    # 构建完整的图像目录路径
    image_path = os.path.join(root, path_suffix)
    # 返回基于JSON的数据集对象
    return BaseJsonDataset(image_path, json_path, mode, n_shot, transform)


# FGVC Aircraft（细粒度飞机分类）数据集类
# 该数据集使用特殊的文本文件格式而非JSON，因此需要单独的类来处理
class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    # 初始化Aircraft数据集
    # 参数:
    #   root: 数据集根目录路径
    #   mode: 数据集模式，'train'（训练）、'val'（验证）或'test'（测试），默认为'train'
    #   n_shot: 少样本学习中每类的样本数，None表示使用全部数据
    #   transform: 图像变换/增强操作，默认为None
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform  # 保存图像变换操作
        self.path = root  # 保存数据集根目录
        self.mode = mode  # 保存数据集模式

        # 读取飞机型号类别名称
        self.cname = []  # 初始化类别名称列表
        # 从variants.txt文件中读取所有飞机型号名称
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            # 去除每行的换行符，得到类别名称列表
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []  # 初始化图像文件名列表
        self.label_list = []  # 初始化标签列表
        # 读取当前模式（train/val/test）的图像列表文件
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            # 读取所有行并去除换行符
            lines = [s.replace("\n", "") for s in fp.readlines()]
            # 解析每一行，格式为: "图像ID 飞机型号名称"
            for l in lines:
                ls = l.split(" ")  # 按空格分割
                img = ls[0]  # 第一部分是图像ID
                label = " ".join(ls[1:])  # 其余部分是飞机型号名称（可能包含空格）
                # 添加图像文件名（加上.jpg后缀）
                self.image_list.append("{}.jpg".format(img))
                # 将飞机型号名称转换为类别索引
                self.label_list.append(self.cname.index(label))

        # 如果指定了少样本数量，则进行少样本采样
        if n_shot is not None:
            few_shot_samples = []  # 存储少样本的索引
            c_range = max(self.label_list) + 1  # 计算类别总数
            # 对每个类别进行采样
            for c in range(c_range):
                # 找出属于类别c的所有样本索引
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)  # 设置随机种子以保证可重复性
                # 从该类别中随机采样n_shot个样本
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            # 根据采样的索引更新图像列表和标签列表
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    # 返回数据集的大小
    # 返回:
    #   int: 数据集中样本的总数
    def __len__(self):
        return len(self.image_list)

    # 获取指定索引的数据样本
    # 参数:
    #   idx: 样本索引
    # 返回:
    #   tuple: (image, label) 图像张量和标签张量
    def __getitem__(self, idx):
        # 构建完整的图像路径（所有图像都在images子目录下）
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        # 加载图像并转换为RGB格式
        image = Image.open(image_path).convert('RGB')
        # 获取对应的标签
        label = self.label_list[idx]
        # 如果定义了变换操作，则应用到图像上
        if self.transform:
            image = self.transform(image)
        
        # 返回图像和标签（标签转换为长整型张量）
        return image, torch.tensor(label).long()

