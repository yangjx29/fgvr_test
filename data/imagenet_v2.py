import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import defaultdict

# 提示词定义
how_to1 = """This is an ImageNetV2 image classification task. ImageNetV2 is a new test set for ImageNet with 1000 different object categories ranging from animals to vehicles to household items. Each image belongs to exactly one of these 1000 categories.

Your task is to identify the specific object category shown in the image. Be precise and specific - ImageNetV2 uses fine-grained categories (e.g., specific dog breeds, car models, bird species).

Key guidelines:
- Look for distinctive features of the object
- Consider the object's shape, color, texture, and context
- Be specific about the category (e.g., "golden retriever" not just "dog")
- If uncertain, choose the most likely category among the 1000 options
- Focus on the main object, ignore background elements

The image contains one of the 1000 ImageNet categories. Identify it accurately."""

how_to2 = """ImageNetV2 Classification Task:

This dataset contains 1,000 diverse object categories including:
- Animals (mammals, birds, insects, fish)
- Vehicles (cars, airplanes, boats)
- Household items (furniture, appliances)
- Food items
- Natural objects (plants, minerals)
- Tools and instruments

Your task: Identify the specific ImageNetV2 category shown in the image.

Instructions:
1. Examine the main object carefully
2. Consider fine-grained details (breed, model, species)
3. Choose from the 1000 predefined categories
4. Be as specific and accurate as possible
5. Ignore background distractions

The image belongs to exactly one of the 1000 ImageNetV2 categories. Provide the most accurate classification."""

# 导出提示词别名
imagenet_v2_how_to1 = how_to1
imagenet_v2_how_to2 = how_to2

class ImageNetV2Prompter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.how_to = how_to1
    
    def get_prompt(self):
        return self.how_to

class ImageNetV2Discovery:
    def __init__(self, cfg, folder_suffix=''):
        # 使用JSON文件加载数据
        json_path = os.path.join(cfg['expt_dir'], 'images_split', f'images_discovery_all{folder_suffix}.json')
        print(f"构建ImageNetV2发现集,json_path: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")
        
        # 加载JSON文件
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.samples = []
        self.targets = []
        self.subcategories = []
        
        # 解析JSON数据 - 格式: [class_name, class_id, image_list]
        for class_data in self.data:
            class_name = class_data[0]
            class_id = class_data[1]
            image_list = class_data[2]
            
            # 为每个图片创建样本
            for img_file in image_list:
                # 检查img_file是否已经是完整路径
                if img_file.startswith('./') or img_file.startswith('/') or os.path.isabs(img_file):
                    # 已经是完整路径，直接使用
                    img_path = img_file
                elif img_file.startswith('datasets/'):
                    # 是相对datasets目录的路径，直接使用
                    img_path = f"./{img_file}"
                else:
                    # 是其他相对路径，需要拼接基础路径
                    img_path = os.path.join(cfg['data_dir'], img_file)
                self.samples.append(img_path)
                self.subcategories.append(class_name)
                self.targets.append(class_id)
        
        # 创建子类别到样本的映射
        self.subcat_to_sample = defaultdict(list)
        for subcat, sample in zip(self.subcategories, self.samples):
            self.subcat_to_sample[subcat].append(sample)
    
    def __getitem__(self, index):
        img_path = self.samples[index]
        subcategory = self.subcategories[index]
        img = Image.open(img_path).convert('RGB')
        return img, subcategory, img_path
    
    def __len__(self):
        return len(self.samples)

class ImageNetV2Dataset(Dataset):
    def __init__(self, cfg, split='test'):
        self.cfg = cfg
        self.split = split
        
        # 加载JSON文件
        json_path = os.path.join(cfg['expt_dir'], 'images_split', f'images_{split}.json')
        print(f"构建ImageNetV2 {split}集, json_path: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.samples = []
        self.targets = []
        self.class_names = []
        
        # 解析JSON数据
        for class_data in self.data:
            class_name = class_data[0]
            class_id = class_data[1]
            image_list = class_data[2]
            
            for img_file in image_list:
                # 处理路径
                if img_file.startswith('./') or img_file.startswith('/') or os.path.isabs(img_file):
                    img_path = img_file
                elif img_file.startswith('datasets/'):
                    img_path = f"./{img_file}"
                else:
                    img_path = os.path.join(cfg['data_dir'], img_file)
                
                self.samples.append(img_path)
                self.targets.append(class_id)
                self.class_names.append(class_name)
        
        self.transform = _transform(self.cfg, self.split)
    
    def __getitem__(self, index):
        img_path = self.samples[index]
        target = self.targets[index]
        class_name = self.class_names[index]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, target, class_name, img_path
    
    def __len__(self):
        return len(self.samples)

def _transform(cfg, split='test'):
    """数据变换函数"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# 构建函数
def build_imagenet_v2_prompter(cfg):
    return ImageNetV2Prompter(cfg)

def build_imagenet_v2_discovery(cfg, folder_suffix=''):
    return ImageNetV2Discovery(cfg, folder_suffix)

def build_imagenet_v2_test(cfg):
    return ImageNetV2Dataset(cfg, split='test')

def build_imagenet_v2_swav_train(cfg):
    return ImageNetV2Dataset(cfg, split='train')

def build_imagenet_v2_val(cfg):
    return ImageNetV2Dataset(cfg, split='val')
