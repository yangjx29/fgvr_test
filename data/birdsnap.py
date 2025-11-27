"""
BirdSnap数据集模块
包含500个鸟类类别，支持训练集、测试集和验证集
"""

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict

# 超类和类别单元定义
SUPERCLASS = 'bird'
CLASSUNIT = 'species'

class BirdsnapPrompter:
    """BirdSnap数据集的提示词生成器"""
    
    def __init__(self):
        self.supercategory = "bird"
        self.first_question = "general"
        self.attributes = [
            'size', 'shape', 'color_pattern', 'beak_type', 'wing_shape', 
            'tail_shape', 'habitat', 'behavior', 'plumage', 'markings',
            'head_features', 'body_features', 'leg_color', 'flight_pattern',
            'vocalization', 'seasonal_appearance'
        ]
    
    def get_attributes(self):
        list_attributes = ['General Description']
        list_attributes.extend(self.attributes)
        return list_attributes
    
    def get_attribute_prompt(self):
        list_prompts = ["Look at this photo carefully and describe what bird species you see in detail."]
        for attr in self.attributes:
            list_prompts.append(self._generate_statement_prompt(attr))
        return list_prompts
    
    def get_llm_prompt(self, list_attr_val):
        if len(list_attr_val) != len(self.attributes) + 1:
            return ""
        
        prompt = f"""I need to identify a bird species based on detailed visual characteristics. Here are the observations:

{list_attr_val[0]}

Detailed characteristics:
- Size: {list_attr_val[1]}
- Shape: {list_attr_val[2]}
- Color Pattern: {list_attr_val[3]}
- Beak Type: {list_attr_val[4]}
- Wing Shape: {list_attr_val[5]}
- Tail Shape: {list_attr_val[6]}
- Habitat: {list_attr_val[7]}
- Behavior: {list_attr_val[8]}
- Plumage: {list_attr_val[9]}
- Markings: {list_attr_val[10]}
- Head Features: {list_attr_val[11]}
- Body Features: {list_attr_val[12]}
- Leg Color: {list_attr_val[13]}
- Flight Pattern: {list_attr_val[14]}
- Vocalization: {list_attr_val[15]}
- Seasonal Appearance: {list_attr_val[16]}

Based on these detailed bird characteristics, what specific bird species does this most likely represent? Consider the combination of all features including size, shape, coloration, beak shape, wing structure, tail pattern, habitat preferences, typical behaviors, plumage characteristics, distinctive markings, head and body features, leg coloration, flight patterns, vocalizations, and seasonal appearance patterns.

Provide only the species name as your answer."""
        
        return prompt
    
    def _generate_statement_prompt(self, attribute):
        return f"Describe the {attribute} of this bird in detail."

# 提示词定义
birdsnap_how_to1 = f"""Your task is to tell me what are the useful attributes for distinguishing {SUPERCLASS} {CLASSUNIT} in a photo of a {SUPERCLASS}.

First, provide a general description of the bird in the photo. Then, analyze the following attributes in sequence:
"""
for i, attr in enumerate(['size', 'shape', 'color_pattern', 'beak_type', 'wing_shape', 'tail_shape', 'habitat', 'behavior', 'plumage', 'markings', 'head_features', 'body_features', 'leg_color', 'flight_pattern', 'vocalization', 'seasonal_appearance'], 1):
    birdsnap_how_to1 += f"{i}. Describe the {attr.replace('_', ' ')} of this bird.\n"

birdsnap_how_to2 = birdsnap_how_to1 + """\nBased on all these attributes, what is the most likely bird species? Consider the combination of all features."""

class BirdsnapDiscovery(Dataset):
    """BirdSnap数据集的发现集加载器（基于JSON文件）"""
    
    def __init__(self, cfg, folder_suffix=''):
        # 使用配置中的实验目录路径
        json_path = os.path.join(cfg['expt_dir'], 'images_split', f'images_discovery_all{folder_suffix}.json')
        print(f"构建发现集,json_path: {json_path}")
        
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
                elif img_file.startswith('images/'):
                    # 是相对images目录的路径，拼接数据目录
                    img_path = os.path.join(cfg['data_dir'], img_file)
                else:
                    # 是纯文件名，需要拼接完整路径
                    img_path = os.path.join(cfg['data_dir'], 'images', img_file)
                self.samples.append(img_path)
                self.subcategories.append(class_name)
                self.targets.append(class_id)
        
        # 创建子类别到样本的映射
        from collections import defaultdict
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

class BirdsnapDataset(Dataset):
    """BirdSnap数据集的PyTorch Dataset实现"""
    
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        # 加载数据集划分信息
        split_file = os.path.join(root, 'split_birdsnap_images.json')
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        self.samples = []
        self.labels = []
        self.class_names = []
        
        # 根据训练/测试/验证模式加载对应数据
        split_key = 'train' if train else 'test'
        for item in split_data[split_key]:
            class_name, class_id, img_path = item
            full_img_path = os.path.join(root, img_path)
            
            if os.path.exists(full_img_path):
                self.samples.append(full_img_path)
                self.labels.append(class_id)
                if class_name not in self.class_names:
                    self.class_names.append(class_name)
        
        self.class_names = sorted(self.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
    
    def __getitem__(self, index):
        img_path = self.samples[index]
        label = self.labels[index]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label, img_path
    
    def __len__(self):
        return len(self.samples)

def _transform(img_size=224):
    """数据变换函数"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def build_birdsnap_prompter(cfg: dict):
    """构建BirdSnap提示词生成器"""
    return BirdsnapPrompter()

def build_birdsnap_discovery(cfg: dict, folder_suffix=''):
    """构建BirdSnap发现集"""
    return BirdsnapDiscovery(cfg, folder_suffix=folder_suffix)

def build_birdsnap_test(cfg):
    """构建BirdSnap测试集"""
    transform = _transform(img_size=cfg.get('image_size', 224))
    dataset = BirdsnapDataset(cfg['data_dir'], train=False, transform=transform)
    
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=cfg.get('batch_size', 32), 
                      shuffle=False, num_workers=4)

def build_birdsnap_swav_train(cfg):
    """构建BirdSnap SwAV训练集"""
    transform = _transform(img_size=cfg.get('image_size', 224))
    dataset = BirdsnapDataset(cfg['data_dir'], train=True, transform=transform)
    
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=cfg.get('batch_size', 32), 
                      shuffle=True, num_workers=4, drop_last=True)
