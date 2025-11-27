"""
基于JSON文件的发现集加载器
支持从JSON文件直接加载发现集数据，无需创建目录结构
"""

import os
import json
from collections import defaultdict
from typing import List, Dict

class JSONDiscovery:
    """
    基于JSON文件的发现集数据加载器
    
    直接从JSON文件读取数据，提供与原有Discovery类兼容的接口
    """
    
    def __init__(self, json_file_path: str, dataset_name: str = None):
        """
        初始化JSON发现集
        
        Args:
            json_file_path: JSON文件路径
            dataset_name: 数据集名称，用于类别名标准化
        """
        self.json_file_path = json_file_path
        self.dataset_name = dataset_name
        
        # 加载数据
        self._load_data()
        
        # 设置索引
        self.index = 0
    
    def _load_data(self):
        """从JSON文件加载数据"""
        print(f"从JSON文件加载发现集: {self.json_file_path}")
        
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"发现集JSON文件不存在: {self.json_file_path}")
        
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 构建样本和标签列表
        self.samples = []
        self.targets = []
        
        # 构建subcat_to_sample映射
        self.subcat_to_sample = defaultdict(list)
        
        # 获取类别名标准化函数
        try:
            from data.class_name_mapper import standardize_test_class_name
            use_standardization = True
        except ImportError:
            use_standardization = False
        
        for class_entry in self.data:
            if len(class_entry) >= 3:
                class_name, class_id, image_paths = class_entry[0], class_entry[1], class_entry[2]
                
                # 标准化类别名称
                if use_standardization and self.dataset_name:
                    standardized_class_name = standardize_test_class_name(class_name, self.dataset_name)
                else:
                    standardized_class_name = class_name
                
                # 添加图像路径到样本列表
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        self.samples.append(img_path)
                        self.targets.append(int(class_id))
                        self.subcat_to_sample[standardized_class_name].append(img_path)
                    else:
                        print(f"⚠️ 图像文件不存在: {img_path}")
        
        # 设置类别名称
        self.classes = list(self.subcat_to_sample.keys())
        
        total_images = len(self.samples)
        total_categories = len(self.subcat_to_sample)
        print(f"✓ 加载完成: {total_categories} 个类别，共 {total_images} 张图像")
    
    def __len__(self):
        """返回样本总数"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        if idx >= len(self.samples):
            raise IndexError("索引超出范围")
        
        return self.samples[idx], self.targets[idx]
    
    def __iter__(self):
        """迭代器支持"""
        self.index = 0
        return self
    
    def __next__(self):
        """迭代下一个样本"""
        if self.index >= len(self.samples):
            raise StopIteration
        
        sample = self.samples[self.index]
        target = self.targets[self.index]
        self.index += 1
        
        return sample, target
    
    def get_class_samples(self, class_name: str) -> List[str]:
        """
        获取指定类别的所有样本路径
        
        Args:
            class_name: 类别名称
            
        Returns:
            图像路径列表
        """
        return self.subcat_to_sample.get(class_name, [])
    
    def get_class_info(self) -> Dict[str, int]:
        """
        获取每个类别的样本数量统计
        
        Returns:
            类别名到样本数的映射
        """
        return {class_name: len(samples) for class_name, samples in self.subcat_to_sample.items()}
    
    def summary(self):
        """打印数据集摘要信息"""
        print(f"=== JSON发现集摘要 ===")
        print(f"JSON文件: {self.json_file_path}")
        print(f"数据集: {self.dataset_name}")
        print(f"类别数: {len(self.subcat_to_sample)}")
        print(f"总样本数: {len(self.samples)}")
        print(f"类别分布:")
        for class_name, samples in self.subcat_to_sample.items():
            print(f"  {class_name}: {len(samples)} 张图像")
        print("=" * 30)
