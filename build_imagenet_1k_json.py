#!/usr/bin/env python3
"""
ImageNet_1k数据集JSON文件构建脚本
根据数据集划分文件构建指南生成所有必需的JSON文件
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import time

class ImageNet1KJSONBuilder:
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.work_dir = self.dataset_root
        self.images_split_dir = self.dataset_root / "images_split"
        self.train_dir = self.dataset_root / "train"
        self.val_dir = self.dataset_root / "val"
        self.test_dir = self.dataset_root / "test"
        
        # 创建images_split目录
        self.images_split_dir.mkdir(exist_ok=True)
        
        # 加载类别映射
        self.class_mapping = self._load_class_mapping()
        
        # 设置随机种子以确保可重现性
        random.seed(42)
        np.random.seed(42)
        
        print(f"初始化ImageNet_1k JSON构建器")
        print(f"数据集根目录: {self.dataset_root}")
        print(f"工作目录: {self.work_dir}")
        print(f"类别数量: {len(self.class_mapping)}")
        
    def _load_class_mapping(self) -> Dict[str, Tuple[str, str]]:
        """加载类别ID到(Imagenet_ID, 类别名)的映射"""
        mapping_file = self.dataset_root / "imagenet_class_index.json"
        
        if not mapping_file.exists():
            raise FileNotFoundError(f"类别映射文件不存在: {mapping_file}")
        
        with open(mapping_file, 'r') as f:
            class_data = json.load(f)
        
        mapping = {}
        for class_idx, (imagenet_id, class_name) in class_data.items():
            mapping[class_idx] = (imagenet_id, class_name)
        
        return mapping
    
    def _get_class_directories(self, split_dir: Path) -> Dict[str, List[str]]:
        """获取指定目录下的类别和图片文件"""
        print(f"扫描目录: {split_dir}")
        
        if not split_dir.exists():
            print(f"警告: 目录不存在 {split_dir}")
            return {}
        
        class_images = {}
        
        # 获取所有类别目录
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith('n')]
        class_dirs.sort()
        
        for class_dir in class_dirs:
            class_id = class_dir.name
            
            # 查找对应的类别索引和名称
            class_info = None
            for idx, (imagenet_id, class_name) in self.class_mapping.items():
                if imagenet_id == class_id:
                    class_info = (idx, class_name)
                    break
            
            if class_info is None:
                print(f"警告: 找不到类别 {class_id} 的映射信息")
                continue
            
            class_idx, class_name = class_info
            
            # 获取图片文件
            image_files = []
            for ext in ['*.JPEG', '*.jpeg', '*.jpg', '*.JPG', '*.png', '*.PNG']:
                image_files.extend(class_dir.glob(ext))
            
            # 转换为相对路径
            relative_paths = []
            for img_file in sorted(image_files):
                rel_path = str(img_file.relative_to(self.work_dir))
                relative_paths.append(rel_path)
            
            if relative_paths:
                class_images[class_idx] = {
                    'class_name': class_name,
                    'imagenet_id': class_id,
                    'images': relative_paths
                }
                print(f"  类别 {class_name} ({class_id}): {len(relative_paths)} 张图片")
        
        return class_images
    
    def build_images_json(self):
        """构建images.json - 所有图片"""
        print("\n=== 构建 images.json ===")
        
        # 合并所有数据集的图片
        all_classes = {}
        
        # 处理训练集
        train_classes = self._get_class_directories(self.train_dir)
        for class_idx, class_info in train_classes.items():
            all_classes[class_idx] = class_info
        
        # 处理验证集
        val_classes = self._get_class_directories(self.val_dir)
        for class_idx, class_info in val_classes.items():
            if class_idx in all_classes:
                all_classes[class_idx]['images'].extend(class_info['images'])
            else:
                all_classes[class_idx] = class_info
        
        # 处理测试集
        test_classes = self._get_class_directories(self.test_dir)
        for class_idx, class_info in test_classes.items():
            if class_idx in all_classes:
                all_classes[class_idx]['images'].extend(class_info['images'])
            else:
                all_classes[class_idx] = class_info
        
        # 按类别索引排序
        sorted_classes = sorted(all_classes.items(), key=lambda x: int(x[0]))
        
        # 构建JSON数据
        result = []
        total_images = 0
        
        for class_idx, class_info in sorted_classes:
            class_name = class_info['class_name']
            images = class_info['images']
            
            result.append([class_name, int(class_idx), images])
            total_images += len(images)
        
        # 保存文件
        output_file = self.images_split_dir / "images.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"images.json 构建完成:")
        print(f"  类别数: {len(result)}")
        print(f"  总图片数: {total_images}")
        print(f"  输出文件: {output_file}")
        
        return result
    
    def build_split_jsons(self):
        """构建训练集、验证集、测试集的JSON文件"""
        print("\n=== 构建分割JSON文件 ===")
        
        # 构建训练集
        train_classes = self._get_class_directories(self.train_dir)
        train_result = self._build_split_json(train_classes, "images_train.json")
        
        # 构建验证集
        val_classes = self._get_class_directories(self.val_dir)
        val_result = self._build_split_json(val_classes, "images_val.json")
        
        # 构建测试集
        test_classes = self._get_class_directories(self.test_dir)
        test_result = self._build_split_json(test_classes, "images_test.json")
        
        # 构建split_ImageNet_1k_images.json
        self._build_main_split_json(train_result, val_result, test_result)
        
        return train_result, val_result, test_result
    
    def _build_split_json(self, class_data: Dict, filename: str) -> List:
        """构建单个分割的JSON文件"""
        print(f"构建 {filename}...")
        
        # 按类别索引排序
        sorted_classes = sorted(class_data.items(), key=lambda x: int(x[0]))
        
        result = []
        total_images = 0
        
        for class_idx, class_info in sorted_classes:
            class_name = class_info['class_name']
            images = class_info['images']
            
            result.append([class_name, int(class_idx), images])
            total_images += len(images)
        
        # 保存文件
        output_file = self.images_split_dir / filename
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"  {filename}: {len(result)} 个类别, {total_images} 张图片")
        
        return result
    
    def _build_main_split_json(self, train_data: List, val_data: List, test_data: List):
        """构建主要的分割JSON文件"""
        print("构建 split_ImageNet_1k_images.json...")
        
        split_data = {"train": [], "val": [], "test": []}
        
        # 处理训练集
        for class_info in train_data:
            class_name, class_idx, images = class_info
            for img_path in images:
                split_data["train"].append([class_name, class_idx, img_path])
        
        # 处理验证集
        for class_info in val_data:
            class_name, class_idx, images = class_info
            for img_path in images:
                split_data["val"].append([class_name, class_idx, img_path])
        
        # 处理测试集
        for class_info in test_data:
            class_name, class_idx, images = class_info
            for img_path in images:
                split_data["test"].append([class_name, class_idx, img_path])
        
        # 保存文件
        output_file = self.images_split_dir / "split_ImageNet_1k_images.json"
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"  split_ImageNet_1k_images.json:")
        print(f"    训练集: {len(split_data['train'])} 张图片")
        print(f"    验证集: {len(split_data['val'])} 张图片")
        print(f"    测试集: {len(split_data['test'])} 张图片")
    
    def build_discovery_jsons(self, train_data: List):
        """构建发现集JSON文件"""
        print("\n=== 构建发现集JSON文件 ===")
        
        # 构建不同数量的发现集
        for k in range(1, 11):
            self._build_discovery_json_k(train_data, k)
        
        # 构建默认发现集（等同于k=3）
        self._build_discovery_json_k(train_data, 3, "images_discovery_all.json")
        
        # 构建随机发现集（长尾分布）
        self._build_random_discovery_json(train_data)
    
    def _build_discovery_json_k(self, train_data: List, k: int, filename: str = None):
        """构建每类k张图片的发现集"""
        if filename is None:
            filename = f"images_discovery_all_{k}.json"
        
        print(f"构建 {filename} (每类{k}张)...")
        
        result = []
        
        for class_info in train_data:
            class_name, class_idx, images = class_info
            
            # 随机选择k张图片
            if len(images) >= k:
                selected_images = random.sample(images, k)
            else:
                # 如果图片数量不足k张，则全部选择
                selected_images = images.copy()
                # 随机重复一些图片直到达到k张
                while len(selected_images) < k:
                    selected_images.append(random.choice(images))
            
            result.append([class_name, class_idx, selected_images])
        
        # 保存文件
        output_file = self.images_split_dir / filename
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        total_images = sum(len(class_info[2]) for class_info in result)
        print(f"  {filename}: {len(result)} 个类别, {total_images} 张图片")
    
    def _build_random_discovery_json(self, train_data: List):
        """构建随机发现集（长尾分布）"""
        print("构建 images_discovery_random.json (长尾分布)...")
        
        result = []
        
        for class_info in train_data:
            class_name, class_idx, images = class_info
            
            # 使用长尾分布确定采样数量
            # 长尾分布参数：大部分类别采样1-3张，少数类别采样更多
            if random.random() < 0.6:  # 60%的类别采样1-3张
                k = random.randint(1, 3)
            elif random.random() < 0.9:  # 30%的类别采样4-7张
                k = random.randint(4, 7)
            else:  # 10%的类别采样8-15张
                k = random.randint(8, 15)
            
            # 确保不超过可用图片数量
            k = min(k, len(images))
            
            # 随机选择k张图片
            selected_images = random.sample(images, k)
            result.append([class_name, class_idx, selected_images])
        
        # 保存文件
        output_file = self.images_split_dir / "images_discovery_random.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        total_images = sum(len(class_info[2]) for class_info in result)
        print(f"  images_discovery_random.json: {len(result)} 个类别, {total_images} 张图片")
    
    def build_all_jsons(self):
        """构建所有JSON文件"""
        print("=" * 60)
        print("开始构建ImageNet_1k数据集JSON文件")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. 构建所有图片的JSON
            images_data = self.build_images_json()
            
            # 2. 构建分割JSON文件
            train_data, val_data, test_data = self.build_split_jsons()
            
            # 3. 构建发现集JSON文件
            self.build_discovery_jsons(train_data)
            
            end_time = time.time()
            
            print("\n" + "=" * 60)
            print("所有JSON文件构建完成!")
            print(f"总耗时: {end_time - start_time:.2f} 秒")
            print(f"输出目录: {self.images_split_dir}")
            
            # 列出生成的文件
            generated_files = list(self.images_split_dir.glob("*.json"))
            print(f"生成的文件数: {len(generated_files)}")
            for file in sorted(generated_files):
                file_size = file.stat().st_size / (1024 * 1024)  # MB
                print(f"  {file.name} ({file_size:.1f} MB)")
            
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"构建过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    dataset_root = "/home/hdl/project/fgvr_test_new/datasets/ImageNet_1k"
    
    # 检查数据集目录
    if not os.path.exists(dataset_root):
        print(f"错误: 数据集目录不存在 {dataset_root}")
        return
    
    # 创建构建器
    builder = ImageNet1KJSONBuilder(dataset_root)
    
    # 构建所有JSON文件
    success = builder.build_all_jsons()
    
    if success:
        print("\n✅ JSON文件构建成功!")
    else:
        print("\n❌ JSON文件构建失败!")

if __name__ == "__main__":
    main()
