#!/usr/bin/env python3
"""
ImageNet_1k数据集JSON文件快速验证脚本
只抽样验证前几个类别，提高验证速度
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class ImageNet1KJSONQuickVerifier:
    def __init__(self, dataset_root: str, sample_classes: int = 3):
        self.dataset_root = Path(dataset_root)
        self.images_split_dir = self.dataset_root / "images_split"
        self.work_dir = self.dataset_root
        self.sample_classes = sample_classes
        
        # 加载类别映射
        self.class_mapping = self._load_class_mapping()
        
        # 存储验证结果
        self.errors = []
        self.warnings = []
        
        print(f"初始化ImageNet_1k JSON快速验证器")
        print(f"数据集根目录: {self.dataset_root}")
        print(f"总类别数量: {len(self.class_mapping)}")
        print(f"抽样验证类别数: {self.sample_classes}")
        
    def _load_class_mapping(self) -> Dict[str, Tuple[str, str]]:
        """加载类别ID到(Imagenet_ID, 类别名)的映射"""
        mapping_file = self.dataset_root / "imagenet_class_index.json"
        
        with open(mapping_file, 'r') as f:
            class_data = json.load(f)
        
        mapping = {}
        for class_idx, (imagenet_id, class_name) in class_data.items():
            mapping[class_idx] = (imagenet_id, class_name)
        
        return mapping
    
    def _log_error(self, message: str):
        """记录错误"""
        self.errors.append(f"❌ {message}")
        print(f"❌ {message}")
    
    def _log_warning(self, message: str):
        """记录警告"""
        self.warnings.append(f"⚠️ {message}")
        print(f"⚠️ {message}")
    
    def _log_success(self, message: str):
        """记录成功信息"""
        print(f"✅ {message}")
    
    def _sample_classes(self, all_classes: List) -> List:
        """抽样选择类别进行验证"""
        if len(all_classes) <= self.sample_classes:
            return all_classes
        
        # 选择前几个类别
        return all_classes[:self.sample_classes]
    
    def verify_json_structure(self, json_data: List, filename: str) -> bool:
        """验证JSON文件结构（抽样）"""
        print(f"验证 {filename} 的JSON结构...")
        
        if not isinstance(json_data, list):
            self._log_error(f"JSON数据应该是列表类型，实际是 {type(json_data)}")
            return False
        
        # 抽样验证类别结构
        sample_data = self._sample_classes(json_data)
        
        for i, class_data in enumerate(sample_data):
            if not isinstance(class_data, list):
                self._log_error(f"类别 {i} 应该是列表类型，实际是 {type(class_data)}")
                return False
            
            if len(class_data) != 3:
                self._log_error(f"类别 {i} 应该有3个元素 [类别名, 类别ID, 图片列表]，实际有 {len(class_data)} 个元素")
                return False
            
            class_name, class_id, image_list = class_data
            
            if not isinstance(class_name, str):
                self._log_error(f"类别 {i} 的类别名应该是字符串类型，实际是 {type(class_name)}")
                return False
            
            if not isinstance(class_id, int):
                self._log_error(f"类别 {i} 的类别ID应该是整数类型，实际是 {type(class_id)}")
                return False
            
            if not isinstance(image_list, list):
                self._log_error(f"类别 {i} 的图片列表应该是列表类型，实际是 {type(image_list)}")
                return False
        
        self._log_success(f"{filename}: JSON结构验证通过 (抽样 {len(sample_data)}/{len(json_data)} 个类别)")
        return True
    
    def verify_class_consistency(self, json_data: List, filename: str) -> bool:
        """验证类别一致性（抽样）"""
        print(f"验证 {filename} 的类别一致性...")
        
        # 抽样验证类别
        sample_data = self._sample_classes(json_data)
        
        for class_data in sample_data:
            class_name, class_id, image_list = class_data
            
            # 检查类别ID是否在映射中
            class_idx_str = str(class_id)
            if class_idx_str not in self.class_mapping:
                self._log_error(f"类别ID {class_id} 不在类别映射中")
                return False
            
            expected_name = self.class_mapping[class_idx_str][1]
            if class_name != expected_name:
                self._log_error(f"类别 {class_id} 名称不匹配: 期望 '{expected_name}', 实际 '{class_name}'")
                return False
        
        self._log_success(f"{filename}: 类别一致性验证通过 (抽样 {len(sample_data)}/{len(json_data)} 个类别)")
        return True
    
    def verify_image_paths(self, json_data: List, filename: str, max_images_per_class: int = 3) -> bool:
        """验证图片路径（抽样）"""
        print(f"验证 {filename} 的图片路径...")
        
        sample_data = self._sample_classes(json_data)
        total_checked = 0
        missing_files = 0
        invalid_paths = 0
        
        for class_data in sample_data:
            class_name, class_id, image_list = class_data
            
            # 每个类别最多检查几张图片
            sample_images = image_list[:max_images_per_class] if len(image_list) > max_images_per_class else image_list
            
            for img_path in sample_images:
                total_checked += 1
                
                # 检查路径格式
                if not isinstance(img_path, str):
                    self._log_error(f"图片路径应该是字符串类型: {img_path}")
                    invalid_paths += 1
                    continue
                
                # 检查路径是否为相对路径
                if img_path.startswith('/') or img_path.startswith('~'):
                    self._log_error(f"图片路径应该是相对路径: {img_path}")
                    invalid_paths += 1
                    continue
                
                # 构建完整路径
                full_path = self.work_dir / img_path
                
                # 检查文件是否存在
                if not full_path.exists():
                    self._log_error(f"图片文件不存在: {full_path}")
                    missing_files += 1
                elif not full_path.is_file():
                    self._log_error(f"路径不是文件: {full_path}")
                    missing_files += 1
        
        if missing_files > 0:
            self._log_error(f"{filename}: {missing_files}/{total_checked} 个图片文件不存在")
            return False
        
        if invalid_paths > 0:
            self._log_error(f"{filename}: {invalid_paths}/{total_checked} 个图片路径无效")
            return False
        
        self._log_success(f"{filename}: 所有 {total_checked} 个图片路径验证通过 (抽样)")
        return True
    
    def verify_split_consistency(self) -> bool:
        """验证分割文件的一致性（抽样）"""
        print("验证分割文件一致性...")
        
        # 读取分割文件
        split_file = self.images_split_dir / "split_ImageNet_1k_images.json"
        train_file = self.images_split_dir / "images_train.json"
        
        try:
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            
            with open(train_file, 'r') as f:
                train_data = json.load(f)
        except Exception as e:
            self._log_error(f"读取分割文件失败: {e}")
            return False
        
        # 验证分割文件结构
        if not isinstance(split_data, dict):
            self._log_error("分割JSON应该是字典类型")
            return False
        
        required_keys = ["train", "val", "test"]
        for key in required_keys:
            if key not in split_data:
                self._log_error(f"分割JSON缺少必需的键: {key}")
                return False
        
        # 抽样验证分割一致性
        split_train_sample = self._sample_classes(split_data["train"])
        
        # 获取训练集中的抽样类别名称
        sample_train_data = self._sample_classes(train_data)
        train_class_names = set(item[0] for item in sample_train_data)
        train_paths = set()
        
        for item in train_data:
            if item[0] in train_class_names:
                train_paths.update(item[2])
        
        split_train_paths = set(item[2] for item in split_train_sample)
        
        # 检查路径是否匹配
        if not split_train_paths.issubset(train_paths):
            self._log_error("分割文件中的训练集路径与train文件不匹配")
            return False
        
        self._log_success("分割文件一致性验证通过 (抽样)")
        return True
    
    def verify_discovery_sets(self) -> bool:
        """验证发现集是否完全来源于训练集（抽样）"""
        print("验证发现集来源...")
        
        # 读取训练集
        train_file = self.images_split_dir / "images_train.json"
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        # 构建训练集图片路径集合（仅抽样类别）
        sample_train_data = self._sample_classes(train_data)
        train_paths = set()
        for class_data in sample_train_data:
            train_paths.update(class_data[2])
        
        # 验证几个发现集文件
        discovery_files = ["images_discovery_all_1.json", "images_discovery_all_3.json", "images_discovery_all_10.json"]
        
        for filename in discovery_files:
            file_path = self.images_split_dir / filename
            if not file_path.exists():
                self._log_warning(f"发现集文件不存在: {filename}")
                continue
            
            with open(file_path, 'r') as f:
                discovery_data = json.load(f)
            
            # 抽样验证
            sample_discovery_data = self._sample_classes(discovery_data)
            
            # 提取发现集图片路径
            discovery_paths = set()
            for class_data in sample_discovery_data:
                discovery_paths.update(class_data[2])
            
            # 检查是否都来自训练集
            non_train_paths = discovery_paths - train_paths
            
            if non_train_paths:
                self._log_error(f"{filename}: {len(non_train_paths)} 个图片不在训练集中")
                for path in list(non_train_paths)[:3]:  # 只显示前3个
                    self._log_error(f"  非训练集图片: {path}")
                return False
            else:
                self._log_success(f"{filename}: 所有 {len(discovery_paths)} 个图片都来自训练集 (抽样)")
        
        return True
    
    def verify_discovery_sampling(self) -> bool:
        """验证发现集采样规则（抽样）"""
        print("验证发现集采样规则...")
        
        # 读取训练集
        train_file = self.images_split_dir / "images_train.json"
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        # 构建训练集图片字典（抽样）
        sample_train_data = self._sample_classes(train_data)
        train_images_by_class = {}
        for class_data in sample_train_data:
            class_name, class_id, image_list = class_data
            train_images_by_class[class_id] = set(image_list)
        
        # 验证几个发现集
        test_cases = [1, 3, 10]
        
        for k in test_cases:
            filename = f"images_discovery_all_{k}.json"
            file_path = self.images_split_dir / filename
            
            if not file_path.exists():
                continue
            
            with open(file_path, 'r') as f:
                discovery_data = json.load(f)
            
            # 抽样验证
            sample_discovery_data = self._sample_classes(discovery_data)
            
            # 检查每个类别的采样数量
            for class_data in sample_discovery_data:
                class_name, class_id, image_list = class_data
                
                if len(image_list) != k:
                    self._log_error(f"{filename}: 类别 {class_name} 应该有 {k} 张图片，实际有 {len(image_list)} 张")
                    return False
                
                # 检查是否来自训练集
                train_class_images = train_images_by_class.get(class_id, set())
                if not set(image_list).issubset(train_class_images):
                    self._log_error(f"{filename}: 类别 {class_name} 包含非训练集图片")
                    return False
            
            self._log_success(f"{filename}: 采样规则验证通过 (抽样)")
        
        return True
    
    def verify_all_jsons(self) -> bool:
        """快速验证所有JSON文件"""
        print("=" * 60)
        print("开始快速验证ImageNet_1k数据集JSON文件")
        print("=" * 60)
        
        all_passed = True
        
        # 验证主要文件
        main_files = [
            "images.json",
            "images_train.json", 
            "images_val.json",
            "images_test.json",
            "split_ImageNet_1k_images.json",
            "images_discovery_all.json",
            "images_discovery_all_1.json",
            "images_discovery_all_3.json",
            "images_discovery_all_10.json"
        ]
        
        for filename in main_files:
            file_path = self.images_split_dir / filename
            if not file_path.exists():
                self._log_error(f"文件不存在: {filename}")
                all_passed = False
                continue
            
            print(f"\n验证文件: {filename}")
            
            try:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                
                # 验证JSON结构
                if filename == "split_ImageNet_1k_images.json":
                    # split文件是字典格式，需要特殊处理
                    if not isinstance(json_data, dict):
                        self._log_error(f"分割JSON应该是字典类型，实际是 {type(json_data)}")
                        all_passed = False
                    else:
                        self._log_success(f"{filename}: JSON结构验证通过")
                else:
                    if not self.verify_json_structure(json_data, filename):
                        all_passed = False
                    
                    # 验证类别一致性
                    if not self.verify_class_consistency(json_data, filename):
                        all_passed = False
                    
                    # 验证图片路径
                    if not self.verify_image_paths(json_data, filename):
                        all_passed = False
            
            except Exception as e:
                self._log_error(f"读取文件 {filename} 失败: {e}")
                all_passed = False
        
        # 验证分割一致性
        if not self.verify_split_consistency():
            all_passed = False
        
        # 验证发现集
        if not self.verify_discovery_sets():
            all_passed = False
        
        # 验证发现集采样
        if not self.verify_discovery_sampling():
            all_passed = False
        
        # 输出验证结果
        print("\n" + "=" * 60)
        print("快速验证结果汇总")
        print("=" * 60)
        
        if self.errors:
            print(f"发现 {len(self.errors)} 个错误:")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print(f"发现 {len(self.warnings)} 个警告:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if all_passed and not self.errors:
            print("🎉 快速验证通过！JSON文件构建基本正确。")
            print("💡 建议: 如需完整验证，可运行 verify_imagenet_1k_json.py")
        else:
            print("❌ 验证失败，请修复上述错误后重新验证。")
        
        print("=" * 60)
        
        return all_passed

def main():
    """主函数"""
    dataset_root = "/home/hdl/project/fgvr_test_new/datasets/ImageNet_1k"
    sample_classes = 3  # 抽样验证前3个类别
    
    # 检查数据集目录
    if not os.path.exists(dataset_root):
        print(f"错误: 数据集目录不存在 {dataset_root}")
        return
    
    # 创建验证器
    verifier = ImageNet1KJSONQuickVerifier(dataset_root, sample_classes)
    
    # 快速验证所有JSON文件
    success = verifier.verify_all_jsons()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
