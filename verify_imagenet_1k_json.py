#!/usr/bin/env python3
"""
ImageNet_1k数据集JSON文件验证脚本
验证所有JSON文件是否符合构建指南要求
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class ImageNet1KJSONVerifier:
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.images_split_dir = self.dataset_root / "images_split"
        self.work_dir = self.dataset_root
        
        # 加载类别映射
        self.class_mapping = self._load_class_mapping()
        
        # 存储验证结果
        self.errors = []
        self.warnings = []
        
        print(f"初始化ImageNet_1k JSON验证器")
        print(f"数据集根目录: {self.dataset_root}")
        print(f"类别数量: {len(self.class_mapping)}")
        
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
    
    def verify_json_structure(self, json_data: List, expected_format: str = "class_list") -> bool:
        """验证JSON文件结构"""
        if not isinstance(json_data, list):
            self._log_error(f"JSON数据应该是列表类型，实际是 {type(json_data)}")
            return False
        
        if expected_format == "class_list":
            for i, class_data in enumerate(json_data):
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
        
        elif expected_format == "split_dict":
            if not isinstance(json_data, dict):
                self._log_error(f"分割JSON数据应该是字典类型，实际是 {type(json_data)}")
                return False
            
            required_keys = ["train", "val", "test"]
            for key in required_keys:
                if key not in json_data:
                    self._log_error(f"分割JSON缺少必需的键: {key}")
                    return False
        
        return True
    
    def verify_class_consistency(self, json_data: List) -> bool:
        """验证类别一致性"""
        print("验证类别一致性...")
        
        # 检查类别数量
        expected_class_count = len(self.class_mapping)
        actual_class_count = len(json_data)
        
        if actual_class_count != expected_class_count:
            self._log_error(f"类别数量不匹配: 期望 {expected_class_count}, 实际 {actual_class_count}")
            return False
        
        # 检查类别ID和名称
        json_class_map = {}
        for class_data in json_data:
            class_name, class_id, image_list = class_data
            json_class_map[class_id] = class_name
        
        # 检查每个类别
        for class_idx, (imagenet_id, expected_name) in self.class_mapping.items():
            class_id_int = int(class_idx)
            
            if class_id_int not in json_class_map:
                self._log_error(f"缺少类别 {class_idx} ({expected_name})")
                return False
            
            actual_name = json_class_map[class_id_int]
            if actual_name != expected_name:
                self._log_error(f"类别 {class_idx} 名称不匹配: 期望 '{expected_name}', 实际 '{actual_name}'")
                return False
        
        self._log_success(f"类别一致性验证通过: {actual_class_count} 个类别")
        return True
    
    def verify_image_paths(self, json_data: List, json_filename: str) -> bool:
        """验证图片路径"""
        print(f"验证 {json_filename} 的图片路径...")
        
        total_images = 0
        missing_files = 0
        invalid_paths = 0
        
        for class_data in json_data:
            class_name, class_id, image_list = class_data
            
            for img_path in image_list:
                total_images += 1
                
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
            self._log_error(f"{json_filename}: {missing_files}/{total_images} 个图片文件不存在")
            return False
        
        if invalid_paths > 0:
            self._log_error(f"{json_filename}: {invalid_paths}/{total_images} 个图片路径无效")
            return False
        
        self._log_success(f"{json_filename}: 所有 {total_images} 个图片路径验证通过")
        return True
    
    def verify_split_consistency(self) -> bool:
        """验证分割文件的一致性"""
        print("验证分割文件一致性...")
        
        # 读取所有分割文件
        split_file = self.images_split_dir / "split_ImageNet_1k_images.json"
        train_file = self.images_split_dir / "images_train.json"
        val_file = self.images_split_dir / "images_val.json"
        test_file = self.images_split_dir / "images_test.json"
        
        try:
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            
            with open(train_file, 'r') as f:
                train_data = json.load(f)
            
            with open(val_file, 'r') as f:
                val_data = json.load(f)
            
            with open(test_file, 'r') as f:
                test_data = json.load(f)
        except Exception as e:
            self._log_error(f"读取分割文件失败: {e}")
            return False
        
        # 验证结构
        if not self.verify_json_structure(split_data, "split_dict"):
            return False
        
        # 提取分割文件中的图片路径
        split_train_paths = set(item[2] for item in split_data["train"])
        split_val_paths = set(item[2] for item in split_data["val"])
        split_test_paths = set(item[2] for item in split_data["test"])
        
        # 提取单独文件中的图片路径
        file_train_paths = set()
        for class_data in train_data:
            file_train_paths.update(class_data[2])
        
        file_val_paths = set()
        for class_data in val_data:
            file_val_paths.update(class_data[2])
        
        file_test_paths = set()
        for class_data in test_data:
            file_test_paths.update(class_data[2])
        
        # 检查一致性
        if split_train_paths != file_train_paths:
            self._log_error("训练集路径不一致: split文件与train文件不匹配")
            return False
        
        if split_val_paths != file_val_paths:
            self._log_error("验证集路径不一致: split文件与val文件不匹配")
            return False
        
        if split_test_paths != file_test_paths:
            self._log_error("测试集路径不一致: split文件与test文件不匹配")
            return False
        
        # 检查分割之间的互斥性
        train_val_overlap = split_train_paths & split_val_paths
        train_test_overlap = split_train_paths & split_test_paths
        val_test_overlap = split_val_paths & split_test_paths
        
        if train_val_overlap:
            self._log_error(f"训练集和验证集有 {len(train_val_overlap)} 个重复图片")
            return False
        
        if train_test_overlap:
            self._log_error(f"训练集和测试集有 {len(train_test_overlap)} 个重复图片")
            return False
        
        if val_test_overlap:
            self._log_error(f"验证集和测试集有 {len(val_test_overlap)} 个重复图片")
            return False
        
        self._log_success("分割文件一致性验证通过")
        return True
    
    def verify_discovery_sets(self) -> bool:
        """验证发现集是否完全来源于训练集"""
        print("验证发现集来源...")
        
        # 读取训练集
        train_file = self.images_split_dir / "images_train.json"
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        # 构建训练集图片路径集合
        train_paths = set()
        for class_data in train_data:
            train_paths.update(class_data[2])
        
        # 验证所有发现集
        discovery_files = [
            f"images_discovery_all_{k}.json" for k in range(1, 11)
        ] + ["images_discovery_all.json", "images_discovery_random.json"]
        
        for filename in discovery_files:
            file_path = self.images_split_dir / filename
            if not file_path.exists():
                self._log_warning(f"发现集文件不存在: {filename}")
                continue
            
            with open(file_path, 'r') as f:
                discovery_data = json.load(f)
            
            # 提取发现集图片路径
            discovery_paths = set()
            for class_data in discovery_data:
                discovery_paths.update(class_data[2])
            
            # 检查是否都来自训练集
            non_train_paths = discovery_paths - train_paths
            
            if non_train_paths:
                self._log_error(f"{filename}: {len(non_train_paths)} 个图片不在训练集中")
                for path in list(non_train_paths)[:5]:  # 只显示前5个
                    self._log_error(f"  非训练集图片: {path}")
                return False
            else:
                self._log_success(f"{filename}: 所有 {len(discovery_paths)} 个图片都来自训练集")
        
        return True
    
    def verify_discovery_sampling(self) -> bool:
        """验证发现集采样规则"""
        print("验证发现集采样规则...")
        
        # 读取训练集
        train_file = self.images_split_dir / "images_train.json"
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        # 构建训练集图片字典
        train_images_by_class = {}
        for class_data in train_data:
            class_name, class_id, image_list = class_data
            train_images_by_class[class_id] = set(image_list)
        
        # 验证固定数量发现集
        for k in range(1, 11):
            filename = f"images_discovery_all_{k}.json"
            file_path = self.images_split_dir / filename
            
            if not file_path.exists():
                continue
            
            with open(file_path, 'r') as f:
                discovery_data = json.load(f)
            
            # 检查每个类别的采样数量
            for class_data in discovery_data:
                class_name, class_id, image_list = class_data
                
                if len(image_list) != k:
                    self._log_error(f"{filename}: 类别 {class_name} 应该有 {k} 张图片，实际有 {len(image_list)} 张")
                    return False
                
                # 检查是否来自训练集
                train_class_images = train_images_by_class.get(class_id, set())
                if not set(image_list).issubset(train_class_images):
                    self._log_error(f"{filename}: 类别 {class_name} 包含非训练集图片")
                    return False
            
            self._log_success(f"{filename}: 采样规则验证通过")
        
        return True
    
    def verify_all_jsons(self) -> bool:
        """验证所有JSON文件"""
        print("=" * 60)
        print("开始验证ImageNet_1k数据集JSON文件")
        print("=" * 60)
        
        all_passed = True
        
        # 验证必需的文件是否存在
        required_files = [
            "images.json",
            "images_train.json", 
            "images_val.json",
            "images_test.json",
            "split_ImageNet_1k_images.json",
            "images_discovery_all.json"
        ]
        
        # 验证可选的发现集文件
        optional_files = [
            f"images_discovery_all_{k}.json" for k in range(1, 11)
        ] + ["images_discovery_random.json"]
        
        all_files = required_files + optional_files
        
        for filename in all_files:
            file_path = self.images_split_dir / filename
            if not file_path.exists():
                if filename in required_files:
                    self._log_error(f"必需文件不存在: {filename}")
                    all_passed = False
                else:
                    self._log_warning(f"可选文件不存在: {filename}")
            else:
                print(f"\n验证文件: {filename}")
                
                try:
                    with open(file_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # 验证JSON结构
                    if filename == "split_ImageNet_1k_images.json":
                        if not self.verify_json_structure(json_data, "split_dict"):
                            all_passed = False
                    else:
                        if not self.verify_json_structure(json_data, "class_list"):
                            all_passed = False
                        
                        # 验证类别一致性
                        if filename not in ["split_ImageNet_1k_images.json"]:
                            if not self.verify_class_consistency(json_data):
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
        print("验证结果汇总")
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
            print("🎉 所有验证通过！JSON文件构建正确。")
        else:
            print("❌ 验证失败，请修复上述错误后重新验证。")
        
        print("=" * 60)
        
        return all_passed

def main():
    """主函数"""
    dataset_root = "/home/hdl/project/fgvr_test_new/datasets/ImageNet_1k"
    
    # 检查数据集目录
    if not os.path.exists(dataset_root):
        print(f"错误: 数据集目录不存在 {dataset_root}")
        return
    
    # 创建验证器
    verifier = ImageNet1KJSONVerifier(dataset_root)
    
    # 验证所有JSON文件
    success = verifier.verify_all_jsons()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
