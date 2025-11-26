#!/usr/bin/env python3
"""
修复JSON文件中的路径，去掉train_和test_前缀
"""

import json
import os
from pathlib import Path

def fix_json_paths(json_file_path):
    """修复JSON文件中的路径，去掉train_和test_前缀"""
    print(f"Processing {json_file_path}...")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        
        def fix_path(path):
            """修复单个路径"""
            if '/cars_train/' in path:
                return path.replace('/cars_train/', '/cars_train/train_')
            elif '/cars_test/' in path:
                return path.replace('/cars_test/', '/cars_test/test_')
            return path
        
        def add_prefix_to_path(path):
            """在路径中添加train_和test_前缀"""
            if '/cars_train/' in path and 'train_' not in path:
                # 提取文件名并添加前缀
                parts = path.split('/')
                filename = parts[-1]
                parts[-1] = 'train_' + filename
                return '/'.join(parts)
            elif '/cars_test/' in path and 'test_' not in path:
                # 提取文件名并添加前缀
                parts = path.split('/')
                filename = parts[-1]
                parts[-1] = 'test_' + filename
                return '/'.join(parts)
            return path
        
        # 递归处理数据结构
        def process_data(item):
            nonlocal modified
            if isinstance(item, list):
                for i, sub_item in enumerate(item):
                    if isinstance(sub_item, str) and sub_item.startswith('./datasets/'):
                        # 添加前缀
                        fixed_path = add_prefix_to_path(sub_item)
                        if fixed_path != sub_item:
                            item[i] = fixed_path
                            modified = True
                    else:
                        process_data(sub_item)
            elif isinstance(item, dict):
                for key, value in item.items():
                    process_data(value)
        
        process_data(data)
        
        if modified:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Modified {json_file_path}")
        else:
            print(f"  - No changes needed for {json_file_path}")
            
    except Exception as e:
        print(f"  ✗ Error processing {json_file_path}: {e}")

def fix_json_directory(directory):
    """修复目录中的所有JSON文件"""
    print(f"Fixing JSON files in {directory}...")
    
    json_files = list(Path(directory).glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {directory}")
        return
    
    for json_file in json_files:
        fix_json_paths(str(json_file))

def main():
    """主函数"""
    # 修复 datasets/car_196/images_split 目录
    datasets_json_dir = "/home/hdl/project/fgvr_test_new/datasets/car_196/images_split"
    if os.path.exists(datasets_json_dir):
        fix_json_directory(datasets_json_dir)
    
    # 修复 experiments/car196/images_split 目录
    experiments_json_dir = "/home/hdl/project/fgvr_test_new/experiments/car196/images_split"
    if os.path.exists(experiments_json_dir):
        fix_json_directory(experiments_json_dir)

if __name__ == "__main__":
    main()
