#!/usr/bin/env python3
"""
构建 Birdsnap 数据集
参考新数据集构建指南，风格参考 datasets/dtd
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def extract_images_from_parquet(parquet_file, output_dir):
    """从 parquet 文件中提取图像"""
    df = pd.read_parquet(parquet_file)
    extracted_count = 0
    
    for idx, row in df.iterrows():
        try:
            image_data = row['image']
            label = row['label']
            
            if isinstance(image_data, dict):
                img_bytes = image_data['bytes']
                img_path = image_data.get('path', f'image_{idx}.jpg')
            else:
                img_bytes = image_data
                img_path = f'image_{idx}.jpg'
            
            label_dir = Path(output_dir) / label
            label_dir.mkdir(parents=True, exist_ok=True)
            
            if isinstance(img_path, str):
                filename = os.path.basename(img_path)
                if not filename.endswith(('.jpg', '.jpeg', '.png')):
                    filename = f"{filename}.jpg"
            else:
                filename = f"image_{idx}.jpg"
            
            output_path = label_dir / filename
            counter = 1
            while output_path.exists():
                name, ext = os.path.splitext(filename)
                output_path = label_dir / f"{name}_{counter}{ext}"
                counter += 1
            
            img = Image.open(BytesIO(img_bytes))
            img.save(output_path, format='JPEG')
            extracted_count += 1
            
        except Exception as e:
            print(f"处理第 {idx} 行时出错: {e}")
            continue
    
    return extracted_count

def load_split_info():
    """加载训练/测试划分信息"""
    split_file = Path('/home/hdl/project/fgvr_test/datasets/birdsnap/split_Birdsnap.json')
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    train_images = set()
    test_images = set()
    
    for img_path, species_id, species_name in split_data['train']:
        train_images.add(f"{species_name}/{img_path}")
    
    for img_path, species_id, species_name in split_data['test']:
        test_images.add(f"{species_name}/{img_path}")
    
    return train_images, test_images

def load_species_info():
    """加载物种信息"""
    species_file = Path('/home/hdl/project/fgvr_test/datasets/birdsnap/species.txt')
    species_info = {}
    
    with open(species_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines[1:]:  # 跳过标题行
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            species_id = int(parts[0])
            common_name = parts[1]
            scientific_name = parts[2]
            dir_name = parts[3]
            species_info[dir_name] = {
                'id': species_id,
                'common_name': common_name,
                'scientific_name': scientific_name,
                'dir_name': dir_name
            }
    
    return species_info

def create_dataset_structure(base_dir, species_info):
    """创建数据集目录结构 - 优化版本，只创建需要的目录"""
    dirs_to_create = [
        'images_train',
        'images_test',
        'images_discovery_all',
        'images_discovery_all_1',
        'images_discovery_all_2',
        'images_discovery_all_3',
        'images_discovery_all_4',
        'images_discovery_all_5',
        'images_discovery_all_6',
        'images_discovery_all_7',
        'images_discovery_all_8',
        'images_discovery_all_9',
        'images_discovery_all_10',
        'images_discovery_random'
    ]
    
    print("创建主要目录结构...")
    for dir_name in dirs_to_create:
        dir_path = Path(base_dir) / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    print("物种子目录将在复制文件时按需创建...")

def load_species_mapping():
    """加载物种映射关系"""
    species_file = Path('/home/hdl/project/fgvr_test/datasets/birdsnap/species.txt')
    common_to_dir = {}
    dir_to_info = {}
    
    with open(species_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines[1:]:  # 跳过标题行
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            species_id = int(parts[0])
            common_name = parts[1]
            scientific_name = parts[2]
            dir_name = parts[3]
            
            # 将常见名转换为下划线格式
            common_underscore = common_name.replace(' ', '_').replace("'", '')
            
            common_to_dir[common_underscore] = dir_name
            dir_to_info[dir_name] = {
                'id': species_id,
                'common_name': common_name,
                'scientific_name': scientific_name,
                'dir_name': dir_name
            }
    
    return common_to_dir, dir_to_info

def copy_file_with_logging(src_path, dest_path, thread_id):
    """复制文件并输出日志"""
    try:
        # 确保目标目录存在
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        shutil.copy2(src_path, dest_path)
        
        # 输出日志
        print(f"[线程{thread_id}] 复制: {src_path.name} -> {dest_path.parent.name}/{dest_path.name}")
        return True
    except Exception as e:
        print(f"[线程{thread_id}] 错误: 复制 {src_path.name} 失败: {e}")
        return False

def distribute_images(images_dir, output_dir, train_images, test_images, species_info):
    """分发图像到相应目录"""
    print("分发图像到训练集和测试集...")
    
    # 加载物种映射
    common_to_dir, dir_to_info = load_species_mapping()
    
    # 统计每个类别的图像数量
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    all_images = defaultdict(list)
    
    # 由于划分文件路径与实际文件名不匹配，我们按照80%训练集，20%测试集来划分
    print("由于划分文件路径与实际文件名不匹配，使用80%训练集，20%测试集划分...")
    
    # 收集所有需要复制的文件
    copy_tasks = []
    processed_species = 0
    total_files_to_copy = 0
    
    # 遍历所有提取的图像
    images_path = Path(images_dir)
    
    for species_dir in images_path.iterdir():
        if not species_dir.is_dir():
            continue
            
        species_name = species_dir.name
        
        # 尝试映射到species.txt中的目录名
        if species_name in common_to_dir:
            mapped_species = common_to_dir[species_name]
        elif species_name in dir_to_info:
            mapped_species = species_name
        else:
            # 如果找不到映射，跳过
            print(f"警告: 找不到物种 {species_name} 的映射，跳过")
            continue
        
        if mapped_species not in species_info:
            continue
            
        processed_species += 1
            
        # 获取该类别的所有图像文件
        img_files = [f for f in species_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if len(img_files) == 0:
            continue
        
        print(f"处理物种: {species_name} -> {mapped_species} ({len(img_files)} 张图像)")
        
        # 随机打乱
        random.shuffle(img_files)
        
        # 按80%训练集，20%测试集划分
        split_idx = int(len(img_files) * 0.8)
        train_files = img_files[:split_idx]
        test_files = img_files[split_idx:]
        
        # 添加训练集复制任务
        for img_file in train_files:
            dest_path = Path(output_dir) / 'images_train' / mapped_species / img_file.name
            copy_tasks.append(('train', img_file, dest_path, mapped_species))
            all_images[mapped_species].append(img_file.name)
        
        # 添加测试集复制任务
        for img_file in test_files:
            dest_path = Path(output_dir) / 'images_test' / mapped_species / img_file.name
            copy_tasks.append(('test', img_file, dest_path, mapped_species))
        
        total_files_to_copy += len(train_files) + len(test_files)
    
    print(f"总共需要复制 {total_files_to_copy} 个文件，使用多线程加速...")
    
    # 使用多线程复制文件
    successful_copies = 0
    failed_copies = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 提交所有任务
        future_to_task = {}
        for i, (dataset_type, src_path, dest_path, species) in enumerate(copy_tasks):
            thread_id = i % 8  # 线程ID用于日志
            future = executor.submit(copy_file_with_logging, src_path, dest_path, thread_id)
            future_to_task[future] = (dataset_type, src_path, dest_path, species)
        
        # 处理完成的任务
        for future in tqdm(as_completed(future_to_task), total=len(copy_tasks), desc="复制文件"):
            dataset_type, src_path, dest_path, species = future_to_task[future]
            try:
                success = future.result()
                if success:
                    successful_copies += 1
                    if dataset_type == 'train':
                        train_counts[species] += 1
                    else:
                        test_counts[species] += 1
                else:
                    failed_copies += 1
            except Exception as e:
                print(f"任务执行失败: {e}")
                failed_copies += 1
    
    print(f"处理的物种数: {processed_species}")
    print(f"训练集类别数: {len(train_counts)}")
    print(f"测试集类别数: {len(test_counts)}")
    print(f"训练集总图像数: {sum(train_counts.values())}")
    print(f"测试集总图像数: {sum(test_counts.values())}")
    print(f"成功复制: {successful_copies}, 失败: {failed_copies}")
    
    return all_images

def create_discovery_sets(all_images, output_dir, species_info):
    """创建discovery数据集"""
    print("创建discovery数据集...")
    
    # 收集所有discovery复制任务
    discovery_tasks = []
    
    for k in range(1, 11):
        print(f"准备 images_discovery_all_{k}...")
        discovery_dir = Path(output_dir) / f'images_discovery_all_{k}'
        
        for species_name, image_list in all_images.items():
            if len(image_list) >= k:
                # 随机选择k张图像
                selected_images = random.sample(image_list, k)
                train_dir = Path(output_dir) / 'images_train' / species_name
                
                for img_name in selected_images:
                    src_path = train_dir / img_name
                    dst_path = discovery_dir / species_name / img_name
                    discovery_tasks.append((src_path, dst_path, f'discovery_{k}'))
    
    # 创建 images_discovery_all (与 images_discovery_all_3 相同)
    print("准备 images_discovery_all...")
    discovery_all_dir = Path(output_dir) / 'images_discovery_all'
    discovery_3_dir = Path(output_dir) / 'images_discovery_all_3'
    
    if discovery_3_dir.exists():
        for species_dir in discovery_3_dir.iterdir():
            if species_dir.is_dir():
                dest_dir = discovery_all_dir / species_dir.name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img_file in species_dir.iterdir():
                    src_path = img_file
                    dst_path = dest_dir / img_file.name
                    discovery_tasks.append((src_path, dst_path, 'discovery_all'))
    
    # 创建 images_discovery_random (zipf长尾分布)
    print("准备 images_discovery_random...")
    create_zipf_distribution(all_images, output_dir, species_info, discovery_tasks)
    
    # 使用多线程执行所有discovery复制任务
    print(f"总共需要复制 {len(discovery_tasks)} 个discovery文件...")
    
    successful_discovery = 0
    failed_discovery = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 提交所有任务
        future_to_task = {}
        for i, (src_path, dst_path, task_type) in enumerate(discovery_tasks):
            thread_id = i % 8
            future = executor.submit(copy_file_with_logging, src_path, dst_path, thread_id)
            future_to_task[future] = (src_path, dst_path, task_type)
        
        # 处理完成的任务
        for future in tqdm(as_completed(future_to_task), total=len(discovery_tasks), desc="复制discovery文件"):
            src_path, dst_path, task_type = future_to_task[future]
            try:
                success = future.result()
                if success:
                    successful_discovery += 1
                else:
                    failed_discovery += 1
            except Exception as e:
                print(f"Discovery任务执行失败: {e}")
                failed_discovery += 1
    
    print(f"Discovery复制完成: 成功 {successful_discovery}, 失败 {failed_discovery}")

def create_zipf_distribution(all_images, output_dir, species_info, discovery_tasks):
    """创建zipf长尾分布的discovery集"""
    discovery_random_dir = Path(output_dir) / 'images_discovery_random'
    
    # 按照图像数量排序物种
    species_by_count = sorted(all_images.items(), key=lambda x: len(x[1]), reverse=True)
    total_species = len(species_by_count)
    
    # Zipf分布: 第i个类别的样本数与1/i成正比
    for rank, (species_name, image_list) in enumerate(species_by_count, 1):
        # 计算该类别应该有的样本数
        zipf_ratio = 1.0 / rank
        max_samples = len(image_list)
        target_samples = max(1, int(max_samples * zipf_ratio))
        
        # 随机选择样本
        if len(image_list) >= target_samples:
            selected_images = random.sample(image_list, target_samples)
        else:
            selected_images = image_list
        
        # 复制图像
        train_dir = Path(output_dir) / 'images_train' / species_name
        for img_name in selected_images:
            src_path = train_dir / img_name
            dst_path = discovery_random_dir / species_name / img_name
            discovery_tasks.append((src_path, dst_path, 'discovery_random'))

def create_class_list(species_info, output_dir):
    """创建类别列表文件"""
    class_list = []
    for species_dir, info in species_info.items():
        class_list.append(info['common_name'])
    
    with open(Path(output_dir) / 'classes.txt', 'w') as f:
        for class_name in class_list:
            f.write(f"{class_name}\n")
    
    print(f"创建了包含 {len(class_list)} 个类别的类别列表")

def verify_dataset(output_dir, species_info):
    """验证数据集完整性"""
    print("\n验证数据集完整性...")
    
    # 检查训练集
    train_dir = Path(output_dir) / 'images_train'
    train_species = set()
    train_total = 0
    
    for species_dir in train_dir.iterdir():
        if species_dir.is_dir():
            img_count = len([f for f in species_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            if img_count > 0:
                train_species.add(species_dir.name)
                train_total += img_count
    
    # 检查测试集
    test_dir = Path(output_dir) / 'images_test'
    test_species = set()
    test_total = 0
    
    for species_dir in test_dir.iterdir():
        if species_dir.is_dir():
            img_count = len([f for f in species_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            if img_count > 0:
                test_species.add(species_dir.name)
                test_total += img_count
    
    print(f"训练集: {len(train_species)} 个类别, {train_total} 张图像")
    print(f"测试集: {len(test_species)} 个类别, {test_total} 张图像")
    print(f"总类别数: {len(species_info)}")
    
    # 检查discovery集
    for k in [1, 3, 10]:
        discovery_dir = Path(output_dir) / f'images_discovery_all_{k}'
        if discovery_dir.exists():
            discovery_species = set()
            discovery_total = 0
            for species_dir in discovery_dir.iterdir():
                if species_dir.is_dir():
                    img_count = len([f for f in species_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    if img_count > 0:
                        discovery_species.add(species_dir.name)
                        discovery_total += img_count
            print(f"Discovery_{k}: {len(discovery_species)} 个类别, {discovery_total} 张图像")

def main():
    """主函数"""
    # 设置路径 - 直接在 birdsnap 目录下创建
    birdsnap_dir = Path('/home/hdl/project/fgvr_test/datasets/birdsnap')
    images_dir = birdsnap_dir / 'images'  # 使用现有的images目录
    output_dir = birdsnap_dir  # 直接输出到 birdsnap 目录
    
    print("=" * 60)
    print("构建 Birdsnap 数据集")
    print("=" * 60)
    
    # 1. 检查图像目录
    if not images_dir.exists():
        print(f"错误: 图像目录不存在: {images_dir}")
        return
    
    # 快速统计物种数量（不遍历每个文件）
    species_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    print(f"找到 {len(species_dirs)} 个物种目录")
    
    # 2. 加载信息
    print("步骤1: 加载物种和划分信息...")
    species_info = load_species_info()
    train_images, test_images = load_split_info()
    
    print(f"加载了 {len(species_info)} 个物种信息")
    print(f"训练集图像数: {len(train_images)}")
    print(f"测试集图像数: {len(test_images)}")
    
    # 3. 创建目录结构
    print("步骤2: 创建目录结构...")
    create_dataset_structure(output_dir, species_info)
    
    # 4. 分发图像
    print("步骤3: 分发图像...")
    all_images = distribute_images(images_dir, output_dir, train_images, test_images, species_info)
    
    # 5. 创建discovery集
    print("步骤4: 创建discovery数据集...")
    create_discovery_sets(all_images, output_dir, species_info)
    
    # 6. 创建类别列表
    print("步骤5: 创建类别列表...")
    create_class_list(species_info, output_dir)
    
    # 7. 验证数据集
    print("步骤6: 验证数据集...")
    verify_dataset(output_dir, species_info)
    
    print("\n" + "=" * 60)
    print("Birdsnap 数据集构建完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    random.seed(42)
    main()
