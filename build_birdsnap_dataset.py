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
    """创建数据集目录结构"""
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
    
    for dir_name in dirs_to_create:
        dir_path = Path(base_dir) / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # 为每个物种创建子目录
        for species_dir in species_info.keys():
            species_dir_path = dir_path / species_dir
            species_dir_path.mkdir(parents=True, exist_ok=True)

def distribute_images(images_dir, output_dir, train_images, test_images, species_info):
    """分发图像到相应目录"""
    print("分发图像到训练集和测试集...")
    
    # 统计每个类别的图像数量
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    all_images = defaultdict(list)
    
    # 由于划分文件路径与实际文件名不匹配，我们按照80%训练集，20%测试集来划分
    print("由于划分文件路径与实际文件名不匹配，使用80%训练集，20%测试集划分...")
    
    # 遍历所有提取的图像
    images_path = Path(images_dir)
    for species_dir in tqdm(images_path.iterdir(), desc="处理物种目录"):
        if not species_dir.is_dir():
            continue
            
        species_name = species_dir.name
        if species_name not in species_info:
            continue
            
        # 获取该类别的所有图像文件
        img_files = [f for f in species_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # 随机打乱
        random.shuffle(img_files)
        
        # 按80%训练集，20%测试集划分
        split_idx = int(len(img_files) * 0.8)
        train_files = img_files[:split_idx]
        test_files = img_files[split_idx:]
        
        # 复制到训练集
        for img_file in train_files:
            dest_path = Path(output_dir) / 'images_train' / species_name / img_file.name
            shutil.copy2(img_file, dest_path)
            train_counts[species_name] += 1
            all_images[species_name].append(img_file.name)
        
        # 复制到测试集
        for img_file in test_files:
            dest_path = Path(output_dir) / 'images_test' / species_name / img_file.name
            shutil.copy2(img_file, dest_path)
            test_counts[species_name] += 1
    
    print(f"训练集类别数: {len(train_counts)}")
    print(f"测试集类别数: {len(test_counts)}")
    print(f"训练集总图像数: {sum(train_counts.values())}")
    print(f"测试集总图像数: {sum(test_counts.values())}")
    
    return all_images

def create_discovery_sets(all_images, output_dir, species_info):
    """创建discovery数据集"""
    print("创建discovery数据集...")
    
    for k in range(1, 11):
        print(f"创建 images_discovery_all_{k}...")
        discovery_dir = Path(output_dir) / f'images_discovery_all_{k}'
        
        for species_name, image_list in all_images.items():
            if len(image_list) >= k:
                # 随机选择k张图像
                selected_images = random.sample(image_list, k)
                train_dir = Path(output_dir) / 'images_train' / species_name
                
                for img_name in selected_images:
                    src_path = train_dir / img_name
                    dst_path = discovery_dir / species_name / img_name
                    if src_path.exists():
                        shutil.copy2(src_path, dst_path)
    
    # 创建 images_discovery_all (与 images_discovery_all_3 相同)
    print("创建 images_discovery_all...")
    discovery_all_dir = Path(output_dir) / 'images_discovery_all'
    discovery_3_dir = Path(output_dir) / 'images_discovery_all_3'
    
    for species_dir in discovery_3_dir.iterdir():
        if species_dir.is_dir():
            dest_dir = discovery_all_dir / species_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_file in species_dir.iterdir():
                shutil.copy2(img_file, dest_dir / img_file.name)
    
    # 创建 images_discovery_random (zipf长尾分布)
    print("创建 images_discovery_random...")
    create_zipf_distribution(all_images, output_dir, species_info)

def create_zipf_distribution(all_images, output_dir, species_info):
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
            if src_path.exists():
                shutil.copy2(src_path, dst_path)

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
    # 设置路径
    birdsnap_dir = Path('/home/hdl/project/fgvr_test/datasets/birdsnap')
    images_dir = birdsnap_dir / 'images'  # 使用现有的images目录
    output_dir = Path('/home/hdl/project/fgvr_test/datasets/birdsnap_processed')
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("构建 Birdsnap 数据集")
    print("=" * 60)
    
    # 1. 检查图像目录
    if not images_dir.exists():
        print(f"错误: 图像目录不存在: {images_dir}")
        return
    
    # 统计现有图像
    total_images = 0
    species_dirs = []
    for species_dir in images_dir.iterdir():
        if species_dir.is_dir():
            img_count = len([f for f in species_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            if img_count > 0:
                species_dirs.append(species_dir.name)
                total_images += img_count
    
    print(f"找到 {len(species_dirs)} 个物种，共 {total_images} 张图像")
    
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
