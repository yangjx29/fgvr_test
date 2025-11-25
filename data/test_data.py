"""
测试集数据处理模块
处理不同数据集的测试集路径和采样逻辑
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math
from data.class_name_mapper import (
    get_dataset_name_from_key,
    standardize_test_class_name
)

# 数据集测试集路径映射
DATASET_TEST_PATHS = {
    'dog120': 'dogs_120/images_test',
    'bird200': 'CUB_200_2011/CUB_200_2011/images_test',
    'flower102': 'flowers_102/images_test',
    'pet37': 'pet_37/images_test',
    'car196': 'car_196/images_test',
    'aircraft100': 'fgvc_aircraft/images_test',
    'eurosat10': 'eurosat/images_test',
    'food101': 'food_101/images_test',
    'dtd47': 'dtd/images_test',
    'caltech101': 'caltech101/images_test',
    'caltech256': 'caltech256/images_test',
    'deepfashion23': 'DeepFashion/images_test',
}


def get_test_set_path(dataset_name: str, data_root: str) -> str:
    """
    获取数据集的测试集路径
    
    Args:
        dataset_name: 数据集名称 (如 'dog120', 'bird200', 'aircraft100' 等)
        data_root: 数据集根目录
    
    Returns:
        测试集的完整路径
    
    Raises:
        ValueError: 如果数据集名称不支持
    """
    if dataset_name not in DATASET_TEST_PATHS:
        raise ValueError(
            f"不支持的数据集: {dataset_name}. "
            f"支持的数据集: {', '.join(DATASET_TEST_PATHS.keys())}"
        )
    
    test_path = os.path.join(data_root, DATASET_TEST_PATHS[dataset_name])
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"测试集目录不存在: {test_path}. "
            f"请确保已创建 images_test 目录"
        )
    
    return test_path


def sample_test_images(
    test_dir: str,
    test_percentage: float,
    seed: int = 42,
    dataset_key: Optional[str] = None,
    use_true_random: bool = False
) -> List[Tuple[str, str]]:
    """
    从测试集中按百分比采样图像
    
    Args:
        test_dir: 测试集目录路径
        test_percentage: 采样百分比 (0-100)
        seed: 随机种子（仅在use_true_random=False时使用）
        dataset_key: 数据集key（如 'flower102'），用于类别名标准化
        use_true_random: 是否使用真随机数生成器（True=真随机，False=伪随机）
    
    Returns:
        采样的图像列表，每个元素为 (图像路径, 标准化后的类别名) 元组
    """
    # 根据use_true_random选择随机数生成方式
    if use_true_random:
        from utils.true_random import get_true_random_generator
        true_random = get_true_random_generator(use_blocking=True, fallback_to_pseudo=True)
        if true_random.is_available():
            print("✓ 使用真随机数生成器进行采样")
        else:
            print("⚠️  真随机数生成器不可用，回退到伪随机数生成器")
            random.seed(seed)
    else:
        random.seed(seed)
        true_random = None
    
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")
    
    sampled_images = []
    
    # 获取数据集名称用于类别名标准化
    dataset_name = None
    if dataset_key:
        dataset_name = get_dataset_name_from_key(dataset_key)
    
    # 遍历所有类别目录
    class_dirs = sorted([d for d in test_path.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs:
        raw_class_name = class_dir.name
        
        # 标准化类别名称
        if dataset_name:
            class_name = standardize_test_class_name(raw_class_name, dataset_name)
        else:
            class_name = raw_class_name
        
        # 获取该类别的所有图像
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_dir.glob(ext)))
        
        if not image_files:
            continue
        
        # 计算要采样的图像数量（向上取整，至少1张）
        total_images = len(image_files)
        num_samples = max(1, math.ceil(total_images * test_percentage / 100.0))
        num_samples = min(num_samples, total_images)  # 不超过总数
        
        # 随机采样：使用真随机或伪随机
        if use_true_random and true_random and true_random.is_available():
            sampled = true_random.sample(image_files, num_samples)
        else:
            sampled = random.sample(image_files, num_samples)
        
        # 添加到结果列表
        for img_path in sampled:
            sampled_images.append((str(img_path), class_name))
    
    return sampled_images


def get_test_images_by_percentage(
    dataset_name: str,
    data_root: str,
    test_percentage: float,
    seed: int = 42,
    use_true_random: bool = False
) -> List[Tuple[str, str]]:
    """
    根据数据集名称和采样百分比获取测试图像
    
    Args:
        dataset_name: 数据集名称
        data_root: 数据集根目录
        test_percentage: 采样百分比 (0-100)
        seed: 随机种子（仅在use_true_random=False时使用）
        use_true_random: 是否使用真随机数生成器
    
    Returns:
        采样的图像列表，每个元素为 (图像路径, 标准化后的类别名) 元组
    """
    test_dir = get_test_set_path(dataset_name, data_root)
    return sample_test_images(test_dir, test_percentage, seed, dataset_key=dataset_name, use_true_random=use_true_random)


def get_all_test_images(dataset_name: str, data_root: str) -> List[Tuple[str, str]]:
    """
    获取测试集的所有图像
    
    Args:
        dataset_name: 数据集名称
        data_root: 数据集根目录
    
    Returns:
        所有测试图像列表，每个元素为 (图像路径, 类别名) 元组
    """
    return get_test_images_by_percentage(dataset_name, data_root, 100.0)


def validate_test_set(dataset_name: str, data_root: str) -> Dict[str, int]:
    """
    验证测试集结构并返回统计信息
    
    Args:
        dataset_name: 数据集名称
        data_root: 数据集根目录
    
    Returns:
        统计信息字典，包含类别数、总图像数等
    """
    test_dir = get_test_set_path(dataset_name, data_root)
    test_path = Path(test_dir)
    
    class_dirs = [d for d in test_path.iterdir() if d.is_dir()]
    total_images = 0
    class_image_counts = {}
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_dir.glob(ext)))
        
        num_images = len(image_files)
        class_image_counts[class_name] = num_images
        total_images += num_images
    
    return {
        'test_dir': test_dir,
        'num_classes': len(class_dirs),
        'total_images': total_images,
        'class_image_counts': class_image_counts,
        'min_images_per_class': min(class_image_counts.values()) if class_image_counts else 0,
        'max_images_per_class': max(class_image_counts.values()) if class_image_counts else 0,
        'avg_images_per_class': total_images / len(class_dirs) if class_dirs else 0,
    }

