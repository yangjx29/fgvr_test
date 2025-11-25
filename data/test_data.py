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
from data import DATA_STATS

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
    'deepfashion_multimodal23': 'DeepFashion/images_test',
    'sun397': 'SUN397/images_test',
}


def get_dataset_key_for_test(cfg: Dict) -> str:
    """
    获取用于测试集的数据集键
    
    对于已经包含编号的数据集（如 caltech101, caltech256），直接使用 dataset_name
    对于其他数据集，拼接 dataset_name 和 num_classes
    
    Args:
        cfg: 配置字典，包含 'dataset_name' 和 'num_classes' 字段
        
    Returns:
        str: 数据集键（如 'dog120', 'caltech101', 'deepfashion23'）
        
    Examples:
        >>> cfg = {'dataset_name': 'dog', 'num_classes': 120}
        >>> get_dataset_key_for_test(cfg)
        'dog120'
        
        >>> cfg = {'dataset_name': 'caltech256', 'num_classes': 257}
        >>> get_dataset_key_for_test(cfg)
        'caltech256'
        
        >>> cfg = {'dataset_name': 'deepfashion_multimodal', 'num_classes': 23}
        >>> get_dataset_key_for_test(cfg)
        'deepfashion_multimodal23'
    """
    dataset_name = cfg.get('dataset_name', '')
    num_classes = cfg.get('num_classes', '')
    
    # 数据集名称已经包含编号的情况
    # 这些数据集的 dataset_name 已经包含了类别数，不需要再拼接
    if dataset_name in ['caltech101', 'caltech256']:
        return dataset_name
    
    # 其他数据集需要拼接编号
    return f"{dataset_name}{num_classes}"


def _sample_sun397_nested(
    test_path: Path,
    test_percentage: float,
    seed: int,
    use_true_random: bool,
    true_random
) -> List[Tuple[str, str]]:
    """
    SUN397特殊处理：递归遍历嵌套目录结构
    
    Args:
        test_path: 测试集根目录
        test_percentage: 采样百分比
        seed: 随机种子
        use_true_random: 是否使用真随机
        true_random: 真随机数生成器
        
    Returns:
        采样的图像列表
    """
    sampled_images = []
    
    # 递归遍历所有包含图片的目录
    for root, dirs, files in os.walk(test_path):
        # 获取相对路径作为类别名（如 "a/abbey"）
        rel_path = os.path.relpath(root, test_path)
        if rel_path == '.':
            continue
        
        # 检查是否有图片文件
        image_files = []
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(root, f))
        
        if not image_files:
            continue
        
        # 将路径分隔符统一为 /（类别名格式）
        class_name = rel_path.replace(os.sep, '/')
        
        # 验证类别名是否在SUN397_STATS中
        if class_name not in DATA_STATS['sun397']['class_names']:
            # 尝试查找匹配的类别（处理可能的格式差异）
            matched = False
            for std_name in DATA_STATS['sun397']['class_names']:
                if std_name.lower() == class_name.lower() or std_name.replace('/', os.sep) == rel_path:
                    class_name = std_name
                    matched = True
                    break
            if not matched:
                continue  # 跳过不在标准类别列表中的目录
        
        # 计算要采样的图像数量
        total_images = len(image_files)
        num_samples = max(1, math.ceil(total_images * test_percentage / 100.0))
        num_samples = min(num_samples, total_images)
        
        # 随机采样
        if use_true_random and true_random and true_random.is_available():
            sampled = true_random.sample(image_files, num_samples)
        else:
            sampled = random.sample(image_files, num_samples)
        
        # 添加到结果列表
        for img_path in sampled:
            sampled_images.append((img_path, class_name))
    
    return sampled_images


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
    
    # SUN397特殊处理：需要处理嵌套目录结构
    if dataset_name == 'sun397':
        # SUN397的测试集可能有嵌套结构（如 a/abbey），需要递归遍历
        sampled_images = _sample_sun397_nested(test_path, test_percentage, seed, use_true_random, true_random)
        return sampled_images
    
    # 遍历所有类别目录（扁平结构）
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

