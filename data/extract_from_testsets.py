"""
测试集数据处理模块
处理不同数据集的测试集路径和采样逻辑
支持真随机和伪随机模式的JSON文件保存
"""

import os
import sys
import random
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    'imagenet_a200': 'ImageNet_A/images_test',
    'imagenet_r200': 'ImageNet_R/images_test',
    'birdsnap500': 'birdsnap/images_test',
}

def load_dataset_config():
    """加载数据集配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "datasets_list.yml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_dataset_info(dataset_name: str) -> dict:
    """获取数据集信息"""
    DATASET_CONFIG = load_dataset_config()
    
    # 从dataset_name中提取基础名称（如从bird200提取bird）
    base_name = dataset_name.rstrip('0123456789')
    if base_name == '':
        base_name = dataset_name
    
    if base_name not in DATASET_CONFIG['dataset_mapping']:
        raise ValueError(f"Unknown dataset: {base_name}. Available: {list(DATASET_CONFIG['dataset_mapping'].keys())}")
    
    dataset_info = DATASET_CONFIG['dataset_mapping'][base_name].copy()
    experiments_root = DATASET_CONFIG.get('experiments_root', './experiments')
    dataset_info['experiments_root'] = experiments_root
    dataset_info['experiment_dir_full'] = os.path.join(experiments_root, dataset_info['experiment_dir'])
    
    return dataset_info

def ensure_test_output_directory(dataset_name: str, use_true_random: bool) -> str:
    """确保测试集输出目录存在"""
    dataset_info = get_dataset_info(dataset_name)
    
    if use_true_random:
        output_dir = os.path.join(dataset_info['experiment_dir_full'], 'result', 'test_data_true_randomness')
    else:
        output_dir = os.path.join(dataset_info['experiment_dir_full'], 'result', 'test_data_true_pseudorandom')
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_test_set_to_json(test_data: List[Tuple[str, str]], dataset_name: str, 
                         test_percentage: float, use_true_random: bool) -> str:
    """
    保存测试集到JSON文件
    
    Args:
        test_data: 测试集数据，格式为 [(image_path, class_name), ...]
        dataset_name: 数据集名称
        test_percentage: 采样百分比
        use_true_random: 是否使用真随机
    
    Returns:
        保存的文件路径
    """
    output_dir = ensure_test_output_directory(dataset_name, use_true_random)
    
    # 生成文件名
    percentage_int = int(test_percentage) if test_percentage == int(test_percentage) else test_percentage
    filename = f"test_{percentage_int}.json"
    
    output_path = os.path.join(output_dir, filename)
    
    # 转换数据格式为JSON格式
    # 按类别分组
    class_dict = {}
    for img_path, class_name in test_data:
        if class_name not in class_dict:
            class_dict[class_name] = []
        class_dict[class_name].append(img_path)
    
    # 转换为标准JSON格式
    json_data = []
    for class_name, img_paths in class_dict.items():
        json_data.append([class_name, class_name, img_paths])  # 使用class_name作为id
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return output_path

def validate_test_json_file(json_path: str) -> bool:
    """验证测试集JSON文件是否正确生成"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查数据格式
        if not isinstance(data, list):
            print(f"❌ JSON文件格式错误: 期望list，实际{type(data)}")
            return False
        
        for i, entry in enumerate(data):
            if not isinstance(entry, list) or len(entry) < 3:
                print(f"❌ 第{i}个条目格式错误: 期望[class_name, class_id, [paths]]")
                return False
            
            class_name, class_id, image_paths = entry[0], entry[1], entry[2]
            
            if not isinstance(class_name, str):
                print(f"❌ 第{i}个条目类别名错误: 期望str，实际{type(class_name)}")
                return False
            
            if not isinstance(image_paths, list):
                print(f"❌ 第{i}个条目图像路径错误: 期望list，实际{type(image_paths)}")
                return False
            
            # 检查图像文件是否存在
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    print(f"⚠️ 图像文件不存在: {img_path}")
        
        print(f"✓ JSON文件验证通过: {len(data)} 个类别")
        return True
        
    except Exception as e:
        print(f"❌ JSON文件验证失败: {e}")
        return False


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


def get_test_json_path(dataset_name: str) -> str:
    """
    获取数据集的测试集JSON文件路径
    
    Args:
        dataset_name: 数据集名称 (如 'bird200', 'dog120' 等)
    
    Returns:
        测试集JSON文件的完整路径
    
    Raises:
        FileNotFoundError: 如果JSON文件不存在
    """
    dataset_info = get_dataset_info(dataset_name)
    experiment_dir = dataset_info['experiment_dir_full']
    
    # 构建JSON文件路径
    json_path = os.path.join(experiment_dir, 'images_split', 'images_test.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"测试集JSON文件不存在: {json_path}. "
            f"请确保已运行数据集复制脚本生成JSON文件"
        )
    
    return json_path


def load_test_set_from_json(json_path: str) -> List[Tuple[str, str]]:
    """
    从JSON文件加载测试集数据
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        测试集数据，格式为 [(image_path, class_name), ...]
    """
    print(f"从JSON文件加载测试集: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_images = []
    
    for class_entry in data:
        if len(class_entry) >= 3:
            class_name, class_id, image_paths = class_entry[0], class_entry[1], class_entry[2]
            
            for img_path in image_paths:
                test_images.append((img_path, class_name))
    
    print(f"✓ 加载完成: {len(data)} 个类别，共 {len(test_images)} 张图像")
    
    return test_images


def sample_test_images_from_json(
    test_data: List[Tuple[str, str]],
    test_percentage: float,
    seed: int = 42,
    use_true_random: bool = False
) -> List[Tuple[str, str]]:
    """
    从已加载的测试集中按百分比采样图像
    
    Args:
        test_data: 已加载的测试集数据，格式为 [(image_path, class_name), ...]
        test_percentage: 采样百分比 (0-100)
        seed: 随机种子（仅在use_true_random=False时使用）
        use_true_random: 是否使用真随机数生成器（True=真随机，False=伪随机）
    
    Returns:
        采样的图像列表，每个元素为 (图像路径, 类别名) 元组
    """
    # 根据use_true_random选择随机数生成方式
    if use_true_random:
        try:
            # 确保项目根目录在Python路径中
            current_file = os.path.abspath(__file__)
            data_dir = os.path.dirname(current_file)  # data目录
            project_root = os.path.dirname(data_dir)  # 项目根目录
            
            # 移除data目录，添加项目根目录
            if data_dir in sys.path:
                sys.path.remove(data_dir)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from utils.true_random import get_true_random_generator
            true_random = get_true_random_generator(use_blocking=True, fallback_to_pseudo=True)
            if true_random.is_available():
                print("✓ 使用真随机数生成器进行采样")
            else:
                print("⚠️  真随机数生成器不可用，回退到伪随机数生成器")
                random.seed(seed)
        except (ImportError, Exception) as e:
            print(f"⚠️  真随机数生成器不可用，回退到伪随机数生成器: {e}")
            random.seed(seed)
            true_random = None
    else:
        random.seed(seed)
        true_random = None
    
    # 按类别分组
    class_groups = {}
    for img_path, class_name in test_data:
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(img_path)
    
    sampled_images = []
    
    for class_name, image_files in class_groups.items():
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
            sampled_images.append((img_path, class_name))
    
    return sampled_images


def sample_test_images(
    test_dir: str,
    test_percentage: float,
    seed: int = 42,
    dataset_key: Optional[str] = None,
    use_true_random: bool = False,
    save_to_json: bool = True,
    dataset_name: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    从测试集中按百分比采样图像
    
    Args:
        test_dir: 测试集目录路径
        test_percentage: 采样百分比 (0-100)
        seed: 随机种子（仅在use_true_random=False时使用）
        dataset_key: 数据集key（如 'flower102'），用于类别名标准化
        use_true_random: 是否使用真随机数生成器（True=真随机，False=伪随机）
        save_to_json: 是否保存到JSON文件
        dataset_name: 数据集名称，用于保存JSON文件
    
    Returns:
        采样的图像列表，每个元素为 (图像路径, 标准化后的类别名) 元组
    """
    # 根据use_true_random选择随机数生成方式
    if use_true_random:
        try:
            # 确保项目根目录在Python路径中
            current_file = os.path.abspath(__file__)
            data_dir = os.path.dirname(current_file)  # data目录
            project_root = os.path.dirname(data_dir)  # 项目根目录
            
            # 移除data目录，添加项目根目录
            if data_dir in sys.path:
                sys.path.remove(data_dir)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from utils.true_random import get_true_random_generator
            true_random = get_true_random_generator(use_blocking=True, fallback_to_pseudo=True)
            if true_random.is_available():
                print("✓ 使用真随机数生成器进行采样")
            else:
                print("⚠️  真随机数生成器不可用，回退到伪随机数生成器")
                random.seed(seed)
        except (ImportError, Exception) as e:
            print(f"⚠️  真随机数生成器不可用，回退到伪随机数生成器: {e}")
            random.seed(seed)
            true_random = None
    else:
        random.seed(seed)
        true_random = None
    
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")
    
    sampled_images = []
    
    # 获取数据集名称用于类别名标准化
    dataset_name_for_standardization = None
    if dataset_key:
        dataset_name_for_standardization = get_dataset_name_from_key(dataset_key)
    
    # SUN397特殊处理：需要处理嵌套目录结构
    if dataset_name_for_standardization == 'sun397':
        # SUN397的测试集可能有嵌套结构（如 a/abbey），需要递归遍历
        sampled_images = _sample_sun397_nested(test_path, test_percentage, seed, use_true_random, true_random)
        
        # 保存到JSON文件
        if save_to_json and dataset_name:
            output_path = save_test_set_to_json(sampled_images, dataset_name, test_percentage, use_true_random)
            print(f"✓ 测试集已保存到: {output_path}")
            
            # 验证JSON文件
            if not validate_test_json_file(output_path):
                print(f"⚠️ JSON文件验证失败: {output_path}")
        
        return sampled_images
    
    # 遍历所有类别目录（扁平结构）
    class_dirs = sorted([d for d in test_path.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs:
        raw_class_name = class_dir.name
        
        # 标准化类别名称
        if dataset_name_for_standardization:
            class_name = standardize_test_class_name(raw_class_name, dataset_name_for_standardization)
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
    
    # 保存到JSON文件
    if save_to_json and dataset_name:
        output_path = save_test_set_to_json(sampled_images, dataset_name, test_percentage, use_true_random)
        print(f"✓ 测试集已保存到: {output_path}")
        
        # 验证JSON文件
        if not validate_test_json_file(output_path):
            print(f"⚠️ JSON文件验证失败: {output_path}")
    
    return sampled_images


def get_test_images_by_percentage(
    dataset_name: str,
    test_percentage: float,
    seed: int = 42,
    use_true_random: bool = False,
    save_to_json: bool = True
) -> List[Tuple[str, str]]:
    """
    根据数据集名称和采样百分比获取测试图像（从JSON文件加载）
    
    Args:
        dataset_name: 数据集名称
        test_percentage: 采样百分比 (0-100)
        seed: 随机种子（仅在use_true_random=False时使用）
        use_true_random: 是否使用真随机数生成器
        save_to_json: 是否保存到JSON文件
    
    Returns:
        采样的图像列表，每个元素为 (图像路径, 标准化后的类别名) 元组
    """
    # 获取测试集JSON文件路径
    json_path = get_test_json_path(dataset_name)
    
    # 从JSON文件加载测试集
    test_data = load_test_set_from_json(json_path)
    
    # 按百分比采样
    sampled_data = sample_test_images_from_json(test_data, test_percentage, seed, use_true_random)
    
    # 保存到JSON文件
    if save_to_json:
        output_path = save_test_set_to_json(sampled_data, dataset_name, test_percentage, use_true_random)
        print(f"✓ 测试集已保存到: {output_path}")
        
        # 验证JSON文件
        if not validate_test_json_file(output_path):
            print(f"⚠️ JSON文件验证失败: {output_path}")
    
    return sampled_data


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


def extract_and_save_test_set(dataset_name: str, test_percentage: float, 
                            use_true_random: bool = False, seed: int = 42) -> List[Tuple[str, str]]:
    """
    抽取并保存测试集的完整流程（从JSON文件加载）
    
    Args:
        dataset_name: 数据集名称
        test_percentage: 采样百分比
        use_true_random: 是否使用真随机
        seed: 随机种子
    
    Returns:
        采样的图像列表
    """
    print(f"🔄 开始抽取测试集: {dataset_name}, 采样比例: {test_percentage}%")
    
    if use_true_random:
        print(f"🎲 真随机模式: 每次运行结果不同")
    else:
        print(f"🔒 固定种子模式: seed={seed}")
    
    # 抽取测试集（从JSON文件加载）
    test_data = get_test_images_by_percentage(
        dataset_name, 
        test_percentage, 
        seed=seed, 
        use_true_random=use_true_random, 
        save_to_json=True
    )
    
    # 打印统计信息
    print(f"✓ 测试集抽取完成: {len(test_data)} 张图像")
    
    # 按类别统计
    class_counts = {}
    for _, class_name in test_data:
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"✓ 类别分布: {len(class_counts)} 个类别")
    
    return test_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从测试集中随机抽取样本')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['dog120', 'bird200', 'flower102', 'pet37', 'car196', 'aircraft100', 
                                'eurosat10', 'food101', 'dtd47', 'caltech101', 'caltech256', 
                                'deepfashion_multimodal23', 'sun397', 'imagenet_a200', 'imagenet_r200', 'birdsnap500'],
                        help='数据集名称')
    parser.add_argument('--percentage', type=float, required=True,
                        help='采样百分比 (0-100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（仅在伪随机模式下使用）')
    parser.add_argument('--true_random', action='store_true',
                        help='使用真随机模式')
    parser.add_argument('--validate_only', action='store_true',
                        help='仅验证现有JSON文件')
    
    args = parser.parse_args()
    
    if args.validate_only:
        # 仅验证模式
        dataset_info = get_dataset_info(args.dataset)
        if args.true_random:
            output_dir = os.path.join(dataset_info['experiment_dir_full'], 'result', 'test_data_true_randomness')
        else:
            output_dir = os.path.join(dataset_info['experiment_dir_full'], 'result', 'test_data_true_pseudorandom')
        
        percentage_int = int(args.percentage) if args.percentage == int(args.percentage) else args.percentage
        filename = f"test_{percentage_int}.json"
        json_path = os.path.join(output_dir, filename)
        
        if validate_test_json_file(json_path):
            print("✅ 验证通过")
        else:
            print("❌ 验证失败")
        return
    
    # 执行抽取
    try:
        test_data = extract_and_save_test_set(
            args.dataset,
            args.percentage,
            use_true_random=args.true_random,
            seed=args.seed
        )
        print("🎉 抽取完成!")
        
        # 显示统计信息
        class_counts = {}
        for _, class_name in test_data:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\n📊 统计信息:")
        print(f"  总图像数: {len(test_data)}")
        print(f"  类别数: {len(class_counts)}")
        print(f"  平均每类: {len(test_data) / len(class_counts):.1f} 张")
        
    except Exception as e:
        print(f"❌ 抽取失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

