import os
import json
import random
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
import yaml

def load_dataset_config():
    """加载数据集配置文件"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "datasets_list.yml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_dataset_info(dataset_name: str) -> dict:
    """获取数据集信息"""
    DATASET_CONFIG = load_dataset_config()
    
    if dataset_name not in DATASET_CONFIG['dataset_mapping']:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG['dataset_mapping'].keys())}")
    
    dataset_info = DATASET_CONFIG['dataset_mapping'][dataset_name].copy()
    experiments_root = DATASET_CONFIG.get('experiments_root', './experiments')
    dataset_info['experiments_root'] = experiments_root
    dataset_info['experiment_dir_full'] = os.path.join(experiments_root, dataset_info['experiment_dir'])
    
    return dataset_info

def load_train_data(dataset_name: str) -> List[List]:
    """从images_train.json加载训练数据"""
    dataset_info = get_dataset_info(dataset_name)
    train_json_path = os.path.join(dataset_info['experiment_dir_full'], 'images_split', 'images_train.json')
    
    if not os.path.exists(train_json_path):
        raise FileNotFoundError(f"训练数据文件不存在: {train_json_path}")
    
    with open(train_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_discovery_set(train_data: List[List], num_per_category: int, random_seed: int = None) -> List[List]:
    """
    从训练集中随机抽取发现集
    
    Args:
        train_data: 训练数据，格式为 [[class_name, class_id, [image_paths]], ...]
        num_per_category: 每个类别抽取的样本数，如果为'random'则随机数量
        random_seed: 随机种子
    
    Returns:
        抽取的发现集数据
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    discovery_data = []
    
    for class_entry in train_data:
        if len(class_entry) < 3:
            continue
            
        class_name, class_id, image_paths = class_entry[0], class_entry[1], class_entry[2]
        
        if not isinstance(image_paths, list) or len(image_paths) == 0:
            continue
        
        # 确定抽取数量
        if isinstance(num_per_category, str) and num_per_category.lower() == 'random':
            # 随机数量，至少1张，最多不超过该类别的所有图片
            extract_count = random.randint(1, len(image_paths))
        else:
            # 固定数量
            extract_count = min(num_per_category, len(image_paths))
        
        # 随机抽取图片
        selected_images = random.sample(image_paths, extract_count)
        
        discovery_data.append([class_name, class_id, selected_images])
    
    return discovery_data

def save_discovery_set(discovery_data: List[List], dataset_name: str, suffix: str) -> str:
    """
    保存发现集到JSON文件
    
    Args:
        discovery_data: 发现集数据
        dataset_name: 数据集名称
        suffix: 文件后缀（如 '_k', '_random'）
    
    Returns:
        保存的文件路径
    """
    dataset_info = get_dataset_info(dataset_name)
    output_dir = os.path.join(dataset_info['experiment_dir_full'], 'images_split')
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    if suffix.startswith('_'):
        filename = f"images_discovery_all{suffix}.json"
    else:
        filename = f"images_discovery_all_{suffix}.json"
    
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(discovery_data, f, indent=2, ensure_ascii=False)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='从训练集中随机抽取发现集')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['pet', 'dog', 'bird', 'flower', 'car', 'aircraft'],
                        help='数据集名称')
    parser.add_argument('--num_per_category', type=str, required=True,
                        help='每个类别抽取的样本数，可以是数字(如1,2,3...)或"random"')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--output_suffix', type=str, default=None,
                        help='输出文件后缀，默认根据num_per_category生成')
    
    args = parser.parse_args()
    
    # 解析num_per_category
    if args.num_per_category.lower() == 'random':
        num_per_category = 'random'
        suffix = 'random' if args.output_suffix is None else args.output_suffix
    else:
        try:
            num_per_category = int(args.num_per_category)
            suffix = str(num_per_category) if args.output_suffix is None else args.output_suffix
        except ValueError:
            raise ValueError(f"num_per_category必须是数字或'random'，得到: {args.num_per_category}")
    
    print(f"正在为数据集 {args.dataset} 抽取发现集...")
    print(f"每个类别抽取数量: {num_per_category}")
    print(f"随机种子: {args.seed}")
    
    # 加载训练数据
    train_data = load_train_data(args.dataset)
    print(f"加载训练数据: {len(train_data)} 个类别")
    
    # 统计训练数据信息
    total_images = sum(len(entry[2]) if len(entry) >= 3 and isinstance(entry[2], list) else 1 
                      for entry in train_data)
    print(f"训练集总图像数: {total_images}")
    
    # 抽取发现集
    discovery_data = extract_discovery_set(train_data, num_per_category, args.seed)
    
    # 统计发现集信息
    discovery_images = sum(len(entry[2]) if len(entry) >= 3 and isinstance(entry[2], list) else 1 
                          for entry in discovery_data)
    print(f"发现集包含: {len(discovery_data)} 个类别，{discovery_images} 张图像")
    
    # 保存发现集
    output_path = save_discovery_set(discovery_data, args.dataset, suffix)
    print(f"发现集已保存到: {output_path}")
    
    # 显示每个类别的详细信息
    print("\n每个类别的抽取情况:")
    for entry in discovery_data:
        if len(entry) >= 3:
            class_name = entry[0]
            image_count = len(entry[2]) if isinstance(entry[2], list) else 1
            print(f"  {class_name}: {image_count} 张图像")

if __name__ == "__main__":
    main()
