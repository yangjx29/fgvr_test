import os
import json
import random
import argparse
from typing import Dict, List, Tuple, Union
from collections import defaultdict
import yaml

class JSONDiscovery:
    """
    基于JSON文件的发现集数据加载器
    
    直接从JSON文件读取数据，提供与原有Discovery类兼容的接口
    """
    
    def __init__(self, json_file_path: str, dataset_name: str = None):
        """
        初始化JSON发现集
        
        Args:
            json_file_path: JSON文件路径
            dataset_name: 数据集名称，用于类别名标准化
        """
        self.json_file_path = json_file_path
        self.dataset_name = dataset_name
        
        # 加载数据
        self._load_data()
        
        # 设置索引
        self.index = 0
    
    def _load_data(self):
        """从JSON文件加载数据"""
        print(f"从JSON文件加载发现集: {self.json_file_path}")
        
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"发现集JSON文件不存在: {self.json_file_path}")
        
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 构建样本和标签列表
        self.samples = []
        self.targets = []
        
        # 构建subcat_to_sample映射
        self.subcat_to_sample = defaultdict(list)
        
        # 获取类别名标准化函数
        try:
            from class_name_mapper import standardize_test_class_name
            use_standardization = True
        except ImportError:
            use_standardization = False
        
        for class_entry in self.data:
            if len(class_entry) >= 3:
                class_name, class_id, image_paths = class_entry[0], class_entry[1], class_entry[2]
                
                # 标准化类别名称
                if use_standardization and self.dataset_name:
                    standardized_class_name = standardize_test_class_name(class_name, self.dataset_name)
                else:
                    standardized_class_name = class_name
                
                # 添加图像路径到样本列表
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        self.samples.append(img_path)
                        self.targets.append(int(class_id))
                        self.subcat_to_sample[standardized_class_name].append(img_path)
                    else:
                        print(f"⚠️ 图像文件不存在: {img_path}")
        
        # 设置类别名称
        self.classes = list(self.subcat_to_sample.keys())
        
        total_images = len(self.samples)
        total_categories = len(self.subcat_to_sample)
        print(f"✓ 加载完成: {total_categories} 个类别，共 {total_images} 张图像")
    
    def __len__(self):
        """返回样本总数"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        if idx >= len(self.samples):
            raise IndexError("索引超出范围")
        
        return self.samples[idx], self.targets[idx]
    
    def __iter__(self):
        """迭代器支持"""
        self.index = 0
        return self
    
    def __next__(self):
        """迭代下一个样本"""
        if self.index >= len(self.samples):
            raise StopIteration
        
        sample = self.samples[self.index]
        target = self.targets[self.index]
        self.index += 1
        
        return sample, target
    
    def get_class_samples(self, class_name: str) -> List[str]:
        """
        获取指定类别的所有样本路径
        
        Args:
            class_name: 类别名称
            
        Returns:
            图像路径列表
        """
        return self.subcat_to_sample.get(class_name, [])
    
    def get_class_info(self) -> Dict[str, int]:
        """
        获取每个类别的样本数量统计
        
        Returns:
            类别名到样本数的映射
        """
        return {class_name: len(samples) for class_name, samples in self.subcat_to_sample.items()}
    
    def summary(self):
        """打印数据集摘要信息"""
        print(f"=== JSON发现集摘要 ===")
        print(f"JSON文件: {self.json_file_path}")
        print(f"数据集: {self.dataset_name}")
        print(f"类别数: {len(self.subcat_to_sample)}")
        print(f"总样本数: {len(self.samples)}")
        print(f"类别分布:")
        for class_name, samples in self.subcat_to_sample.items():
            print(f"  {class_name}: {len(samples)} 张图像")
        print("=" * 30)

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

def extract_discovery_set(train_data: List[List], num_per_category: Union[int, str], random_seed: int = None) -> List[List]:
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

def ensure_output_directory(dataset_name: str, extract_type: str = 'knowledge_base', use_true_random: bool = True) -> str:
    """确保输出目录存在，根据逻辑文档构建目录结构"""
    dataset_info = get_dataset_info(dataset_name)
    
    # 根据抽取类型确定基础目录
    if extract_type == 'knowledge_base':
        base_dir = os.path.join(dataset_info['experiment_dir_full'], 'result', 'discovering_extract_knowledge_base')
    elif extract_type == 'fast_slow':
        base_dir = os.path.join(dataset_info['experiment_dir_full'], 'result', 'discovering_extract_fast_slow')
    elif extract_type == 'test_extract_fast_slow':
        base_dir = os.path.join(dataset_info['experiment_dir_full'], 'result', 'test_extract_fast_slow')
    else:
        raise ValueError(f"未知的抽取类型: {extract_type}")
    
    # 根据随机类型选择子目录
    if use_true_random:
        output_dir = os.path.join(base_dir, 'true_randomness')
    else:
        output_dir = os.path.join(base_dir, 'pseudorandom')
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_discovery_set(discovery_data: List[List], dataset_name: str, suffix: str, extract_type: str = 'knowledge_base', use_true_random: bool = True) -> str:
    """
    保存发现集到JSON文件
    
    Args:
        discovery_data: 发现集数据
        dataset_name: 数据集名称
        suffix: 文件后缀（如 '_k', '_random'）
        extract_type: 抽取类型 ('knowledge_base', 'fast_slow' 或 'test_extract_fast_slow')
        use_true_random: 是否使用真随机（影响子目录选择）
    
    Returns:
        保存的文件路径
    """
    output_dir = ensure_output_directory(dataset_name, extract_type, use_true_random)
    
    # 生成文件名
    if extract_type == 'test_extract_fast_slow':
        # 测试集抽取使用不同的命名规则
        if suffix.startswith('_'):
            filename = f"test{suffix}.json"
        else:
            filename = f"test_{suffix}.json"
    else:
        # 发现集抽取命名规则
        if suffix.startswith('_'):
            filename = f"images_discovery_all{suffix}.json"
        else:
            filename = f"images_discovery_all_{suffix}.json"
    
    output_path = os.path.join(output_dir, filename)
    
    # 如果文件已存在，先删除（根据逻辑文档要求）
    if os.path.exists(output_path):
        print(f"⚠️ 文件已存在，删除旧文件: {output_path}")
        os.remove(output_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(discovery_data, f, indent=2, ensure_ascii=False)
    
    return output_path

def validate_json_file(json_path: str) -> bool:
    """验证JSON文件是否正确生成"""
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

def extract_and_save_discovery_set(dataset_name: str, num_per_category: Union[int, str], 
                                 random_seed: int = None, output_suffix: str = None, 
                                 extract_type: str = 'knowledge_base') -> JSONDiscovery:
    """
    抽取并保存发现集的完整流程
    
    Args:
        dataset_name: 数据集名称
        num_per_category: 每类抽取数量或'random'
        random_seed: 随机种子，None表示真随机
        output_suffix: 输出文件后缀
        extract_type: 抽取类型 ('knowledge_base', 'fast_slow' 或 'test_extract_fast_slow')
    
    Returns:
        JSONDiscovery对象
    """
    print(f"🔄 开始抽取发现集: {dataset_name}, 每类{num_per_category}个样本, 类型: {extract_type}")
    
    # 判断是否为真随机
    use_true_random = (random_seed is None)
    
    if use_true_random:
        print(f"🎲 真随机模式: 每次运行结果不同")
        random_type = "true_randomness"
    else:
        print(f"🔒 固定种子模式: seed={random_seed}")
        random_type = "pseudorandom"
    
    # 加载训练数据
    train_data = load_train_data(dataset_name)
    print(f"✓ 加载训练数据: {len(train_data)} 个类别")
    
    # 抽取发现集
    discovery_data = extract_discovery_set(train_data, num_per_category, random_seed)
    print(f"✓ 抽取发现集: {len(discovery_data)} 个类别")
    
    # 保存发现集
    output_path = save_discovery_set(discovery_data, dataset_name, output_suffix or str(num_per_category), extract_type, use_true_random)
    print(f"✓ 发现集已保存到: {output_path}")
    print(f"✓ 保存位置: {random_type} 子目录")
    
    # 验证JSON文件
    if not validate_json_file(output_path):
        raise RuntimeError(f"JSON文件验证失败: {output_path}")
    
    # 创建JSONDiscovery对象
    json_discovery = JSONDiscovery(output_path, dataset_name)
    
    # 打印统计信息
    total_images = sum(len(entry[2]) for entry in discovery_data if len(entry) >= 3)
    print(f"✓ 发现集统计: {len(discovery_data)} 个类别，共 {total_images} 张图像")
    
    return json_discovery

def main():
    parser = argparse.ArgumentParser(description='从训练集中随机抽取发现集')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['pet', 'dog', 'bird', 'flower', 'car', 'aircraft', 'eurosat', 'food', 'dtd', 'sun397', 'imagenet_a', 'imagenet_r', 'birdsnap'],
                        help='数据集名称')
    parser.add_argument('--num_per_category', type=str, required=True,
                        help='每个类别抽取的样本数，可以是数字(如1,2,3...)或"random"')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--output_suffix', type=str, default=None,
                        help='输出文件后缀，默认根据num_per_category生成')
    parser.add_argument('--extract_type', type=str, default='knowledge_base',
                        choices=['knowledge_base', 'fast_slow', 'test_extract_fast_slow'],
                        help='抽取类型: knowledge_base(知识库构建), fast_slow(快慢测试) 或 test_extract_fast_slow(测试集抽取)')
    parser.add_argument('--validate_only', action='store_true', 
                        help='仅验证现有JSON文件')
    
    args = parser.parse_args()
    
    if args.validate_only:
        # 仅验证模式
        use_true_random = (args.seed is None)
        output_dir = ensure_output_directory(args.dataset, args.extract_type, use_true_random)
        suffix = args.output_suffix if args.output_suffix else args.num_per_category
        
        if args.extract_type == 'test_extract_fast_slow':
            # 测试集抽取命名规则
            if suffix.startswith('_'):
                filename = f"test{suffix}.json"
            else:
                filename = f"test_{suffix}.json"
        else:
            # 发现集抽取命名规则
            if suffix.startswith('_'):
                filename = f"images_discovery_all{suffix}.json"
            else:
                filename = f"images_discovery_all_{suffix}.json"
        
        json_path = os.path.join(output_dir, filename)
        if validate_json_file(json_path):
            print("✅ 验证通过")
        else:
            print("❌ 验证失败")
        return
    
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
    print(f"抽取类型: {args.extract_type}")
    
    # 执行抽取
    try:
        json_discovery = extract_and_save_discovery_set(
            args.dataset, 
            num_per_category, 
            args.seed, 
            suffix,
            args.extract_type
        )
        print("🎉 抽取完成!")
        
        # 打印详细统计
        json_discovery.summary()
        
    except Exception as e:
        print(f"❌ 抽取失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
