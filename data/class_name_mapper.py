"""
类别名称映射和标准化模块
处理不同格式的类别名称，统一映射到标准格式
"""

import re
from typing import Dict, Optional, List
from data import DATA_STATS


def normalize_class_name(name: str) -> str:
    """
    标准化类别名称：统一转换为小写，替换分隔符为空格
    
    Args:
        name: 原始类别名称
        
    Returns:
        标准化后的类别名称
    """
    if not name:
        return ""
    
    # 转换为小写
    normalized = name.lower()
    
    # 替换各种分隔符为空格
    # 下划线、连字符、点号等都替换为空格
    normalized = re.sub(r'[_\-\.,;:]', ' ', normalized)
    
    # 移除多余的空格
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # 去除首尾空格
    normalized = normalized.strip()
    
    return normalized


def build_class_name_mapping(dataset_name: str) -> Dict[str, str]:
    """
    为指定数据集构建类别名映射表
    将不同格式的类别名映射到标准格式（来自DATA_STATS）
    
    Args:
        dataset_name: 数据集名称（如 'flower', 'dog', 'bird' 等）
        
    Returns:
        映射字典 {normalized_name: standard_name}
    """
    if dataset_name not in DATA_STATS:
        raise ValueError(f"未知的数据集: {dataset_name}")
    
    standard_names = DATA_STATS[dataset_name]['class_names']
    mapping = {}
    
    # 为每个标准类别名创建映射
    for standard_name in standard_names:
        normalized = normalize_class_name(standard_name)
        mapping[normalized] = standard_name
        
        # 也添加下划线版本的映射（用于discovery集）
        underscore_version = standard_name.replace(' ', '_').replace('-', '_')
        normalized_underscore = normalize_class_name(underscore_version)
        if normalized_underscore not in mapping:
            mapping[normalized_underscore] = standard_name
        
        # 添加全小写版本的映射（用于测试集）
        lowercase_version = standard_name.lower()
        normalized_lower = normalize_class_name(lowercase_version)
        if normalized_lower not in mapping:
            mapping[normalized_lower] = standard_name
    
    return mapping


def map_to_standard_name(folder_name: str, dataset_name: str) -> Optional[str]:
    """
    将文件夹名称映射到标准类别名称
    
    Args:
        folder_name: 文件夹名称（可能包含前缀编号，如下划线分隔等）
        dataset_name: 数据集名称
        
    Returns:
        标准类别名称，如果找不到则返回None
    """
    if dataset_name not in DATA_STATS:
        return None
    
    # 移除前缀编号（如 "000.", "001." 等）
    # 格式可能是: "000.Pink_Primrose" 或 "000 Pink Primrose"
    cleaned_name = folder_name
    if '.' in cleaned_name:
        # 移除 "000." 这样的前缀
        parts = cleaned_name.split('.', 1)
        if len(parts) > 1 and parts[0].isdigit():
            cleaned_name = parts[1]
    elif ' ' in cleaned_name and cleaned_name.split()[0].isdigit():
        # 处理 "000 Pink Primrose" 格式
        parts = cleaned_name.split(' ', 1)
        if len(parts) > 1:
            cleaned_name = parts[1]
    
    # 标准化
    normalized = normalize_class_name(cleaned_name)
    
    # 查找映射
    mapping = build_class_name_mapping(dataset_name)
    standard_name = mapping.get(normalized)
    
    # 如果直接匹配失败，尝试模糊匹配
    if standard_name is None:
        # 尝试在标准名称中查找最相似的
        standard_names = DATA_STATS[dataset_name]['class_names']
        for std_name in standard_names:
            if normalize_class_name(std_name) == normalized:
                standard_name = std_name
                break
    
    return standard_name


def standardize_test_class_name(class_name: str, dataset_name: str) -> str:
    """
    标准化测试集中的类别名称，使其与知识库中的标准格式一致
    
    对于SUN397等有嵌套结构的数据集，需要特殊处理：
    - 如果测试集目录是嵌套的（如 a/abbey），直接使用该路径作为类别名
    - 如果测试集目录是扁平的（如 abbey），需要映射到嵌套格式（如 a/abbey）
    
    Args:
        class_name: 测试集中的类别名称（通常是文件夹名，可能是嵌套路径如 'a/abbey'）
        dataset_name: 数据集名称
        
    Returns:
        标准化后的类别名称
    """
    # SUN397特殊处理：嵌套类别结构
    if dataset_name == 'sun397':
        # 如果class_name已经包含 /，说明是嵌套路径，直接使用
        if '/' in class_name:
            # 验证是否在标准类别列表中
            if class_name in DATA_STATS['sun397']['class_names']:
                return class_name
            # 尝试查找匹配（处理可能的格式差异）
            for std_name in DATA_STATS['sun397']['class_names']:
                if std_name.lower() == class_name.lower():
                    return std_name
            # 如果找不到，返回原始输入
            return class_name
        else:
            # 扁平化的类别名（如 "abbey"），需要映射到嵌套格式（如 "a/abbey"）
            # 查找以该名称结尾的嵌套类别
            for std_name in DATA_STATS['sun397']['class_names']:
                std_name_parts = std_name.split('/')
                if std_name_parts[-1].lower() == class_name.lower():
                    return std_name
            # 如果找不到，返回原始输入
            return class_name
    
    # 其他数据集的原有逻辑
    # 尝试映射到标准名称
    standard_name = map_to_standard_name(class_name, dataset_name)
    
    if standard_name:
        return standard_name
    
    # 如果映射失败，返回标准化后的名称（至少保证格式一致）
    return normalize_class_name(class_name).title()  # 首字母大写


def get_dataset_name_from_key(dataset_key: str) -> Optional[str]:
    """
    从数据集key（如 'flower102', 'dog120'）中提取数据集名称（如 'flower', 'dog'）
    
    Args:
        dataset_key: 数据集key
        
    Returns:
        数据集名称，如果无法识别则返回None
    """
    dataset_key_lower = dataset_key.lower()
    
    # 已知的数据集映射
    known_datasets = ['flower', 'dog', 'bird', 'pet', 'car', 'aircraft', 'eurosat', 'food', 'dtd', 'caltech101', 'caltech256', 'deepfashion_multimodal', 'sun397']
    
    for ds_name in known_datasets:
        if ds_name in dataset_key_lower:
            return ds_name
    
    return None

