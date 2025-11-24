"""
分类结果保存模块
将分类结果保存到各数据集的experiments目录下
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


def save_classification_result(
    dataset_name: str,
    experiment_dir: str,
    results: List[Dict],
    metadata: Optional[Dict] = None
) -> str:
    """
    保存分类结果到指定目录
    
    Args:
        dataset_name: 数据集名称（如 'dog120', 'flower102'）
        experiment_dir: 实验目录（如 './experiments/dog120'）
        results: 结果列表，每个元素为6元组：
            1. label: 正确标签
            2. prediction: 预测结果
            3. is_correct: 是否正确
            4. fast_result: 快思考分类结果
            5. slow_result: 慢思考分类结果（可选）
            6. image_path: 测试图片路径（相对路径）
        metadata: 元数据（如准确率、总样本数等）
        
    Returns:
        保存的文件路径
    """
    # 创建保存目录
    save_dir = os.path.join(experiment_dir, 'classify')
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建保存文件路径
    save_path = os.path.join(save_dir, 'classify_result.json')
    
    # 准备保存的数据
    save_data = {
        'dataset': dataset_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(results),
        'metadata': metadata or {},
        'results': results
    }
    
    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 分类结果已保存到: {save_path}")
    print(f"  总样本数: {len(results)}")
    
    return save_path


def create_result_entry(
    label: str,
    prediction: str,
    is_correct: bool,
    fast_result: Optional[Dict] = None,
    slow_result: Optional[Dict] = None,
    image_path: Optional[str] = None,
    confidence: Optional[float] = None,
    project_root: Optional[str] = None
) -> Dict:
    """
    创建单个分类结果条目（6元组）
    
    6元组格式：
    1. label: 正确标签
    2. prediction: 预测结果
    3. is_correct: 是否正确
    4. fast_result: 快思考分类结果
    5. slow_result: 慢思考分类结果（可选）
    6. image_path: 测试图片路径（相对路径）
    
    Args:
        label: 正确标签
        prediction: 预测结果
        is_correct: 是否正确
        fast_result: 快思考分类结果
        slow_result: 慢思考分类结果
        image_path: 图像路径（绝对路径或相对路径）
        confidence: 置信度
        project_root: 项目根目录，用于将绝对路径转换为相对路径
        
    Returns:
        结果字典（6元组）
    """
    # 处理图片路径：转换为相对路径
    relative_image_path = None
    if image_path:
        if project_root and os.path.isabs(image_path):
            # 如果是绝对路径，转换为相对于项目根目录的相对路径
            try:
                relative_image_path = os.path.relpath(image_path, project_root)
            except ValueError:
                # 如果无法转换为相对路径（例如在不同驱动器上），使用原始路径
                relative_image_path = image_path
        else:
            # 如果已经是相对路径或没有提供project_root，直接使用
            relative_image_path = image_path
    
    entry = {
        'label': label,                    # 1. 正确标签
        'prediction': prediction,           # 2. 预测结果
        'is_correct': is_correct,           # 3. 是否正确
        'fast_result': fast_result or {},   # 4. 快思考分类结果
        'slow_result': slow_result or {},   # 5. 慢思考分类结果
        'image_path': relative_image_path    # 6. 测试图片路径（相对路径）
    }
    
    if confidence is not None:
        entry['confidence'] = confidence
    
    return entry


def get_experiment_dir_from_dataset_info(dataset_info: Dict) -> Optional[str]:
    """
    从dataset_info中获取实验目录
    
    Args:
        dataset_info: 数据集信息字典
        
    Returns:
        实验目录路径，如果无法获取则返回None
    """
    # 优先使用experiment_dir_full
    if 'experiment_dir_full' in dataset_info:
        return dataset_info['experiment_dir_full']
    
    # 其次使用project_root和experiment_dir组合
    if 'project_root' in dataset_info and 'experiment_dir' in dataset_info:
        return os.path.join(
            dataset_info['project_root'],
            'experiments',
            dataset_info['experiment_dir']
        )
    
    # 最后尝试从knowledge_base_dir推断
    if 'knowledge_base_dir' in dataset_info:
        kb_dir = dataset_info['knowledge_base_dir']
        # 假设knowledge_base_dir是 ./experiments/dog120/knowledge_base
        # 需要提取 ./experiments/dog120
        if 'knowledge_base' in kb_dir:
            return os.path.dirname(kb_dir)
    
    return None

