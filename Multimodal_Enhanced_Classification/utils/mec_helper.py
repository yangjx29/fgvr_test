"""
MEC集成辅助函数
为discovering.py提供MEC集成支持
"""

import os
import json
import subprocess
import torch
from typing import Dict, List, Tuple, Optional

def parse_mec_output(stdout: str, stderr: str) -> Dict:
    """
    解析MEC输出，提取准确率和其他统计信息
    
    Args:
        stdout: MEC程序的标准输出
        stderr: MEC程序的错误输出
    
    Returns:
        Dict: 包含解析结果的字典
    """
    result = {
        "accuracy": 0.0,
        "success": False,
        "error_message": "",
        "output_lines": stdout.split('\n') if stdout else []
    }
    
    if stderr:
        result["error_message"] = stderr
        return result
    
    try:
        lines = stdout.split('\n')
        for line in lines:
            # 查找准确率信息
            if 'Acc@1' in line and '%' in line:
                # 提取百分比数字
                import re
                match = re.search(r'(\d+\.?\d*)%', line)
                if match:
                    result["accuracy"] = float(match.group(1)) / 100.0
                    result["success"] = True
                    break
    except Exception as e:
        result["error_message"] = str(e)
    
    return result

def create_mec_data_structure(test_samples: List, retrieved_samples: List, 
                              test_descriptions: Dict, retrieved_descriptions: Dict,
                              mec_data_dir: str, dataset_name: str) -> bool:
    """
    为MEC创建标准的数据结构
    
    Args:
        test_samples: 测试样本列表
        retrieved_samples: 检索样本列表  
        test_descriptions: 测试描述字典
        retrieved_descriptions: 检索描述字典
        mec_data_dir: MEC数据目录
        dataset_name: 数据集名称
    
    Returns:
        bool: 是否创建成功
    """
    try:
        # 创建数据目录结构
        test_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_test")
        retrieved_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_retrieved")
        
        os.makedirs(test_data_dir, exist_ok=True, mode=0o755)
        os.makedirs(retrieved_data_dir, exist_ok=True, mode=0o755)
        
        # 为兼容ImageFolder，创建虚拟类别目录
        # 所有图像都放在同一个类别目录下（类别为0）
        test_class_dir = os.path.join(test_data_dir, "0")
        retrieved_class_dir = os.path.join(retrieved_data_dir, "0")
        
        os.makedirs(test_class_dir, exist_ok=True, mode=0o755)
        os.makedirs(retrieved_class_dir, exist_ok=True, mode=0o755)
        
        # 复制测试图像到类别目录
        for sample in test_samples:
            src_path = sample["path"]
            dst_name = sample["name"]
            dst_path = os.path.join(test_class_dir, dst_name)
            
            if os.path.exists(src_path):
                import shutil
                shutil.copy2(src_path, dst_path)
        
        # 复制检索图像到类别目录
        for sample in retrieved_samples:
            src_path = sample["path"]
            dst_name = sample["name"]
            dst_path = os.path.join(retrieved_class_dir, dst_name)
            
            if os.path.exists(src_path):
                import shutil
                shutil.copy2(src_path, dst_path)
        
        # 保存描述文件
        descriptions_dir = os.path.join(os.path.dirname(mec_data_dir), 'descriptions')
        os.makedirs(descriptions_dir, exist_ok=True)
        
        test_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_test_descriptions.json")
        retrieved_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_retrieved_descriptions.json")
        
        with open(test_desc_file, 'w', encoding='utf-8') as f:
            json.dump(test_descriptions, f, ensure_ascii=False, indent=2)
            
        with open(retrieved_desc_file, 'w', encoding='utf-8') as f:
            json.dump(retrieved_descriptions, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ 创建MEC数据结构失败: {e}")
        return False

def cleanup_mec_temp_files(mec_data_dir: str, dataset_name: str):
    """
    清理MEC临时文件
    """
    import shutil
    
    try:
        # 清理数据目录
        test_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_test")
        retrieved_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_retrieved")
        
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        if os.path.exists(retrieved_data_dir):
            shutil.rmtree(retrieved_data_dir)
        
        # 清理描述文件
        descriptions_dir = os.path.join(os.path.dirname(mec_data_dir), 'descriptions')
        test_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_test_descriptions.json")
        retrieved_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_retrieved_descriptions.json")
        
        if os.path.exists(test_desc_file):
            os.remove(test_desc_file)
        if os.path.exists(retrieved_desc_file):
            os.remove(retrieved_desc_file)
            
        # 清理特征文件
        feat_dir = os.path.join(os.path.dirname(mec_data_dir), 'pre_extracted_feat/ViT-B16/seed0')
        test_feat_file = os.path.join(feat_dir, f"{dataset_name}_test.pth")
        retrieved_feat_file = os.path.join(feat_dir, f"{dataset_name}_retrieved.pth")
        
        if os.path.exists(test_feat_file):
            os.remove(test_feat_file)
        if os.path.exists(retrieved_feat_file):
            os.remove(retrieved_feat_file)
            
        print(f"✅ 已清理MEC临时文件: {dataset_name}")
        
    except Exception as e:
        print(f"⚠️  清理MEC临时文件失败: {e}")

def run_mec_pipeline(mec_path: str, mec_data_dir: str, dataset_name: str, 
                     arch: str = "ViT-B/16", seed: int = 0, batch_size: int = 50) -> Dict:
    """
    运行完整的MEC流水线
    
    Args:
        mec_path: MEC代码路径
        mec_data_dir: MEC数据目录
        dataset_name: 数据集名称
        arch: CLIP模型架构
        seed: 随机种子
        batch_size: 批处理大小/视图数量
    
    Returns:
        Dict: MEC运行结果
    """
    result = {
        "success": False,
        "accuracy": 0.0,
        "error_message": "",
        "pre_extract_output": "",
        "evaluate_output": ""
    }
    
    try:
        # 步骤1: 特征预提取
        print(f"🚀 MEC步骤1: 预提取特征...")
        pre_extract_cmd = [
            'python', 'pre_extract.py',
            mec_data_dir,
            '--test_set', dataset_name,
            '--arch', arch,
            '--batch-size', str(batch_size),
            '--seed', str(seed)
        ]
        
        pre_result = subprocess.run(pre_extract_cmd, capture_output=True, text=True, cwd=mec_path)
        result["pre_extract_output"] = pre_result.stdout
        
        if pre_result.returncode != 0:
            result["error_message"] = f"特征提取失败: {pre_result.stderr}"
            return result
        
        print("✅ MEC特征提取成功")
        
        # 步骤2: 多模态评估
        print(f"🔍 MEC步骤2: 多模态评估...")
        evaluate_cmd = [
            'python', 'evaluate.py',
            '--test_set', dataset_name,
            '--arch', arch,
            '--seed', str(seed)
        ]
        
        eval_result = subprocess.run(evaluate_cmd, capture_output=True, text=True, cwd=mec_path)
        result["evaluate_output"] = eval_result.stdout
        
        if eval_result.returncode != 0:
            result["error_message"] = f"评估失败: {eval_result.stderr}"
            return result
        
        print("✅ MEC评估成功")
        
        # 解析结果
        mec_output = parse_mec_output(eval_result.stdout, eval_result.stderr)
        result.update(mec_output)
        
        return result
        
    except Exception as e:
        result["error_message"] = f"MEC流水线异常: {str(e)}"
        return result
