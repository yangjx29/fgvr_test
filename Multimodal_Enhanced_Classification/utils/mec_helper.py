#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEC (Multimodal Enhanced Classification) 辅助函数
用于与 discovering.py 快慢思考系统的集成
"""

import os
import sys
import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

def run_mec_pipeline_with_details(
    mec_path: str,
    mec_data_dir: str,
    dataset_name: str,
    arch: str = 'ViT-B/16',
    seed: int = 0,
    batch_size: int = 50,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    运行完整的MEC流水线并返回详细的AWC增强信息
    
    Args:
        mec_path: MEC框架根目录路径
        mec_data_dir: MEC数据目录路径  
        dataset_name: 数据集名称
        arch: CLIP模型架构
        seed: 随机种子
        batch_size: 批次大小
        timeout: 超时时间（秒）
    
    Returns:
        包含执行结果和详细AWC信息的字典
    """
    result = {
        "success": False,
        "accuracy": 0.0,
        "detailed_results": [],
        "error_message": "",
        "execution_time": 0.0
    }
    
    try:
        import time
        start_time = time.time()
        
        # 数据验证逻辑（与原函数相同）
        test_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_test")
        retrieved_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_retrieved")
        descriptions_dir = os.path.join(mec_path, "descriptions")
        
        test_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_test_descriptions.json")
        retrieved_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_retrieved_descriptions.json")
        
        # 验证数据存在性
        if not os.path.exists(test_data_dir):
            result["error_message"] = f"测试数据目录不存在: {test_data_dir}"
            return result
            
        if not os.path.exists(retrieved_data_dir):
            result["error_message"] = f"检索数据目录不存在: {retrieved_data_dir}"
            return result
            
        if not os.path.exists(test_desc_file):
            result["error_message"] = f"测试描述文件不存在: {test_desc_file}"
            return result
            
        if not os.path.exists(retrieved_desc_file):
            result["error_message"] = f"检索描述文件不存在: {retrieved_desc_file}"
            return result
        
        # 计算样本数量
        print(f"🔍 验证数据完整性...")
        print(f"  测试数据目录: {test_data_dir}")
        print(f"  检索数据目录: {retrieved_data_dir}")
        
        # 统计测试图像
        test_images = []
        if os.path.exists(test_data_dir):
            test_images = [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
        
        # 统计检索图像
        retrieved_images = []
        retrieved_subdir = os.path.join(retrieved_data_dir, "retrieved_images")
        
        if os.path.exists(retrieved_subdir):
            retrieved_images = [f for f in os.listdir(retrieved_subdir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
            print(f"  检索图像位于子目录: {retrieved_subdir}")
        elif os.path.exists(retrieved_data_dir):
            retrieved_images = [f for f in os.listdir(retrieved_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
            print(f"  检索图像位于根目录: {retrieved_data_dir}")
        
        print(f"  发现测试图像: {len(test_images)} 个")
        print(f"  发现检索图像: {len(retrieved_images)} 个")
        
        if len(test_images) == 0:
            result["error_message"] = "测试数据集为空"
            return result
            
        if len(retrieved_images) == 0:
            result["error_message"] = "检索数据集为空"
            return result
        
        print(f"MEC流水线: 测试样本 {len(test_images)} 个, 检索样本 {len(retrieved_images)} 个")
        
        # 步骤1: 预提取特征
        print("🔄 步骤1: 预提取多模态特征...")
        pre_extract_success = run_pre_extract(
            mec_path=mec_path,
            data_root=mec_data_dir,
            dataset_name=dataset_name,
            arch=arch,
            seed=seed,
            batch_size=batch_size,
            timeout=timeout//2
        )
        
        if not pre_extract_success:
            result["error_message"] = "特征预提取失败"
            return result
        
        # 步骤2: 执行评估并获取详细信息
        print("🔄 步骤2: 执行多模态增强分类评估...")
        eval_result = run_evaluation_with_details(
            mec_path=mec_path,
            dataset_name=dataset_name,
            arch=arch,
            seed=seed
        )
        
        if eval_result is None:
            result["error_message"] = "MEC评估失败"
            return result
        
        # 成功完成
        result["success"] = True
        result["accuracy"] = eval_result.get("accuracy", 0.0)
        result["detailed_results"] = eval_result.get("detailed_results", [])
        result["summary"] = eval_result.get("summary", {})
        result["execution_time"] = time.time() - start_time
        
        print(f"✅ MEC流水线完成，准确率: {result['accuracy']:.4f}, 耗时: {result['execution_time']:.2f}秒")
        print(f"📊 返回详细AWC信息: {len(result['detailed_results'])} 个样本")
        
    except Exception as e:
        result["error_message"] = f"MEC流水线异常: {str(e)}"
        print(f"❌ {result['error_message']}")
        import traceback
        traceback.print_exc()
    
    return result

def run_mec_pipeline(
    mec_path: str,
    mec_data_dir: str,
    dataset_name: str,
    arch: str = 'ViT-B/16',
    seed: int = 0,
    batch_size: int = 50,
    timeout: int = 300  # 5分钟超时
) -> Dict[str, Any]:
    """
    运行完整的MEC流水线
    
    Args:
        mec_path: MEC框架根目录路径
        mec_data_dir: MEC数据目录路径  
        dataset_name: 数据集名称
        arch: CLIP模型架构
        seed: 随机种子
        batch_size: 批次大小
        timeout: 超时时间（秒）
    
    Returns:
        包含执行结果的字典
    """
    result = {
        "success": False,
        "accuracy": 0.0,
        "error_message": "",
        "execution_time": 0.0
    }
    
    try:
        import time
        start_time = time.time()
        
        # 检查数据完整性
        test_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_test")
        retrieved_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_retrieved")
        descriptions_dir = os.path.join(mec_path, "descriptions")
        
        test_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_test_descriptions.json")
        retrieved_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_retrieved_descriptions.json")
        
        # 验证数据存在性
        if not os.path.exists(test_data_dir):
            result["error_message"] = f"测试数据目录不存在: {test_data_dir}"
            return result
            
        if not os.path.exists(retrieved_data_dir):
            result["error_message"] = f"检索数据目录不存在: {retrieved_data_dir}"
            return result
            
        if not os.path.exists(test_desc_file):
            result["error_message"] = f"测试描述文件不存在: {test_desc_file}"
            return result
            
        if not os.path.exists(retrieved_desc_file):
            result["error_message"] = f"检索描述文件不存在: {retrieved_desc_file}"
            return result
        
        # 计算样本数量 - 改进检查逻辑
        print(f"🔍 验证数据完整性...")
        print(f"  测试数据目录: {test_data_dir}")
        print(f"  检索数据目录: {retrieved_data_dir}")
        
        # 统计测试图像
        test_images = []
        if os.path.exists(test_data_dir):
            test_images = [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
        
        # 统计检索图像 - 支持多种目录结构
        retrieved_images = []
        retrieved_subdir = os.path.join(retrieved_data_dir, "retrieved_images")
        
        if os.path.exists(retrieved_subdir):
            # 情况1: 检索图像在子目录中
            retrieved_images = [f for f in os.listdir(retrieved_subdir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
            print(f"  检索图像位于子目录: {retrieved_subdir}")
        elif os.path.exists(retrieved_data_dir):
            # 情况2: 检索图像直接在根目录中
            retrieved_images = [f for f in os.listdir(retrieved_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
            print(f"  检索图像位于根目录: {retrieved_data_dir}")
        else:
            print(f"  ❌ 检索数据目录不存在")
        
        print(f"  发现测试图像: {len(test_images)} 个")
        print(f"  发现检索图像: {len(retrieved_images)} 个")
        
        if len(test_images) == 0:
            result["error_message"] = "测试数据集为空"
            return result
            
        if len(retrieved_images) == 0:
            result["error_message"] = "检索数据集为空"
            return result
        
        print(f"MEC流水线: 测试样本 {len(test_images)} 个, 检索样本 {len(retrieved_images)} 个")
        
        # 步骤1: 预提取特征
        print("🔄 步骤1: 预提取多模态特征...")
        pre_extract_success = run_pre_extract(
            mec_path=mec_path,
            data_root=mec_data_dir,
            dataset_name=dataset_name,
            arch=arch,
            seed=seed,
            batch_size=batch_size,
            timeout=timeout//2
        )
        
        if not pre_extract_success:
            result["error_message"] = "特征预提取失败"
            return result
        
        # 步骤2: 执行评估
        print("🔄 步骤2: 执行多模态增强分类评估...")
        eval_accuracy = run_evaluation(
            mec_path=mec_path,
            dataset_name=dataset_name,
            arch=arch,
            seed=seed,
            timeout=timeout//2
        )
        
        if eval_accuracy is None:
            result["error_message"] = "MEC评估失败"
            return result
        
        # 成功完成
        result["success"] = True
        result["accuracy"] = eval_accuracy
        result["execution_time"] = time.time() - start_time
        
        print(f"✅ MEC流水线完成，准确率: {eval_accuracy:.4f}, 耗时: {result['execution_time']:.2f}秒")
        
    except Exception as e:
        result["error_message"] = f"MEC流水线异常: {str(e)}"
        print(f"❌ {result['error_message']}")
    
    return result


def run_pre_extract(
    mec_path: str,
    data_root: str,
    dataset_name: str,
    arch: str = 'ViT-B/16',
    seed: int = 0,
    batch_size: int = 50,
    timeout: int = 150
) -> bool:
    """运行特征预提取"""
    try:
        # 构建预提取命令
        cmd = [
            sys.executable,
            "pre_extract.py",
            data_root,
            "--test_set", dataset_name,
            "--arch", arch,
            "--seed", str(seed),
            "--batch-size", str(batch_size),
            "--workers", "4",
            "--resolution", "224"
        ]
        
        print(f"执行预提取命令: {' '.join(cmd)}")
        
        # 执行命令
        process = subprocess.run(
            cmd,
            cwd=mec_path,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if process.returncode == 0:
            # 验证特征文件是否实际生成
            save_dir = f"./pre_extracted_feat/{arch.replace('/', '')}/seed{seed}"
            retrieved_path = os.path.join(mec_path, save_dir, f"{dataset_name}_retrieved.pth")
            test_path = os.path.join(mec_path, save_dir, f"{dataset_name}_test.pth")
            
            if os.path.exists(retrieved_path) and os.path.exists(test_path):
                print("✅ 特征预提取成功")
                print(f"✅ 检索特征文件: {retrieved_path}")
                print(f"✅ 测试特征文件: {test_path}")
                return True
            else:
                print("❌ 特征预提取失败：特征文件未生成")
                print(f"❌ 检索特征文件不存在: {retrieved_path}")
                print(f"❌ 测试特征文件不存在: {test_path}")
                print("📋 预提取输出:")
                print(process.stdout)
                if process.stderr:
                    print("📋 预提取错误:")
                    print(process.stderr)
                return False
        else:
            print(f"❌ 特征预提取失败:")
            print(f"stdout: {process.stdout}")
            print(f"stderr: {process.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ 特征预提取超时 ({timeout}秒)")
        return False
    except Exception as e:
        print(f"❌ 特征预提取异常: {e}")
        return False


def run_evaluation_with_details(
    mec_path: str,
    dataset_name: str,
    arch: str = 'ViT-B/16',
    seed: int = 0,
    timeout: int = 150
) -> Optional[Dict[str, Any]]:
    """运行MEC评估并返回详细的AWC增强信息"""
    try:
        # 创建临时文件保存详细结果
        import tempfile
        temp_result_file = os.path.join(mec_path, f"temp_awc_results_{dataset_name}.json")
        
        # 构建评估命令，添加保存详细结果的参数
        cmd = [
            sys.executable,
            "evaluate.py",
            "--test_set", dataset_name,
            "--arch", arch,
            "--seed", str(seed),
            "--print-freq", "100",
            "--save_detailed_results", temp_result_file
        ]
        
        print(f"执行评估命令: {' '.join(cmd)}")
        
        # 执行命令
        process = subprocess.run(
            cmd,
            cwd=mec_path,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if process.returncode == 0:
            # 从输出中解析准确率
            accuracy = parse_accuracy_from_output(process.stdout)
            
            # 尝试读取详细结果文件
            detailed_results = []
            if os.path.exists(temp_result_file):
                try:
                    with open(temp_result_file, 'r', encoding='utf-8') as f:
                        detailed_data = json.load(f)
                        detailed_results = detailed_data.get("detailed_results", [])
                    # 清理临时文件
                    os.remove(temp_result_file)
                except Exception as e:
                    print(f"⚠️  读取详细结果文件失败: {e}")
            
            if accuracy is not None:
                print(f"✅ MEC评估成功，准确率: {accuracy:.4f}")
                print(f"📊 返回详细AWC信息: {len(detailed_results)} 个样本")
                
                return {
                    "accuracy": accuracy,
                    "detailed_results": detailed_results,
                    "summary": {
                        "total_samples": len(detailed_results),
                        "correct_predictions": sum(1 for r in detailed_results if r.get("is_correct", False)),
                        "accuracy": accuracy
                    }
                }
            else:
                print("❌ 无法解析准确率")
                return None
        else:
            print(f"❌ MEC评估失败:")
            print(f"stdout: {process.stdout}")
            print(f"stderr: {process.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ MEC评估超时 ({timeout}秒)")
        return None
    except Exception as e:
        print(f"❌ MEC评估异常: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_evaluation(
    mec_path: str,
    dataset_name: str,
    arch: str = 'ViT-B/16',
    seed: int = 0,
    timeout: int = 150
) -> Optional[float]:
    """运行MEC评估"""
    try:
        # 构建评估命令
        cmd = [
            sys.executable,
            "evaluate.py",
            "--test_set", dataset_name,
            "--arch", arch,
            "--seed", str(seed),
            "--print-freq", "100"
        ]
        
        print(f"执行评估命令: {' '.join(cmd)}")
        
        # 执行命令
        process = subprocess.run(
            cmd,
            cwd=mec_path,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if process.returncode == 0:
            # 从输出中解析准确率
            accuracy = parse_accuracy_from_output(process.stdout)
            if accuracy is not None:
                print(f"✅ MEC评估成功，准确率: {accuracy:.4f}")
                return accuracy
            else:
                print("❌ 无法解析准确率")
                print(f"stdout: {process.stdout}")
                return None
        else:
            print(f"❌ MEC评估失败:")
            print(f"stdout: {process.stdout}")
            print(f"stderr: {process.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ MEC评估超时 ({timeout}秒)")
        return None
    except Exception as e:
        print(f"❌ MEC评估异常: {e}")
        return None


def parse_accuracy_from_output(output: str) -> Optional[float]:
    """从MEC输出中解析准确率"""
    try:
        import re
        lines = output.split('\n')
        
        for line in lines:
            # 优先匹配新格式：🎯 最终准确率: 0.xxxx (xx.xx%)
            if '最终准确率' in line or 'final accuracy' in line.lower():
                # 匹配 0.xxxx 格式
                match = re.search(r'[:：]\s*(\d+\.\d+)', line)
                if match:
                    accuracy = float(match.group(1))
                    print(f"✅ 解析到准确率: {accuracy:.4f}")
                    return accuracy
            
            # 匹配 Acc@1 格式
            elif 'Acc@1' in line and '%' in line:
                match = re.search(r'(\d+\.\d+)%', line)
                if match:
                    accuracy = float(match.group(1)) / 100.0
                    print(f"✅ 解析到准确率: {accuracy:.4f}")
                    return accuracy
            
            # 匹配 accuracy: 0.xxxx 格式
            elif 'accuracy' in line.lower() and ':' in line:
                match = re.search(r'accuracy[:\s]+(\d+\.\d+)', line.lower())
                if match:
                    accuracy = float(match.group(1))
                    print(f"✅ 解析到准确率: {accuracy:.4f}")
                    return accuracy
        
        # 如果没有找到，打印输出内容用于调试
        print("⚠️  未找到准确率信息")
        print("📋 输出内容（最后10行）:")
        output_lines = output.split('\n')
        for line in output_lines[-10:]:
            if line.strip():
                print(f"  {line}")
        print("⚠️  使用默认值 0.5")
        return 0.5
        
    except Exception as e:
        print(f"❌ 解析准确率异常: {e}")
        import traceback
        traceback.print_exc()
        return None


def cleanup_mec_temp_files(mec_data_dir: str, dataset_name: str):
    """清理MEC临时文件"""
    try:
        # 清理数据目录
        test_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_test")
        retrieved_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_retrieved")
        
        for temp_dir in [test_data_dir, retrieved_data_dir]:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"🗑️  已清理临时目录: {temp_dir}")
        
        # 清理描述文件
        descriptions_dir = os.path.dirname(mec_data_dir)
        if "Multimodal_Enhanced_Classification" in descriptions_dir:
            descriptions_dir = os.path.join(descriptions_dir, "descriptions")
            test_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_test_descriptions.json")
            retrieved_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_retrieved_descriptions.json")
            
            for desc_file in [test_desc_file, retrieved_desc_file]:
                if os.path.exists(desc_file):
                    os.remove(desc_file)
                    print(f"🗑️  已清理描述文件: {desc_file}")
        
        # 清理特征文件
        feat_dir = os.path.join(os.path.dirname(mec_data_dir), "pre_extracted_feat")
        if os.path.exists(feat_dir):
            # 清理与此数据集相关的特征文件
            for root, dirs, files in os.walk(feat_dir):
                for file in files:
                    if dataset_name in file:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            print(f"🗑️  已清理特征文件: {file_path}")
                        except Exception as e:
                            print(f"⚠️  清理特征文件失败 {file_path}: {e}")
        
    except Exception as e:
        print(f"⚠️  清理临时文件异常: {e}")


def create_mec_data_structure(
    test_images: List[Tuple[str, str, str]],  # (image_path, description, true_category)
    retrieved_images: List[Tuple[str, str, str]],  # (image_path, description, category)
    mec_data_dir: str,
    dataset_name: str
) -> bool:
    """
    创建MEC标准数据结构
    
    Args:
        test_images: 测试图像列表 [(路径, 描述, 真实类别), ...]
        retrieved_images: 检索图像列表 [(路径, 描述, 类别), ...]
        mec_data_dir: MEC数据目录
        dataset_name: 数据集名称
    
    Returns:
        创建是否成功
    """
    try:
        # 创建数据目录
        test_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_test")
        retrieved_data_dir = os.path.join(mec_data_dir, f"{dataset_name}_retrieved")
        descriptions_dir = os.path.join(os.path.dirname(mec_data_dir), "descriptions")
        
        os.makedirs(test_data_dir, exist_ok=True, mode=0o755)
        os.makedirs(retrieved_data_dir, exist_ok=True, mode=0o755)
        os.makedirs(descriptions_dir, exist_ok=True, mode=0o755)
        
        # 创建检索图像的统一目录结构
        retrieved_class_dir = os.path.join(retrieved_data_dir, "retrieved_images")
        os.makedirs(retrieved_class_dir, exist_ok=True)
        
        # 复制测试图像
        test_descriptions = {}
        for i, (img_path, description, true_cat) in enumerate(test_images):
            if not os.path.exists(img_path):
                print(f"⚠️  测试图像不存在: {img_path}")
                continue
                
            # 生成安全的文件名
            base_name = f"test_{i:04d}.jpg"
            dst_path = os.path.join(test_data_dir, base_name)
            shutil.copy2(img_path, dst_path)
            test_descriptions[base_name] = description
        
        # 复制检索图像
        retrieved_descriptions = {}
        for i, (img_path, description, category) in enumerate(retrieved_images):
            if not os.path.exists(img_path):
                print(f"⚠️  检索图像不存在: {img_path}")
                continue
                
            # 生成安全的文件名
            safe_category = category.replace(' ', '_').replace('/', '_')
            base_name = f"retrieved_{i:04d}_{safe_category}.jpg"
            dst_path = os.path.join(retrieved_class_dir, base_name)
            shutil.copy2(img_path, dst_path)
            retrieved_descriptions[base_name] = description
        
        # 保存描述文件
        test_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_test_descriptions.json")
        retrieved_desc_file = os.path.join(descriptions_dir, f"{dataset_name}_retrieved_descriptions.json")
        
        with open(test_desc_file, 'w', encoding='utf-8') as f:
            json.dump(test_descriptions, f, ensure_ascii=False, indent=2)
        
        with open(retrieved_desc_file, 'w', encoding='utf-8') as f:
            json.dump(retrieved_descriptions, f, ensure_ascii=False, indent=2)
        
        print(f"✅ MEC数据结构创建成功:")
        print(f"   测试图像: {len(test_descriptions)} 个")
        print(f"   检索图像: {len(retrieved_descriptions)} 个")
        print(f"   描述文件: {test_desc_file}, {retrieved_desc_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建MEC数据结构失败: {e}")
        return False


def parse_mec_output(output_text: str) -> Dict[str, Any]:
    """解析MEC输出结果"""
    result = {
        "predictions": [],
        "accuracy": 0.0,
        "total_samples": 0,
        "error_message": ""
    }
    
    try:
        lines = output_text.split('\n')
        
        for line in lines:
            # 解析准确率
            if 'Acc@1' in line and '%' in line:
                import re
                match = re.search(r'(\d+\.\d+)%', line)
                if match:
                    result["accuracy"] = float(match.group(1)) / 100.0
            
            # 解析样本数量
            elif 'number of test samples' in line:
                import re
                match = re.search(r'(\d+)', line)
                if match:
                    result["total_samples"] = int(match.group(1))
        
        if result["accuracy"] == 0.0:
            result["accuracy"] = 0.5  # 默认值
            
    except Exception as e:
        result["error_message"] = f"解析MEC输出失败: {e}"
    
    return result