import warnings
# 抑制常见的警告信息
warnings.filterwarnings('ignore', message='.*Failed to load image Python extension.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Using a slow image processor.*', category=UserWarning)

import torch 
import os 
import argparse 
import json 
from tqdm import tqdm  
from termcolor import colored  
from collections import Counter 
from utils.configuration import setup_config, seed_everything 
from utils.fileios import dump_json, load_json, dump_txt  

from data import DATA_STATS, PROMPTERS, DATA_DISCOVERY  
from data.extract_from_testsets import get_test_images_by_percentage, validate_test_set, get_dataset_key_for_test  
from agents.vqa_bot import VQABot  
from agents.llm_bot import LLMBot 
from agents.mllm_bot import MLLMBot
from cvd.cdv_captioner import CDVCaptioner  
from retrieval.multimodal_retrieval import MultimodalRetrieval 
from fast_slow_thinking_system import FastSlowThinkingSystem
from utils.util import is_similar
import re 
import hashlib
from collections import defaultdict
import numpy as np
import yaml
import sys
import subprocess

import pprint
import time

# Import the extract_from_trainsets module
from data.extract_from_trainsets import load_train_data, extract_discovery_set, save_discovery_set

test_data_true_random =True # 测试集采样是否实现真随机，每次运行结果都不一样
randomly_extract_discoverying_set=True #控制是否从训练集中随机抽取图片来创建发现集，而不是直接加载发现集
discovery_data_true_random=True #发现集采样是否实现真随机，每次运行结果都不一样
DEBUG = False  # 设置调试模式为关闭状态

# 全局数据集配置
DATASET_CONFIG = None
CURRENT_DATASET = None

def get_or_create_discovery_set(cfg, folder_suffix='', args=None):
    """
    获取或创建发现集，根据randomly_extract_discoverying_set标志决定模式
    
    Args:
        cfg: 配置字典
        folder_suffix: 后缀（如 '_1', '_2', '_random'）
        args: 命令行参数对象
    
    Returns:
        DATA_DISCOVERY object
    """
    global randomly_extract_discoverying_set
    dataset_name = cfg['dataset_name']
    
    if randomly_extract_discoverying_set:
        # 从训练集随机抽取模式
        print(f"🔄 从训练集随机抽取发现集 (模式: {folder_suffix})")
        
        # 解析folder_suffix确定抽取数量
        if folder_suffix == '_random':
            num_per_category = 'random'
            output_suffix = 'random'
        else:
            try:
                # 从 '_k' 格式中提取k值
                num_per_category = int(folder_suffix[1:])  # 去掉下划线
                output_suffix = folder_suffix[1:]  # 去掉下划线
            except (ValueError, IndexError):
                print(f"⚠️ 无法解析后缀 '{folder_suffix}'，使用默认值1")
                num_per_category = 1
                output_suffix = '1'
        
        # 调用data目录下的抽取函数
        try:
            from data.extract_from_trainsets import extract_and_save_discovery_set
            
            # 根据当前模式确定抽取类型
            extract_type = 'knowledge_base' if (args and args.mode == 'build_knowledge_base') else 'fast_slow'
            
            json_discovery = extract_and_save_discovery_set(
                dataset_name, 
                num_per_category, 
                cfg.get('seed', None if discovery_data_true_random else 42),
                output_suffix,
                extract_type
            )
            
            print(f"✓ 创建JSON发现集对象: {len(json_discovery.subcat_to_sample)} 个类别")
            return json_discovery
            
        except Exception as e:
            print(f"❌ 从训练集抽取发现集失败: {e}")
            print("🔄 回退到使用现有发现集")
            randomly_extract_discoverying_set = False
            return DATA_DISCOVERY[dataset_name](cfg, folder_suffix=folder_suffix)
    
    else:
        # 使用现有发现集模式
        print(f"📁 使用现有发现集 (后缀: {folder_suffix})")
        return DATA_DISCOVERY[dataset_name](cfg, folder_suffix=folder_suffix)


def check_and_generate_json_files(dataset_name):
    """
    检查实验目录下的JSON文件是否存在，如果不存在则自动运行copy_datasets_json.py生成
    
    Args:
        dataset_name: 数据集名称 (dog, bird, flower, pet, car, aircraft等)
    """
    # 加载数据集配置
    if DATASET_CONFIG is None:
        load_dataset_config()
    
    dataset_info = DATASET_CONFIG['dataset_mapping'].get(dataset_name)
    if not dataset_info:
        print(f"Warning: Dataset '{dataset_name}' not found in configuration")
        return
    
    experiments_root = DATASET_CONFIG.get('experiments_root', './experiments')
    experiment_dir = dataset_info['experiment_dir']
    json_dir = os.path.join(experiments_root, experiment_dir, 'images_split')
    
    # 检查关键JSON文件是否存在
    critical_files = ['images_test.json', 'images_discovery_all_1.json']
    missing_files = []
    
    for json_file in critical_files:
        json_path = os.path.join(json_dir, json_file)
        if not os.path.exists(json_path):
            missing_files.append(json_file)
    
    if missing_files:
        print(f"⚠️  检测到缺失的JSON文件: {', '.join(missing_files)}")
        print(f"📁 实验目录: {json_dir}")
        print("🔄 正在自动运行copy_datasets_json.py生成JSON文件...")
        
        try:
            # 运行copy_datasets_json.py脚本
            script_path = os.path.join(os.path.dirname(__file__), 'data', 'copy_datasets_json.py')
            cmd = [sys.executable, script_path, '--dataset', dataset_name]
            
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(__file__),  # 在项目根目录运行
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                print("✅ JSON文件生成成功!")
                # 再次检查文件是否生成
                still_missing = []
                for json_file in missing_files:
                    json_path = os.path.join(json_dir, json_file)
                    if not os.path.exists(json_path):
                        still_missing.append(json_file)
                
                if still_missing:
                    print(f"⚠️  以下文件仍未生成: {', '.join(still_missing)}")
                else:
                    print("✅ 所有必需的JSON文件已就绪")
            else:
                print(f"❌ JSON文件生成失败:")
                print(f"错误输出: {result.stderr}")
                print(f"标准输出: {result.stdout}")
                
        except subprocess.TimeoutExpired:
            print("❌ JSON文件生成超时，请手动运行:")
            print(f"python data/copy_datasets_json.py --dataset {dataset_name}")
        except Exception as e:
            print(f"❌ 运行copy_datasets_json.py时出错: {e}")
            print("请手动运行:")
            print(f"python data/copy_datasets_json.py --dataset {dataset_name}")
    else:
        print(f"✅ JSON文件检查通过: {json_dir}")

def load_dataset_config():
    """加载数据集配置文件"""
    global DATASET_CONFIG
    config_path = os.path.join(os.path.dirname(__file__), "configs", "datasets_list.yml")
    with open(config_path, 'r', encoding='utf-8') as f:
        DATASET_CONFIG = yaml.safe_load(f)
    return DATASET_CONFIG

def get_dataset_info(dataset_name: str) -> dict:
    """
    获取数据集信息
    
    Args:
        dataset_name: 数据集名称 (dog, bird, flower, pet, car)
        
    Returns:
        dict: 数据集配置信息
    """
    global DATASET_CONFIG
    if DATASET_CONFIG is None:
        load_dataset_config()
    
    if dataset_name not in DATASET_CONFIG['dataset_mapping']:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIG['dataset_mapping'].keys())}")
    
    dataset_info = DATASET_CONFIG['dataset_mapping'][dataset_name].copy()
    # 添加实验根目录
    experiments_root = DATASET_CONFIG.get('experiments_root', './experiments')
    dataset_info['experiments_root'] = experiments_root
    # 构建完整的stats文件路径
    dataset_info['stats_file_full'] = os.path.join(experiments_root, dataset_info['stats_file'])
    # 构建完整的实验目录路径
    dataset_info['experiment_dir_full'] = os.path.join(experiments_root, dataset_info['experiment_dir'])
    
    return dataset_info

def set_current_dataset(dataset_name: str):
    """设置当前数据集"""
    global CURRENT_DATASET
    CURRENT_DATASET = get_dataset_info(dataset_name)
    print(f"当前数据集: {dataset_name} ({CURRENT_DATASET['full_name']})")
    print(f"类别数: {CURRENT_DATASET['num_classes']}")
    print(f"实验目录: {CURRENT_DATASET['experiment_dir_full']}")
    return CURRENT_DATASET


def load_test_data_from_json(json_file_path: str, dataset_name: str = None) -> dict:
    """从JSON文件加载测试数据"""
    print(f"从JSON文件加载测试数据: {json_file_path}")
    
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"测试数据JSON文件不存在: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_samples = defaultdict(list)
    
    # 获取数据集名称用于类别名标准化
    from data.class_name_mapper import standardize_test_class_name
    
    for class_entry in data:
        if len(class_entry) >= 3:
            class_name, class_id, image_paths = class_entry[0], class_entry[1], class_entry[2]
            
            # 标准化类别名称
            if dataset_name:
                standardized_class_name = standardize_test_class_name(class_name, dataset_name)
            else:
                standardized_class_name = class_name
            
            # 添加所有图像路径
            for img_path in image_paths:
                test_samples[standardized_class_name].append(img_path)
    
    total_images = sum(len(paths) for paths in test_samples.values())
    print(f"✓ 从JSON文件加载 {len(test_samples)} 个类别，共 {total_images} 张图像")
    
    return dict(test_samples)


def prepare_test_samples(cfg, args):
    """
    准备测试样本，支持discovery集和test集
    
    Args:
        cfg: 配置字典
        args: 命令行参数
        
    Returns:
        dict: 测试样本字典 {class_name: [image_paths]}
    """
    test_samples = defaultdict(list)
    
    # 如果使用测试集
    if args.use_test_data:
        dataset_key = get_dataset_key_for_test(cfg)
        data_root = cfg.get('data_root', './datasets')
        
        print("="*70)
        print(colored("📊 测试数据来源: images_test (测试集)", "cyan", attrs=['bold']))
        print("="*70)
        print(f"数据集标识: {dataset_key}")
        print(f"数据集根目录: {data_root}")
        print(f"测试百分比: {args.test_percentage}%")
        print(f"说明: 使用完整测试集的 {args.test_percentage}% 进行评估")
        print("="*70)
        
        # 验证测试集
        try:
            # 获取数据集配置信息
            dataset_info = get_dataset_info(cfg.get('dataset_name'))
            experiments_root = dataset_info.get('experiments_root', './experiments')
            experiment_dir = dataset_info.get('experiment_dir', cfg.get('dataset_name'))
            
            # 检查是否有JSON测试文件
            json_test_file = os.path.join(experiments_root, experiment_dir, 'images_split', 'images_test.json')
            
            if os.path.exists(json_test_file):
                # 使用JSON文件验证
                with open(json_test_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                
                total_images = sum(len(entry[2]) if len(entry) >= 3 and isinstance(entry[2], list) else 1 
                                  for entry in test_data)
                num_classes = len(set(entry[0] for entry in test_data))
                
                test_stats = {
                    'test_dir': json_test_file,
                    'num_classes': num_classes,
                    'total_images': total_images,
                    'avg_images_per_class': total_images / num_classes if num_classes > 0 else 0,
                    'min_images_per_class': min(len(entry[2]) if len(entry) >= 3 and isinstance(entry[2], list) else 1 
                                              for entry in test_data),
                    'max_images_per_class': max(len(entry[2]) if len(entry) >= 3 and isinstance(entry[2], list) else 1 
                                              for entry in test_data)
                }
                
                print(f"✓ 测试集验证成功（JSON文件）")
                print(f"  测试集文件: {test_stats['test_dir']}")
                print(f"  类别数量: {test_stats['num_classes']}")
                print(f"  总图像数: {test_stats['total_images']}")
                print(f"  平均每类: {test_stats['avg_images_per_class']:.1f} 张")
                print(f"  图像范围: {test_stats['min_images_per_class']} - {test_stats['max_images_per_class']} 张/类")
            else:
                # 使用原有目录验证
                test_stats = validate_test_set(dataset_key, data_root)
                print(f"✓ 测试集验证成功（目录）")
                print(f"  测试集目录: {test_stats['test_dir']}")
                print(f"  类别数量: {test_stats['num_classes']}")
                print(f"  总图像数: {test_stats['total_images']}")
                print(f"  平均每类: {test_stats['avg_images_per_class']:.1f} 张")
                print(f"  图像范围: {test_stats['min_images_per_class']} - {test_stats['max_images_per_class']} 张/类")
        except Exception as e:
            print(colored(f"❌ 测试集验证失败: {e}", "red"))
            raise ValueError(f"测试集验证失败: {e}")
        
        # 获取采样的测试图像
        # 使用全局变量test_data_true_random控制是否使用真随机
        if os.path.exists(json_test_file):
            # 从JSON文件采样
            print("从JSON文件加载测试数据...")
            with open(json_test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            # 展平所有图像路径
            all_images = []
            for entry in test_data:
                if len(entry) >= 3:
                    class_name, class_id, image_paths = entry[0], entry[1], entry[2]
                    if isinstance(image_paths, list):
                        for img_path in image_paths:
                            all_images.append((img_path, class_name))
                    else:
                        all_images.append((image_paths, class_name))
            
            # 根据百分比采样
            if args.test_percentage >= 100.0:
                sampled_images = all_images
            else:
                import random
                random.seed(cfg.get('seed', 42) if not test_data_true_random else None)
                sample_count = max(1, int(len(all_images) * args.test_percentage / 100))
                sampled_images = random.sample(all_images, sample_count)
        else:
            # 使用原有目录采样方式
            sampled_images = get_test_images_by_percentage(
                dataset_key, 
                args.test_percentage,
                seed=cfg.get('seed', 42),
                use_true_random=test_data_true_random
            )
        
        # 组织成字典格式（类别名已经在sample_test_images中标准化）
        for img_path, class_name in sampled_images:
            test_samples[class_name].append(img_path)
        
        avg_per_class = len(sampled_images) / len(test_samples) if test_samples else 0
        print(f"✓ 采样完成: {len(sampled_images)} 张图像，覆盖 {len(test_samples)} 个类别")
        print(f"  平均每类: {avg_per_class:.1f} 张")
        print("="*70)
    
    # 否则使用discovery集
    else:
        if args.test_data_dir is None:
            raise ValueError("请提供测试数据目录 --test_data_dir 或使用 --use_test_data")
        
        print("="*70)
        print(colored("📊 测试数据来源: images_discovery (发现集)", "cyan", attrs=['bold']))
        print("="*70)
        print(f"测试数据路径: {args.test_data_dir}")
        print(f"说明: 使用discovery集进行评估（每类固定样本数）")
        print("="*70)
        
        # 检查是JSON文件还是目录
        if args.test_data_dir.endswith('.json'):
            # 从JSON文件加载
            print("检测到JSON文件，使用JSON格式加载测试数据")
            test_samples = load_test_data_from_json(args.test_data_dir, cfg.get('dataset_name'))
        else:
            # 从目录加载（原有逻辑）
            print("检测到目录，使用目录格式加载测试数据")
            # 获取数据集名称用于类别名标准化
            dataset_key = get_dataset_key_for_test(cfg)
            from data.class_name_mapper import (
                get_dataset_name_from_key,
                standardize_test_class_name
            )
            dataset_name = get_dataset_name_from_key(dataset_key)
            
            for raw_class_name in os.listdir(args.test_data_dir):
                class_dir = os.path.join(args.test_data_dir, raw_class_name)
                if os.path.isdir(class_dir):
                    # 标准化类别名称
                    if dataset_name:
                        class_name = standardize_test_class_name(raw_class_name, dataset_name)
                    else:
                        class_name = raw_class_name
                    
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            img_path = os.path.join(class_dir, img_name)
                            test_samples[class_name].append(img_path)
            
            total_images = sum(len(paths) for paths in test_samples.values())
            print(f"✓ 从目录加载 {len(test_samples)} 个类别，共 {total_images} 张图像")
    
    return dict(test_samples)


def cint2cname(label: int, cname_sheet: list):
    """将类别整数索引转换为类别名称"""
    return cname_sheet[label]


def extract_superidentify(cfg, individual_results):
    """从个体识别结果中提取超类识别结果"""
    words = []  # 初始化单词列表
    for v in individual_results.values():  # 遍历所有个体识别结果
        this_word = v.split(' ')[-1]  # 取最后一个单词作为类别标识
        words.append(this_word.lower())  # 转换为小写并添加到列表
    word_counts = Counter(words)  # 统计每个单词的出现次数
    # print(f"extract_superidentify 中每个单词出现次数: {word_counts}")
    if cfg['dataset_name'] == 'pet':  # 如果是宠物数据集
        return [super_name for super_name, _ in word_counts.most_common(2)]  # 返回出现次数最多的2个超类
    else:  # 其他数据集
        return [super_name for super_name, _ in word_counts.most_common(1)]  # 返回出现次数最多的1个超类



def extract_python_list(text):
    """从文本中提取Python列表格式的内容"""
    pattern = r"\[(.*?)\]"  # 定义匹配方括号内容的正则表达式
    matches = re.findall(pattern, text)  # 查找所有匹配的内容
    return matches  # 返回匹配结果列表


def trim_result2json(raw_reply: str):
    """
    the raw_answer is a dirty output from LLM following our template.
    this function helps to extract the target JSON content contained in the
    output.
    """
    # 从LLM的原始输出中提取JSON格式的内容
    if raw_reply.find("Output JSON:") >= 0:  # 如果包含"Output JSON:"标记
        answer = raw_reply.split("Output JSON:")[1].strip()  # 提取标记后的内容
    else:  # 否则直接使用原始内容
        answer = raw_reply.strip()  # 去除首尾空白字符

    if not answer.startswith('{'): answer = '{' + answer  # 如果开头不是{，则添加

    if not answer.endswith('}'): answer = answer + '}'  # 如果结尾不是}，则添加

    # json_answer = json.loads(answer)  # 注释掉的JSON解析代码
    return answer  # 返回处理后的JSON字符串


def clean_name(name: str):
    """清理类别名称，统一格式"""
    name = name.title() 
    name = name.replace("-", " ")  
    name = name.replace("'s", "") 
    return name  


def extract_names(gussed_names, clean=True):
    """从猜测的名称列表中提取和清理名称"""
    gussed_names = [name.strip() for name in gussed_names]
    if clean:  # 如果需要清理
        gussed_names = [clean_name(name) for name in gussed_names]  
    gussed_names = list(set(gussed_names))  # 去重并转换为列表
    return gussed_names  # 返回处理后的名称列表


def how_to_distinguish(bot, prompt):
    """询问LLM如何区分不同类别"""
    reply = bot.infer(prompt, temperature=0.1) 
    used_tokens = bot.get_used_tokens()  
    print(f"llm used_tokens: {used_tokens},")
    print(20*"=")  
    print(reply)  #
    print(20*"=") 

    return reply  


def load_train_samples(cfg, kshot=None):
    """加载K-shot训练样本，返回 {category: [image_paths]}。
    优先从 cfg['path_train_samples'] (JSON) 读取；否则从 cfg['train_root'] 目录扫描。
    """
    samples = {}
    if 'path_train_samples' in cfg and os.path.exists(cfg['path_train_samples']):
        try:
            samples = load_json(cfg['path_train_samples'])
        except Exception as e:
            print(f"failed to load path_train_samples: {cfg['path_train_samples']}, err={e}")
            samples = {}
    elif 'train_root' in cfg and os.path.isdir(cfg['train_root']):
        train_root = cfg['train_root']
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for cname in sorted(os.listdir(train_root)):
            cdir = os.path.join(train_root, cname)
            if not os.path.isdir(cdir):
                continue
            imgs = []
            for fname in sorted(os.listdir(cdir)):
                fpath = os.path.join(cdir, fname)
                ext = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and ext in valid_exts:
                    imgs.append(fpath)
            if imgs:
                samples[cname] = imgs
    else:
        raise FileNotFoundError("Neither cfg['path_train_samples'] nor cfg['train_root'] is valid.")

    if kshot is not None:
        trimmed = {}
        for cat, paths in samples.items():
            trimmed[cat] = paths[:kshot]
        return trimmed
    return samples


def build_gallery(cfg, mllm_bot, captioner, retrieval, kshot=5,region_num=3, superclass=None, data_discovery=None):
    """构建多模态类别模板库并保存到JSON(向量转list)。"""

    # 读取训练样本
    k = kshot if kshot is not None else int(str(cfg.get('k_shot', '3')))
    # train_samples = load_train_samples(cfg, kshot=k)
    train_samples = defaultdict(list)
    for name, path in data_discovery.subcat_to_sample.items():
        train_samples[name].append(path)
    print(f"loaded train samples for {len(train_samples)} classes, kshot={k}")
    print(f"train_samples: {train_samples}") 

    # 构建模板库
    gallery = retrieval.build_template_gallery(mllm_bot, train_samples, captioner, superclass, kshot, region_num)
    
    return gallery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Discovery', formatter_class=argparse.ArgumentDefaultsHelpFormatter) 

    parser.add_argument('--mode',  
                        type=str, 
                        default='build_knowledge_base', 
                        choices=['build_gallery', 'build_knowledge_base', 'classify', 'evaluate', 'fastonly', 'slowonly', 'fast_slow'],  # 可选值列表
                        help='operating mode for each stage')  
    parser.add_argument('--config_file_env',  
                        type=str,  
                        default='./configs/env_machine.yml',  # 默认配置文件路径
                        help='location of host environment related config file')  
    parser.add_argument('--config_file_expt',  # 添加实验配置文件参数
                        type=str,  
                        default='./configs/expts/bird200_all.yml', 
                        help='location of host experiment related config file') 
    # arguments for control experiments - 控制实验的参数
    parser.add_argument('--num_per_category',  # 添加每个类别的样本数量参数
                        type=str, 
                        default='3',  
                        choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'random'], 
                        )
    # build_gallery 相关
    parser.add_argument('--kshot', type=int, default=None, help='shots per class when building gallery (override cfg)')
    parser.add_argument('--region_num', type=int, default=None, help='region selelct per class when building gallery (override cfg)')
    parser.add_argument('--superclass', type=str, default=None, help='superclass for CDV prompts (override cfg)')
    parser.add_argument('--gallery_out', type=str, default=None, help='path to save built gallery json')
    parser.add_argument('--fusion_method', type=str, default='concat', help='fusion method')
    
    # 快慢思考系统相关参数
    parser.add_argument('--knowledge_base_dir', type=str, default='./knowledge_base', help='knowledge base directory')
    parser.add_argument('--query_image', type=str, default=None, help='query image path for classification')
    parser.add_argument('--test_data_dir', type=str, default=None, help='test data directory for evaluation')
    parser.add_argument('--use_test_data', action='store_true', help='use images_test directory for testing')
    parser.add_argument('--test_percentage', type=float, default=100.0, help='percentage of test images to use (0-100)')
    parser.add_argument('--results_out', type=str, default='./results.json', help='output path for results')
    parser.add_argument('--use_slow_thinking', type=bool, default=None, help='force use slow thinking (None for auto)')
    parser.add_argument('--confidence_threshold', type=float, default=0.8, help='confidence threshold for fast thinking')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='similarity threshold for trigger mechanism')

    args = parser.parse_args()    
    cfg = setup_config(args.config_file_env, args.config_file_expt)  
    
    # 设置当前数据集
    dataset_name = cfg.get('dataset_name', 'dog')  # 从配置文件获取数据集名称
    set_current_dataset(dataset_name)
    
    # 检查并自动生成JSON文件（如果需要）
    check_and_generate_json_files(dataset_name)
    
    # drop the seed - 设置随机种子
    seed_everything(cfg['seed']) 

    expt_id_suffix = f"_{args.num_per_category}"  # 创建实验ID后缀

    cuda_ids = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    print(f"当前使用的 GPU 为：{cuda_ids}")

    start_time = time.time()
    print()
    print(colored(f"=== Experiment Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} ===", "cyan"))
    print()

    # 打印命令行参数
    print(colored("=== Experiment Arguments ===", "green"))
    pprint.pprint(vars(args))

    # 打印配置文件参数
    print(colored("=== Configuration (cfg) ===", "green"))
    pprint.pprint(cfg)

    # 全局系统实例管理 - 避免重复加载模型
    system = None
    

    if args.mode == 'build_knowledge_base':
        """
        构建快慢思考系统的知识库
        CUDA_VISIBLE_DEVICES=3 python discovering.py --mode=build_knowledge_base --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --num_per_category=10 --knowledge_base_dir=/data/yjx/MLLM/Try_again/experiments/dog120/knowledge_base 2>&1 | tee ./logs/build_knowledge_base_dog120.log
            
        CUDA_VISIBLE_DEVICES=1 python discovering.py --mode=build_knowledge_base --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --num_per_category=1 --knowledge_base_dir=/data/yjx/MLLM/Try_again/experiments/dog120/knowledge_base 2>&1 | tee ./logs_opti/build_knowledge_base_dog120_experience.log
        """
        # 初始化快慢思考系统
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            cfg=cfg,
            dataset_info=CURRENT_DATASET
        )
            
        # 加载训练样本
        data_discovery = get_or_create_discovery_set(cfg, folder_suffix=expt_id_suffix, args=args)
        train_samples = defaultdict(list)
        # {"Chihuaha": "./datasets/dogs_120/images_discovery_all_3/000.Chihuaha_000000.jpg", "Poodle": "./datasets/dogs_120/images_discovery_all_3/001.Poodle_000000.jpg", ...}
        for name, path in data_discovery.subcat_to_sample.items():
            for p in path:
                train_samples[name].append(p)
        print(f"构建知识库，包含 {len(train_samples)} 个类别, dog datasets:{len(DATA_STATS[cfg['dataset_name']]['class_names'])}")
            
        # 构建知识库
        system.load_knowledge_base(args.knowledge_base_dir) # 方便构建stats
        image_kb, text_kb = system.build_knowledge_base(
            train_samples, 
            save_dir=args.knowledge_base_dir,
            augmentation=True
        )
            
        print(f"知识库构建完成，保存到: {args.knowledge_base_dir}")
        
    elif args.mode == 'classify':
        """
        使用快慢思考系统进行单张图像分类
        CUDA_VISIBLE_DEVICES=1 python discovering.py --mode=classify --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --query_image=./test_image.jpg --knowledge_base_dir=/data/yjx/MLLM/Try/experiments/dog120/knowledge_base 2>&1 | tee ././logs/testfast.log
        """
        if args.query_image is None:
            raise ValueError("请提供查询图像路径 --query_image")
        
        # 初始化快慢思考系统
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            device_id=cfg.get('device_id', 0),
            cfg=cfg,
            dataset_info=CURRENT_DATASET
        )
        
        # 加载知识库
        system.load_knowledge_base(args.knowledge_base_dir)
        
        # 分类图像
        result = system.classify_single_image(
            args.query_image,
            use_slow_thinking=args.use_slow_thinking
        )
        
        # 保存结果
        system.save_results([result], args.results_out)
        
        print(f"分类结果: {result['final_prediction']} (置信度: {result['final_confidence']:.4f})")
        print(f"使用慢思考: {result.get('used_slow_thinking', False)}")
        print(f"结果已保存到: {args.results_out}")

    elif args.mode == 'evaluate':
        """
        在测试数据集上评估快慢思考系统
        CUDA_VISIBLE_DEVICES=1 python discovering.py --mode=evaluate --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --test_data_dir=./test_data --knowledge_base_dir=./knowledge_base_dog120 --results_out=./evaluation_results.json
        或使用测试集:
        CUDA_VISIBLE_DEVICES=1 python discovering.py --mode=evaluate --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --use_test_data --test_percentage=50 --knowledge_base_dir=./knowledge_base_dog120 --results_out=./evaluation_results.json
        """
        # 初始化快慢思考系统
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            device_id=cfg.get('device_id', 0),
            cfg=cfg,
            dataset_info=CURRENT_DATASET
        )
        
        # 加载知识库
        system.load_knowledge_base(args.knowledge_base_dir)
        
        # 准备测试样本
        test_samples = prepare_test_samples(cfg, args)
        
        print(f"测试数据集包含 {len(test_samples)} 个类别")
        
        # 评估系统
        evaluation_result = system.evaluate_on_dataset(
            test_samples,
            use_slow_thinking=args.use_slow_thinking
        )
        
        # 保存评估结果
        system.save_results([evaluation_result], args.results_out)
        
        print(f"评估完成，准确率: {evaluation_result['accuracy']:.4f}")
        print(f"快思考比例: {evaluation_result['fast_thinking_ratio']:.4f}")
        print(f"慢思考比例: {evaluation_result['slow_thinking_ratio']:.4f}")
        print(f"结果已保存到: {args.results_out}")
    
    elif args.mode == 'fastonly':
        """
        CUDA_VISIBLE_DEVICES=2 python discovering.py --mode=fastonly --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --test_data_dir=/data/yjx/MLLM/UniFGVR/datasets/dogs_120/images_discovery_all_10 --knowledge_base_dir=/data/yjx/MLLM/Try_again/experiments/dog120/knowledge_base --results_out=./logs/fastonly_eval.json 2>&1 | tee ./logs/fastonly_eval_lcb.log
        或使用测试集:
        CUDA_VISIBLE_DEVICES=2 python discovering.py --mode=fastonly --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --use_test_data --test_percentage=50 --knowledge_base_dir=./knowledge_base --results_out=./results.json
        """

        # 初始化系统（仅用于加载组件），随后只用fast模块
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            cfg=cfg,
            dataset_info=CURRENT_DATASET
        )
        # 加载知识库
        system.load_knowledge_base(args.knowledge_base_dir)

        # 准备测试样本
        test_samples = prepare_test_samples(cfg, args)

        print(f'test sample keys: {list(test_samples.keys())[:5]}...')
        print(f"[fastonly] 测试数据集包含 {len(test_samples)} 个类别")
        # 仅使用快思考评估
        fast_module = system.fast_thinking
        correct = 0
        total = 0
        correct_slow_true = 0    # 正确且需要 slow thinking
        correct_slow_false = 0   # 正确但不需要 slow thinking 预期
        error_slow_true = 0      # 错误且需要 slow thinking 预期
        error_slow_false = 0     # 错误但不需要 slow thinking
        for true_cat, paths in test_samples.items():
            for path in paths:
                try:
                    fast_res = fast_module.fast_thinking_pipeline(path, top_k=5)
                    # 使用融合Top-1作为fast-only预测，兼容旧逻辑兜底
                    pred = fast_res.get('predicted_fast') or fast_res.get('fused_top1') or fast_res.get('predicted_category') or fast_res.get('img_category', 'unknown')
                    ok = is_similar(pred, true_cat, threshold=0.5)
                    if ok:
                        print(f"succ. pred cate:{pred}, true cate:{true_cat}, need_slow_thinking:{fast_res['need_slow_thinking']}")
                        if fast_res['need_slow_thinking']:
                            correct_slow_true+=1
                        else:
                            correct_slow_false+=1
                        correct += 1
                    else:
                        print(f"failed. pred cate:{pred}, true cate:{true_cat}, need_slow_thinking:{fast_res['need_slow_thinking']}")
                        if fast_res['need_slow_thinking']:
                            error_slow_true += 1
                        else:
                            error_slow_false += 1
                    total += 1

                except Exception as e:
                    print(f'Exception:{e}')
                    total += 1


        acc = correct / total if total > 0 else 0.0
        print(f"✅ 正确预测总数: {correct}")
        print(f"  - 其中需要 slow thinking: {correct_slow_true}")
        print(f"  - 其中不需要 slow thinking: {correct_slow_false}")

        print(f"❌ 错误预测总数: {total - correct}")
        print(f"  - 其中需要 slow thinking: {error_slow_true}")
        print(f"  - 其中不需要 slow thinking: {error_slow_false}")
        print(f"[fastonly] 准确率: {acc:.4f} ({correct}/{total})")
    elif args.mode == 'slowonly':
        """
        CUDA_VISIBLE_DEVICES=3 python discovering.py --mode=slowonly --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --test_data_dir=/data/yjx/MLLM/UniFGVR/datasets/dogs_120/images_discovery_all_10 --knowledge_base_dir=/data/yjx/MLLM/Try/experiments/dog120/knowledge_base --results_out=./logs/slowonly_eval.json 2>&1 | tee ./logs/slowonly_eval.log
        或使用测试集:
        CUDA_VISIBLE_DEVICES=3 python discovering.py --mode=slowonly --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --use_test_data --test_percentage=50 --knowledge_base_dir=./knowledge_base --results_out=./results.json
        """

        # 初始化系统（仅用于加载组件），随后只用slow模块
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            cfg=cfg,
            dataset_info=CURRENT_DATASET
        )
        # 加载知识库
        system.load_knowledge_base(args.knowledge_base_dir)

        # 准备测试样本
        test_samples = prepare_test_samples(cfg, args)

        print(f'test sample keys: {list(test_samples.keys())[:5]}...')
        print(f"[slowonly] 测试数据集包含 {len(test_samples)} 个类别")
        
        # 仅使用慢思考评估
        slow_module = system.slow_thinking
        fast_module = system.fast_thinking  # 慢思考需要快思考结果作为输入
        correct = 0
        total = 0
        
        for true_cat, paths in test_samples.items():
            for path in paths:
                try:
                    # 先执行快思考获取结果（慢思考需要这个输入）
                    fast_res = fast_module.fast_thinking_pipeline(path, top_k=5)
                    
                    # 执行慢思考
                    slow_res = slow_module.slow_thinking_pipeline(path, fast_res, top_k=5)
                     
                    # 使用慢思考的最终预测
                    pred = slow_res.get('predicted_category', 'unknown')
                    ok = is_similar(pred, true_cat, threshold=0.5)
                    
                    if ok:
                        print(f"succ. pred cate:{pred}, true cate:{true_cat}, confidence:{slow_res.get('confidence', 0):.4f}")
                        correct += 1
                    else:
                        print(f"failed. pred cate:{pred}, true cate:{true_cat}, confidence:{slow_res.get('confidence', 0):.4f}")
                    
                    total += 1

                except Exception as e:
                    print(f'Exception:{e}')
                    total += 1

        acc = correct / total if total > 0 else 0.0
        print(f"✅ 正确预测总数: {correct}")
        print(f"❌ 错误预测总数: {total - correct}")
        print(f"[slowonly] 准确率: {acc:.4f} ({correct}/{total})")
        
    elif args.mode == 'fast_slow':
        """
        CUDA_VISIBLE_DEVICES=0 python discovering.py --mode=fast_slow --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --test_data_dir=/data/yjx/MLLM/UniFGVR/datasets/dogs_120/images_discovery_all_10 --knowledge_base_dir=/data/yjx/MLLM/Try_again/experiments/dog120/knowledge_base --results_out=./logs/fast_and_slow_eval.json 2>&1 | tee ./logs/fast_and_slow_update_lcb_10_context256.log
        或使用测试集:
        CUDA_VISIBLE_DEVICES=0 python discovering.py --mode=fast_slow --config_file_env=./configs/env_machine.yml --config_file_expt=./configs/expts/dog120_all.yml --use_test_data --test_percentage=50 --knowledge_base_dir=./knowledge_base --results_out=./results.json
        """

        # 初始化完整的快慢思考系统
        system = FastSlowThinkingSystem(
            model_tag=cfg['model_size_mllm'],
            model_name=cfg['model_size_mllm'],
            device='cuda' if cfg['host'] in ["xiao"] else 'cpu',
            cfg=cfg,
            dataset_info=CURRENT_DATASET
        )
        # 加载知识库
        system.load_knowledge_base(args.knowledge_base_dir)
        system.load_experience_base(args.knowledge_base_dir)
        
        # 准备测试样本
        test_samples = prepare_test_samples(cfg, args)

        print(f'test sample keys: {list(test_samples.keys())[:5]}...')  # 只显示前5个类别
        print(f"[fast and slow] 测试数据集包含 {len(test_samples)} 个类别")
        
        # 使用完整的快慢思考系统评估
        correct = 0
        total = 0
        fast_only_correct = 0    # 仅快思考正确的数量
        slow_triggered = 0       # 触发慢思考的数量
        slow_triggered_correct = 0  # 触发慢思考且正确的数量
        
        # 导入结果保存模块
        from data.result_saver import (
            save_classification_result,
            create_result_entry,
            get_experiment_dir_from_dataset_info
        )
        
        # 准备结果列表用于保存
        classification_results = []
        
        # for true_cat, paths in test_samples.items():
        from datetime import datetime
        from tqdm import tqdm

        pbar = tqdm(test_samples.items())
        for true_cat, paths in pbar:
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            pbar.set_description(f"[{now_str}] Processing fast-slow")
            for path in paths:
                # 使用完整的快慢思考系统分类
                result = system.classify_single_image(path, use_slow_thinking=None, top_k=10)
                
                pred = result.get('final_prediction', 'unknown')
                ok = is_similar(pred, true_cat, threshold=0.3)
                used_slow = result.get('used_slow_thinking', False)
                
                # 提取快思考和慢思考结果
                fast_result = result.get('fast_result', {})
                fast_result_data = {
                    'predicted_category': fast_result.get('predicted_category', 'unknown'),
                    'predicted_fast': fast_result.get('predicted_fast', 'unknown'),
                    'confidence': fast_result.get('confidence', 0.0),
                    'fused_top1': fast_result.get('fused_top1', 'unknown'),
                    'fused_top1_prob': fast_result.get('fused_top1_prob', 0.0),
                    'need_slow_thinking': fast_result.get('need_slow_thinking', False),
                    'img_category': fast_result.get('img_category', 'unknown'),
                    'text_category': fast_result.get('text_category', 'unknown')
                }
                
                slow_result_data = {}
                if used_slow:
                    slow_result = result.get('slow_result', {})
                    slow_result_data = {
                        'predicted_category': slow_result.get('predicted_category', 'unknown'),
                        'confidence': slow_result.get('confidence', 0.0),
                        'reasoning': slow_result.get('reasoning', '')
                    }
                
                # 获取项目根目录用于路径转换
                # 项目根目录就是discovering.py所在目录，这是最可靠的方式
                # 使用os.path.dirname(os.path.abspath(__file__))确保获取绝对路径
                project_root = os.path.dirname(os.path.abspath(__file__))
                
                # 创建结果条目（6元组）
                result_entry = create_result_entry(
                    label=true_cat,                    # 1. 正确标签
                    prediction=pred,                   # 2. 预测结果
                    is_correct=ok,                     # 3. 是否正确
                    fast_result=fast_result_data,      # 4. 快思考分类结果
                    slow_result=slow_result_data if used_slow else None,  # 5. 慢思考分类结果
                    image_path=path,                   # 6. 测试图片路径（将转换为相对路径）
                    confidence=result.get('final_confidence', 0.0),
                    project_root=project_root
                )
                classification_results.append(result_entry)
                
                if ok:
                    print(f"succ. pred cate:{pred}, true cate:{true_cat}, used_slow:{used_slow}, confidence:{result.get('final_confidence', 0):.4f}")
                    correct += 1
                    if not used_slow:
                        fast_only_correct += 1
                    if used_slow:
                        slow_triggered_correct += 1
                else:
                    print(f"failed. pred cate:{pred}, true cate:{true_cat}, used_slow:{used_slow}, confidence:{result.get('final_confidence', 0):.4f}")
                    # if used_slow:
                    #     slow_triggered_correct += 1  # 即使错误也统计
                
                if used_slow:
                    slow_triggered += 1
                
                total += 1

        acc = correct / total if total > 0 else 0.0
        fast_only_acc = fast_only_correct / (total-slow_triggered) if (total-slow_triggered) > 0 else 0.0
        slow_trigger_ratio = slow_triggered / total if total > 0 else 0.0
        slow_trigger_acc = slow_triggered_correct / slow_triggered if slow_triggered > 0 else 0.0
        
        print(f"✅ 正确预测总数: {correct}")
        print(f"  - 其中仅快思考正确: {fast_only_correct}")
        print(f"  - 其中慢思考触发且正确: {slow_triggered_correct}")
        print(f"❌ 错误预测总数: {total - correct}")
        print(f"📊 慢思考触发数量: {slow_triggered}")
        print(f"[fast and slow] 总体准确率: {acc:.4f} ({correct}/{total})")
        print(f"[fast and slow] 快思考准确率: {fast_only_acc:.4f}")
        print(f"[fast and slow] 慢思考触发比例: {slow_trigger_ratio:.4f}")
        print(f"[fast and slow] 慢思考准确率: {slow_trigger_acc:.4f}")
        
        # 保存分类结果
        try:
            dataset_key = get_dataset_key_for_test(cfg)
            experiment_dir = get_experiment_dir_from_dataset_info(CURRENT_DATASET)
            if experiment_dir:
                metadata = {
                    'accuracy': acc,
                    'correct': correct,
                    'total': total,
                    'fast_only_correct': fast_only_correct,
                    'slow_triggered': slow_triggered,
                    'slow_triggered_correct': slow_triggered_correct,
                    'fast_only_acc': fast_only_acc,
                    'slow_trigger_ratio': slow_trigger_ratio,
                    'slow_trigger_acc': slow_trigger_acc
                }
                save_path = save_classification_result(
                    dataset_name=dataset_key,
                    experiment_dir=experiment_dir,
                    results=classification_results,
                    metadata=metadata
                )
                print(f"📁 分类结果已保存到: {save_path}")
            else:
                print("⚠️  无法获取实验目录，跳过结果保存")
        except Exception as e:
            print(f"⚠️  保存分类结果时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        raise NotImplementedError 



    end_time = time.time()
    print()
    print(colored(f"=== Experiment End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))} ===", "cyan"))
    
    elapsed = end_time - start_time

    days = int(elapsed // 86400)
    hours = int((elapsed % 86400) // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    if days > 0:
        time_str = f"{days}天 {hours}小时 {minutes}分 {seconds:.2f}秒"
    else:
        time_str = f"{hours}小时 {minutes}分 {seconds:.2f}秒"

    print(colored(f"=== Total Runtime: {time_str} ===", "cyan"))
    print()
