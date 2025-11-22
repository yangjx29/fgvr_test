"""
知识库构建模块
用于构建快慢思考细粒度图像识别的知识检索库

包含功能：
1. 构建图像知识库：包含原始图像、增强图像等
2. 构建文本知识库：包含类别描述、属性描述等
3. 支持多种数据增强策略
4. 提供检索接口
"""

import os
import json
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import re
from collections import defaultdict

from agents.mllm_bot import MLLMBot
from retrieval.multimodal_retrieval import MultimodalRetrieval
from utils.fileios import dump_json, load_json, dump_json_override


class KnowledgeBaseBuilder:
    """知识库构建器"""
    
    def __init__(self, 
                 image_encoder_name="./models/Clip/clip-vit-base-patch32",
                 text_encoder_name="./models/Clip/clip-vit-base-patch32", 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 cfg=None,
                 dataset_info=None):
        """
        初始化知识库构建器
        
        Args:
            image_encoder_name: 图像编码器名称
            text_encoder_name: 文本编码器名称  
            device: 设备
            cfg: 配置参数
        """

        from datetime import datetime
        init_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n================= 知识库构建器初始化 =================")
        print(f"🕒 初始化时间: {init_time}")
        print(f"🖥️ 设备: {device}")
        print(f"🖼️ 图像编码器: {image_encoder_name}")
        print(f"📝 文本编码器: {text_encoder_name}")

        self.device = device
        self.cfg = cfg
        self.dataset_info = dataset_info or {}

        if self.dataset_info:
            print(f"📊 数据集信息 keys: {list(self.dataset_info.keys())}")
        else:
            print("📊 数据集信息: 空")
        
        # 初始化检索模块
        print("🔍 初始化 MultimodalRetrieval 模块...")
        self.retrieval = MultimodalRetrieval(
            image_encoder_name=image_encoder_name,
            text_encoder_name=text_encoder_name,
            fusion_method='weighted',
            device=device
        )
        print("✓ 检索模块初始化完成")
        
        # 知识库存储
        self.image_knowledge_base = {}  # {category: [image_features]}
        self.text_knowledge_base = {}   # {category: [text_features]}
        self.category_descriptions = {} # {category: description}
        print("📚 知识库初始化完成")
        print("====================================================\n")
        
    def augment_image(self, image_path: str, augmentation_type: str = "all") -> List[str]:
        """
        对图像进行数据增强
        
        Args:
            image_path: 原始图像路径
            augmentation_type: 增强类型 ("all", "rotation", "brightness", "contrast", "blur", "flip")
            
        Returns:
            List[str]: 增强后图像路径列表
        """
        image = Image.open(image_path).convert("RGB")
        augmented_paths = []
        
        # 保存原始图像
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        base_dir = os.path.dirname(image_path)
        
        if augmentation_type in ["all", "rotation"]:
            # 旋转增强
            for angle in [90, 180, 270]:
                rotated = image.rotate(angle, expand=True)
                rotated_path = os.path.join(base_dir, f"{base_name}_rot_{angle}.jpg")
                rotated.save(rotated_path)
                augmented_paths.append(rotated_path)
        
        if augmentation_type in ["all", "brightness"]:
            # 亮度增强
            for factor in [0.7, 1.3]:
                enhancer = ImageEnhance.Brightness(image)
                bright = enhancer.enhance(factor)
                bright_path = os.path.join(base_dir, f"{base_name}_bright_{factor}.jpg")
                bright.save(bright_path)
                augmented_paths.append(bright_path)
        
        if augmentation_type in ["all", "contrast"]:
            # 对比度增强
            for factor in [0.8, 1.2]:
                enhancer = ImageEnhance.Contrast(image)
                contrast = enhancer.enhance(factor)
                contrast_path = os.path.join(base_dir, f"{base_name}_contrast_{factor}.jpg")
                contrast.save(contrast_path)
                augmented_paths.append(contrast_path)
        
        if augmentation_type in ["all", "blur"]:
            # 模糊增强
            for radius in [1, 2]:
                blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
                blur_path = os.path.join(base_dir, f"{base_name}_blur_{radius}.jpg")
                blurred.save(blur_path)
                augmented_paths.append(blur_path)
        
        if augmentation_type in ["all", "flip"]:
            # 翻转增强
            flipped_h = image.transpose(Image.FLIP_LEFT_RIGHT)
            flip_h_path = os.path.join(base_dir, f"{base_name}_flip_h.jpg")
            flipped_h.save(flip_h_path)
            augmented_paths.append(flip_h_path)
            
            flipped_v = image.transpose(Image.FLIP_TOP_BOTTOM)
            flip_v_path = os.path.join(base_dir, f"{base_name}_flip_v.jpg")
            flipped_v.save(flip_v_path)
            augmented_paths.append(flip_v_path)
        
        return augmented_paths
    
    def get_wiki_description(self, category_name: str) -> str:
        """
        从Wikipedia获取类别描述
        
        Args:
            category_name: 类别名称
            
        Returns:
            str: Wikipedia描述
        """
        try:
            # 构建Wikipedia URL
            url = f"https://en.wikipedia.org/wiki/{category_name.replace('_', ' ').replace('-', ' ')}"
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 提取第一段描述
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 100 and category_name.lower() in text.lower():
                    return text
            
            return f"Information about {category_name} from Wikipedia."
        except:
            return f"Description for {category_name} category."
    
    def generate_category_description(self, mllm_bot: MLLMBot, category_name: str, 
                                   sample_images: List[str] = None) -> str:
        """
        使用MLLM生成类别描述
        
        Args:
            mllm_bot: MLLM模型
            category_name: 类别名称
            sample_images: 样本图像路径列表
            
        Returns:
            str: 生成的类别描述
        """
        # 首先尝试从Wikipedia获取
        wiki_desc = self.get_wiki_description(category_name)
        print(f'wiki desc: {wiki_desc}')
        if sample_images and len(sample_images) > 0:
            # 使用样本图像生成更准确的描述
            prompt = f"""Based on the provided images of {category_name}, generate a comprehensive and discriminative description that captures the key visual characteristics that distinguish this category from other similar categories. 
            
            Focus on:
            1. Distinctive physical features
            2. Color patterns and markings
            3. Size and proportions
            4. Behavioral characteristics (if applicable)
            5. Unique identifying traits
            
            The description should be concise but informative, suitable for fine-grained visual recognition.
            """
            
            # 加载图像
            images = [Image.open(img_path).convert("RGB") for img_path in sample_images[:3]]  # 最多使用3张图像
            
            try:
                reply, description = mllm_bot.describe_attribute(images, prompt)
                if isinstance(description, list):
                    description = " ".join(description)
                print(f'MLLM description: {description}')
                
                # 清理图像以释放内存
                del images
                import gc
                gc.collect()
                
                return description
            except Exception as e:
                print(f"MLLM描述生成失败: {e}")
                # 清理图像
                del images
                import gc
                gc.collect()
                return wiki_desc
        else:
            return wiki_desc
    
    def build_image_knowledge_base(self, train_samples: Dict[str, List[str]], 
                                 augmentation: bool = True) -> Dict[str, np.ndarray]:
        """
        构建图像知识库
        
        Args:
            train_samples: {category: [image_paths]} 训练样本
            augmentation: 是否进行数据增强
            
        Returns:
            Dict[str, np.ndarray]: {category: image_features}
        """
        print("构建图像知识库...")
        image_kb = {}
        
        for category, image_paths in tqdm(train_samples.items()):
            print(f"处理类别: {category}")
            category_features = []
            
            for img_path in image_paths:
                # 提取原始图像特征
                try:
                    feat = self.retrieval.extract_image_feat(img_path)
                    category_features.append(feat)
                except Exception as e:
                    print(f"提取图像特征失败 {img_path}: {e}")
                    continue
                
                # 数据增强
                # if augmentation:
                #     try:
                #         augmented_paths = self.augment_image(img_path, "all")
                #         for aug_path in augmented_paths:
                #             try:
                #                 aug_feat = self.retrieval.extract_image_feat(aug_path)
                #                 category_features.append(aug_feat)
                #             except Exception as e:
                #                 print(f"提取增强图像特征失败 {aug_path}: {e}")
                #                 continue
                #             # 清理临时文件
                #             if os.path.exists(aug_path):
                #                 os.remove(aug_path)
                #     except Exception as e:
                #         print(f"图像增强失败 {img_path}: {e}")
                #         continue
            
            if category_features:
                # 计算类别平均特征
                category_features = np.array(category_features)
                avg_feature = np.mean(category_features, axis=0)
                image_kb[category] = avg_feature
                print(f"类别 {category} 图像特征维度: {avg_feature.shape}")
            else:
                print(f"警告: 类别 {category} 没有有效的图像特征")
        
        self.image_knowledge_base = image_kb
        return image_kb
    
    def build_text_knowledge_base(self, mllm_bot: MLLMBot, train_samples: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        构建文本知识库
        
        Args:
            mllm_bot: MLLM模型
            train_samples: {category: [image_paths]} 训练样本
            
        Returns:
            Dict[str, np.ndarray]: {category: text_features}
        """
        print("构建文本知识库...")
        text_kb = {}
        
        for category, image_paths in tqdm(train_samples.items()):
            print(f"处理类别: {category}")
            
            # 生成类别描述
            description = self.generate_category_description(mllm_bot, category, image_paths)
            self.category_descriptions[category] = description
            
            # 提取文本特征
            try:
                text_feat = self.retrieval.extract_text_feat(description)
                text_kb[category] = text_feat
                print(f"类别 {category} 文本特征维度: {text_feat.shape}")
                print(f"描述: {description[:100]}...")
            except Exception as e:
                print(f"提取文本特征失败 {category}: {e}")
                continue
        
        self.text_knowledge_base = text_kb
        return text_kb
    
    def build_knowledge_base(self, mllm_bot: MLLMBot, train_samples: Dict[str, List[str]], 
                           augmentation: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        构建完整知识库
        
        Args:
            mllm_bot: MLLM模型
            train_samples: 训练样本
            augmentation: 是否进行数据增强
            
        Returns:
            Tuple: (image_kb, text_kb)
        """
        print("开始构建知识库...")
        
        # 构建图像知识库
        image_kb = self.build_image_knowledge_base(train_samples, augmentation)
        
        # 构建文本知识库
        text_kb = self.build_text_knowledge_base(mllm_bot, train_samples)
        
        print("知识库构建完成!")
        print(f"图像知识库包含 {len(image_kb)} 个类别")
        print(f"文本知识库包含 {len(text_kb)} 个类别")
        
        return image_kb, text_kb
    
    def initialize_lcb_stats_with_labels(self, train_samples: Dict[str, List[str]], 
                                       fast_thinking_module, top_k: int = 5) -> Dict:
        """
        使用有标签的训练集初始化LCB统计参数
        
        Args:
            train_samples: {category: [image_paths]} 训练样本
            fast_thinking_module
            top_k: 检索top-k结果
            
        Returns:
            Dict: 初始化统计结果
        """
        stats_summary = {
            "total_samples": 0,
            "correct_predictions": 0,
            "category_stats": {},
            "initialization_accuracy": 0.0
        }
        
        # 重置统计量
        fast_thinking_module.category_stats = defaultdict(lambda: {"n": 0, "m": 0})
        fast_thinking_module.total_predictions = 0
        
        correct_count = 0
        total_count = 0
        
        print(f'test train_samples:{train_samples}')
        for category, image_paths in train_samples.items():
            # print(f"处理类别: {category} ({len(image_paths)} 张图像)")
            category_correct = 0
            category_total = 0
            # print(f'image_paths:{image_paths}')
            for img_path in image_paths:
                # 执行快思考流程
                # print(f'path test: {img_path}')
                result = fast_thinking_module.fast_thinking_pipeline(img_path, top_k)
                
                predicted_category = result.get("fused_top1", "unknown")
                
                # 判断预测是否正确
                from utils.util import is_similar
                is_correct = is_similar(predicted_category, category, threshold=0.5)
                
                # 更新统计量
                fast_thinking_module.update_stats(category, is_correct)
                
                if is_correct:
                    correct_count += 1
                    category_correct += 1
                
                total_count += 1
                category_total += 1
            
            # 记录类别统计
            stats_summary["category_stats"][category] = {
                "total_samples": category_total,
                "correct_predictions": category_correct,
                "accuracy": category_correct / category_total if category_total > 0 else 0.0
            }
            
            print(f"类别 {category}: {category_correct}/{category_total} 正确")
        
        # 计算整体统计
        stats_summary["total_samples"] = total_count
        stats_summary["correct_predictions"] = correct_count
        stats_summary["initialization_accuracy"] = correct_count / total_count if total_count > 0 else 0.0
        
        # 保存统计量
        fast_thinking_module.save_stats()
        
        print(f"LCB统计参数初始化完成!")
        print(f"总样本数: {total_count}")
        print(f"正确预测: {correct_count}")
        print(f"初始化准确率: {stats_summary['initialization_accuracy']:.4f}")
        print(f"总预测次数: {fast_thinking_module.total_predictions}")
        
        return stats_summary
    
    def save_knowledge_base(self, save_dir: str):
        """
        保存知识库到文件
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存图像知识库
        image_kb_path = os.path.join(save_dir, "image_knowledge_base.json")
        image_kb_to_save = {cat: feat.tolist() for cat, feat in self.image_knowledge_base.items()}
        dump_json(image_kb_path, image_kb_to_save)
        
        # 保存文本知识库
        text_kb_path = os.path.join(save_dir, "text_knowledge_base.json")
        text_kb_to_save = {cat: feat.tolist() for cat, feat in self.text_knowledge_base.items()}
        dump_json(text_kb_path, text_kb_to_save)
        
        # 保存类别描述
        desc_path = os.path.join(save_dir, "category_descriptions.json")
        dump_json(desc_path, self.category_descriptions)
        
        print(f"知识库已保存到: {save_dir}")
    
    def update_knowledge_base(self, save_dir: str):
        """
        保存知识库到文件
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存图像知识库
        image_kb_path = os.path.join(save_dir, "image_knowledge_base.json")
        image_kb_to_save = {cat: feat.tolist() for cat, feat in self.image_knowledge_base.items()}
        dump_json_override(image_kb_path, image_kb_to_save)
        
        # 保存文本知识库
        text_kb_path = os.path.join(save_dir, "text_knowledge_base.json")
        text_kb_to_save = {cat: feat.tolist() for cat, feat in self.text_knowledge_base.items()}
        dump_json_override(text_kb_path, text_kb_to_save)
        
        # 保存类别描述
        desc_path = os.path.join(save_dir, "category_descriptions.json")
        dump_json_override(desc_path, self.category_descriptions)

        print(f"知识库已更新: {save_dir}")
    
    def load_knowledge_base(self, load_dir: str):
        """
        从文件加载知识库
        
        Args:
            load_dir: 加载目录
        """
        # 加载图像知识库
        image_kb_path = os.path.join(load_dir, "image_knowledge_base.json")
        if os.path.exists(image_kb_path):
            image_kb_data = load_json(image_kb_path)
            # 处理两种格式：字典格式或列表格式（旧版本）
            if isinstance(image_kb_data, dict):
                self.image_knowledge_base = {cat: np.array(feat) for cat, feat in image_kb_data.items()}
            elif isinstance(image_kb_data, list) and len(image_kb_data) > 0:
                # 旧格式：列表的第一个元素是字典
                if isinstance(image_kb_data[0], dict):
                    self.image_knowledge_base = {cat: np.array(feat) for cat, feat in image_kb_data[0].items()}
                else:
                    print(f"警告: 图像知识库格式不正确，跳过加载")
                    self.image_knowledge_base = {}
            else:
                print(f"警告: 图像知识库为空或格式不正确")
                self.image_knowledge_base = {}
        
        # 加载文本知识库
        text_kb_path = os.path.join(load_dir, "text_knowledge_base.json")
        if os.path.exists(text_kb_path):
            text_kb_data = load_json(text_kb_path)
            # 处理两种格式：字典格式或列表格式（旧版本）
            if isinstance(text_kb_data, dict):
                self.text_knowledge_base = {cat: np.array(feat) for cat, feat in text_kb_data.items()}
            elif isinstance(text_kb_data, list) and len(text_kb_data) > 0:
                # 旧格式：列表的第一个元素是字典
                if isinstance(text_kb_data[0], dict):
                    self.text_knowledge_base = {cat: np.array(feat) for cat, feat in text_kb_data[0].items()}
                else:
                    print(f"警告: 文本知识库格式不正确，跳过加载")
                    self.text_knowledge_base = {}
            else:
                print(f"警告: 文本知识库为空或格式不正确")
                self.text_knowledge_base = {}
        
        # 加载类别描述
        desc_path = os.path.join(load_dir, "category_descriptions.json")
        if os.path.exists(desc_path):
            desc_data = load_json(desc_path)
            # 处理两种格式：字典格式或列表格式
            if isinstance(desc_data, dict):
                self.category_descriptions = desc_data
            elif isinstance(desc_data, list) and len(desc_data) > 0:
                # 列表格式：取第一个元素（应该是字典）
                if isinstance(desc_data[0], dict):
                    self.category_descriptions = desc_data[0]
                else:
                    print(f"警告: 类别描述格式不正确，跳过加载")
                    self.category_descriptions = {}
            else:
                print(f"警告: 类别描述为空或格式不正确")
                self.category_descriptions = {}
        
        # 加载self-belief
        belief_path = os.path.join(load_dir, "self_belief.txt")
        if os.path.exists(belief_path):
            with open(belief_path, 'r', encoding='utf-8') as f:
                self.self_belief = f.read()
            print(f"Self-Belief已从 {belief_path} 加载")
        print(f"知识库已从 {load_dir} 加载")
    
    def image_retrieval(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        图像检索
        
        Args:
            query_image_path: 查询图像路径
            top_k: 返回top-k结果
            
        Returns:
            List[Tuple[str, float]]: [(category, similarity_score), ...]
        """
        if not self.image_knowledge_base:
            raise ValueError("图像知识库为空，请先构建知识库")
        
        # 提取查询图像特征
        query_feat = self.retrieval.extract_image_feat(query_image_path)
        
        # 计算相似度
        similarities = []
        for category, feat in self.image_knowledge_base.items():
            sim = np.dot(query_feat, feat)  # 余弦相似度
            similarities.append((category, sim))
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def text_retrieval(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        文本检索
        
        Args:
            query_text: 查询文本
            top_k: 返回top-k结果
            
        Returns:
            List[Tuple[str, float]]: [(category, similarity_score), ...]
        """
        if not self.text_knowledge_base:
            raise ValueError("文本知识库为空，请先构建知识库")
        
        # 提取查询文本特征
        query_feat = self.retrieval.extract_text_feat(query_text)
        
        # 计算相似度
        similarities = []
        for category, feat in self.text_knowledge_base.items():
            sim = np.dot(query_feat, feat)  # 余弦相似度
            similarities.append((category, sim))
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def incremental_update(self, category: str, image_path: str, structured_description: str,key_regions: List[Dict], confidence: float, mllm_bot, save_dir: str,
                         signals: Dict = None) -> bool:
        """
        增量更新知识库：严格的质量控制 + 平滑更新机制
        
        Args:
            category: 预测类别
            image_path: 图像路径
            structured_description: 结构化描述
            key_regions: 关键区域信息
            confidence: 预测置信度
            mllm_bot: MLLM模型
            save_dir: 保存目录
            signals: 辅助信号（LCB、一致性等）
            
        Returns:
            bool: 是否成功更新
        """
        signals = signals or {}
        
        # === 严格的质量判断机制 ===
        quality_score = self._evaluate_update_quality(category, confidence, signals)
        
        if quality_score < 0.7:  # 质量阈值
            print(f"跳过更新：质量分数过低 {quality_score:.3f}")
            return False
        
        print(f"质量分数: {quality_score:.3f}，开始增量更新")
        # === 图像知识库更新 ===
        self._update_image_knowledge_base(category, image_path, quality_score)
        
        # === 文本知识库更新 ===
        # self._update_text_knowledge_base(category, structured_description, mllm_bot, quality_score)
        
        # === 保存更新后的知识库 ===
        self.update_knowledge_base(save_dir)
        
        print(f"知识库增量更新完成: {category}")
        return True

    
    def _evaluate_update_quality(self, category: str, confidence: float, signals: Dict) -> float:
        """
        评估更新质量：综合多个指标
        
        Args:
            category: 预测类别
            confidence: 预测置信度
            signals: 辅助信号
            
        Returns:
            float: 质量分数 [0, 1]
        """
        quality_factors = []
        
        # 1. 置信度因子 (权重: 0.3)
        conf_factor = min(1.0, confidence / 0.8)  # 0.8为高置信度基准
        quality_factors.append(("confidence", conf_factor, 0.3))
        
        # 2. LCB因子 (权重: 0.25)
        lcb_value = signals.get('lcb', 0.5)
        lcb_factor = min(1.0, lcb_value / 0.7)  # 0.7为LCB高值基准
        quality_factors.append(("lcb", lcb_factor, 0.25))
        
        # 3. 快慢思考一致性因子 (权重: 0.2)
        fast_slow_consistent = signals.get('fast_slow_consistent', False)
        consistency_factor = 1.0 if fast_slow_consistent else 0.3
        quality_factors.append(("consistency", consistency_factor, 0.2))
        
        # 4. 融合Top-1概率因子 (权重: 0.15)
        fused_prob = signals.get('fused_top1_prob', 0.5)
        fused_factor = min(1.0, fused_prob / 0.7)
        quality_factors.append(("fused_prob", fused_factor, 0.15))
        
        # 5. 排名因子 (权重: 0.1)
        rank_fast = signals.get('rank_in_fast_topk', 10)  # 默认排名靠后
        rank_enh = signals.get('rank_in_enhanced_topk', 10)
        rank_factor = max(0.0, 1.0 - (min(rank_fast, rank_enh) / 5.0))  # 排名越靠前越好
        quality_factors.append(("ranking", rank_factor, 0.1))
        
        # 计算加权质量分数
        total_score = sum(factor * weight for _, factor, weight in quality_factors)
        
        print(f"质量评估详情: {[(name, f'{factor:.3f}', f'{weight:.2f}') for name, factor, weight in quality_factors]}")
        
        return total_score
    
    def _update_image_knowledge_base(self, category: str, image_path: str, quality_score: float):
        """
        平滑更新图像知识库
        
        Args:
            category: 类别
            image_path: 图像路径
            quality_score: 质量分数
        """
        # 提取新图像特征
        new_feat = self.retrieval.extract_image_feat(image_path)
        
        if category in self.image_knowledge_base:
            # 平滑更新：新特征权重基于质量分数
            old_feat = self.image_knowledge_base[category]
            
            # 动态权重：质量越高，新特征权重越大，但不超过0.1
            new_weight = min(0.1, quality_score * 0.04)
            old_weight = 1.0 - new_weight
            
            # 加权平均更新
            updated_feat = old_weight * old_feat + new_weight * new_feat
            self.image_knowledge_base[category] = updated_feat
            
            print(f"图像知识库平滑更新: {category}, 新权重={new_weight:.3f}")
        else:
            # 新类别：直接添加
            # self.image_knowledge_base[category] = new_feat
            # print(f"图像知识库新增类别: {category}")
            print(f'图像知识库跳过新增类别: {category}')
    
    def _update_text_knowledge_base(self, category: str, structured_description: str, 
                                  mllm_bot, quality_score: float):
        """
        平滑更新文本知识库
        
        Args:
            category: 类别
            structured_description: 结构化描述
            mllm_bot: MLLM模型
            quality_score: 质量分数
        """
        try:
            # 生成新的文本描述（基于结构化描述）
            if structured_description and len(structured_description.strip()) > 10:
                # 使用结构化描述生成更精确的类别描述
                prompt = f"""Based on the detailed structured description of this {category} image, generate a concise but discriminative text description that captures the key visual characteristics for fine-grained recognition.

                Structured description: {structured_description}

                Focus on distinctive features that help distinguish this category from similar ones."""
                
                try:
                    # 这里需要根据实际的MLLM接口调整
                    new_description = structured_description  # 暂时使用结构化描述
                except:
                    new_description = f"Updated description for {category}: {structured_description[:100]}..."
            else:
                new_description = f"Updated description for {category} category."
            
            # 提取新文本特征
            new_text_feat = self.retrieval.extract_text_feat(new_description)
            
            if category in self.text_knowledge_base:
                # 平滑更新文本特征
                old_text_feat = self.text_knowledge_base[category]
                
                # 文本特征更新权重更保守
                new_weight = min(0.2, quality_score * 0.3)
                old_weight = 1.0 - new_weight
                
                updated_text_feat = old_weight * old_text_feat + new_weight * new_text_feat
                self.text_knowledge_base[category] = updated_text_feat
                
                # 更新描述
                if category in self.category_descriptions:
                    old_desc = self.category_descriptions[category]
                    self.category_descriptions[category] = f"{old_desc}\n\nUpdated: {new_description}"
                else:
                    self.category_descriptions[category] = new_description
                
                print(f"文本知识库平滑更新: {category}, 新权重={new_weight:.3f}")
            else:
                # 新类别
                self.text_knowledge_base[category] = new_text_feat
                self.category_descriptions[category] = new_description
                print(f"文本知识库新增类别: {category}")
                
        except Exception as e:
            print(f"文本知识库更新失败: {e}")
            raise


# 示例使用
if __name__ == "__main__":
    # 初始化
    builder = KnowledgeBaseBuilder()
    mllm_bot = MLLMBot(model_tag="Qwen2.5-VL-7B", model_name="Qwen2.5-VL-7B", device="cuda")
    
    # 示例训练样本
    train_samples = {
        "Chihuahua": ["path/to/chihuahua1.jpg", "path/to/chihuahua2.jpg"],
        "Shiba Inu": ["path/to/shiba1.jpg", "path/to/shiba2.jpg"]
    }
    
    # 构建知识库
    image_kb, text_kb = builder.build_knowledge_base(mllm_bot, train_samples)
    
    # 保存知识库
    builder.save_knowledge_base("./knowledge_base")
    
    print("知识库构建完成!")
