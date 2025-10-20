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
from utils.fileios import dump_json, load_json


class KnowledgeBaseBuilder:
    """知识库构建器"""
    
    def __init__(self, 
                 image_encoder_name="/home/Dataset/Models/Clip/clip-vit-base-patch32",
                 text_encoder_name="/home/Dataset/Models/Clip/clip-vit-base-patch32", 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 cfg=None):
        """
        初始化知识库构建器
        
        Args:
            image_encoder_name: 图像编码器名称
            text_encoder_name: 文本编码器名称  
            device: 设备
            cfg: 配置参数
        """
        self.device = device
        self.cfg = cfg
        
        # 初始化检索模块
        self.retrieval = MultimodalRetrieval(
            image_encoder_name=image_encoder_name,
            text_encoder_name=text_encoder_name,
            fusion_method='weighted',
            device=device
        )
        
        # 知识库存储
        self.image_knowledge_base = {}  # {category: [image_features]}
        self.text_knowledge_base = {}   # {category: [text_features]}
        self.category_descriptions = {} # {category: description}
        
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
                return description
            except:
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
                if augmentation:
                    try:
                        augmented_paths = self.augment_image(img_path, "all")
                        for aug_path in augmented_paths:
                            try:
                                aug_feat = self.retrieval.extract_image_feat(aug_path)
                                category_features.append(aug_feat)
                            except Exception as e:
                                print(f"提取增强图像特征失败 {aug_path}: {e}")
                                continue
                            # 清理临时文件
                            if os.path.exists(aug_path):
                                os.remove(aug_path)
                    except Exception as e:
                        print(f"图像增强失败 {img_path}: {e}")
                        continue
            
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
            self.image_knowledge_base = {cat: np.array(feat) for cat, feat in image_kb_data[0].items()}
        
        # 加载文本知识库
        text_kb_path = os.path.join(load_dir, "text_knowledge_base.json")
        if os.path.exists(text_kb_path):
            text_kb_data = load_json(text_kb_path)
            self.text_knowledge_base = {cat: np.array(feat) for cat, feat in text_kb_data[0].items()}
        
        # 加载类别描述
        desc_path = os.path.join(load_dir, "category_descriptions.json")
        if os.path.exists(desc_path):
            self.category_descriptions = load_json(desc_path)
        
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
