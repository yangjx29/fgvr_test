"""
优化后的慢思考模块
主要优化:
1. 减少MLLM调用次数 - 合并相似步骤
2. 简化慢思考流程 - 使用更简单的推理方法
3. 添加缓存机制 - 缓存MLLM响应
4. 快速路径优化 - 在慢思考中也可以提前退出
"""

import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Dict, Optional
import json
import re
from collections import Counter
import hashlib
from functools import lru_cache

from agents.mllm_bot import MLLMBot
from knowledge_base_builder import KnowledgeBaseBuilder
from experience_base_builder import ExperienceBaseBuilder
from fast_thinking import FastThinking
from utils.util import is_similar
from data.data_stats import DOG_STATS
from difflib import get_close_matches


class SlowThinkingOptimized:
    """优化后的慢思考模块"""
    
    def __init__(self, mllm_bot: MLLMBot, knowledge_base_builder: KnowledgeBaseBuilder,
                 fast_thinking: FastThinking,
                 experience_base_builder: ExperienceBaseBuilder=None,
                 enable_cache: bool = True,
                 cache_size: int = 1000,
                 simplified_reasoning: bool = True,
                 use_experience_base: bool = True,
                 top_k_experience: int = 1):
        """
        初始化优化后的慢思考模块
        
        Args:
            mllm_bot: MLLM模型
            knowledge_base_builder: 知识库构建器
            fast_thinking: 快思考模块
            experience_base_builder: 经验库构建器
            enable_cache: 是否启用缓存
            cache_size: 缓存大小
            simplified_reasoning: 是否使用简化的推理方法
            use_experience_base: 是否使用经验库
            top_k_experience: 使用top-k个类别的经验
        """
        self.mllm_bot = mllm_bot
        self.kb_builder = knowledge_base_builder
        self.fast_thinking = fast_thinking
        self.exp_builder = experience_base_builder
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.simplified_reasoning = simplified_reasoning
        self.use_experience_base = use_experience_base
        self.top_k_experience = top_k_experience
        
        # 缓存机制
        self._mllm_cache = {}  # MLLM响应缓存
        self._description_cache = {}  # 描述缓存
        
        # 类别名称映射
        self.normalized_to_original = {
            self.normalize_name(cls): cls for cls in DOG_STATS['class_names']
        }
        self.normalized_class_names = list(self.normalized_to_original.keys())
    
    def set_experience_base(self, experience_base_builder: ExperienceBaseBuilder):
        """设置经验库构建器"""
        self.exp_builder = experience_base_builder
        print("经验库已设置到慢思考模块")
    
    def _get_self_belief_context(self) -> str:
        """
        获取当前Self-Belief策略上下文
        """
        if not self.use_experience_base or not self.exp_builder:
            return ""
        try:
            policy_text = self.exp_builder.get_self_belief()
        except AttributeError:
            return self.exp_builder.INITIAL_SELF_BELIEF
        if not policy_text or not isinstance(policy_text, str):
            return ""
        return f"Self-Belief Policy Context:\n{policy_text.strip()}"
    
    def normalize_name(self, name):
        """标准化类别名称"""
        name = name.lower()
        name = re.sub(r'[-_]', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip()
    
    def _get_cache_key(self, image_path: str, prompt: str) -> str:
        """生成缓存键"""
        if not self.enable_cache:
            return None
        cache_key = hashlib.md5(f"{image_path}_{prompt}".encode()).hexdigest()
        return cache_key
    
    def _cached_mllm_call(self, image_path: str, prompt: str, image: Image.Image = None) -> str:
        """缓存的MLLM调用"""
        cache_key = self._get_cache_key(image_path, prompt)
        if cache_key and cache_key in self._mllm_cache:
            return self._mllm_cache[cache_key]
        
        # 调用MLLM
        if image is None:
            image = Image.open(image_path).convert("RGB")
        
        reply, response = self.mllm_bot.describe_attribute(image, prompt)
        if isinstance(response, list):
            response = " ".join(response)
        
        # 缓存结果
        if cache_key:
            if len(self._mllm_cache) >= self.cache_size:
                # 删除最旧的缓存项
                oldest_key = next(iter(self._mllm_cache))
                del self._mllm_cache[oldest_key]
            self._mllm_cache[cache_key] = response
        
        return response
    
    def reasoning_with_experience_base(self, query_image_path: str, fast_result: Dict,
                                       top_k_candidates: List[str],
                                       experience_context: str = "",
                                       top_k: int = 5) -> Dict:
        """
        基于经验库的推理流程 - 使用经验库指导MLLM推理
        移除增强检索，直接使用快思考结果 + 经验库
        
        Args:
            query_image_path: 查询图像路径
            fast_result: 快思考结果
            top_k_candidates: 快思考的top-k候选类别
            experience_context: 经验库上下文
            top_k: 候选类别数量
            
        Returns:
            Dict: 推理结果
        """
        image = Image.open(query_image_path).convert("RGB")
        
        # 收集候选类别：使用快思考的top-k结果
        candidates = []
        fused_results = fast_result.get("fused_results", [])
        for cat, score in fused_results:
            candidates.append((str(cat), float(score)))
        
        # 组织候选展示文本
        candidate_text = ""
        for i, (cat, sc) in enumerate(candidates, start=1):
            candidate_text += f"{i}. {cat} (score: {sc:.4f})\n"
        
        # 构建提示词 - 简化版本，经验库作为辅助信息
        prompt = f"""Candidate classes (highly likely to contain the correct option):
        {candidate_text}\n{experience_context}"""
        # f"""You are an expert in fine-grained visual recognition. Analyze the image and determine the most likely category from the candidates:
        # {candidate_text}
        # Fast thinking results:
        # - Top prediction: {fast_result.get('fused_top1', 'unknown')} (confidence: {fast_result.get('fused_top1_prob', 0):.3f})
        # - Image prediction: {fast_result.get('img_category', 'unknown')} (confidence: {fast_result.get('img_confidence', 0):.3f})
        # - Text prediction: {fast_result.get('text_category', 'unknown')} (confidence: {fast_result.get('text_confidence', 0):.3f})
        # - Margin: {fast_result.get('fused_margin', 0):.3f}

        # """
        
        # 如果有经验库上下文，简洁地添加到提示词中
        # if experience_context:
        #     prompt += f"{experience_context}\n\n"
        
        # prompt += """Analyze the image and consider:
        # 1. Visual characteristics (size, shape, color, texture, distinctive features)
        # 2. Similarity scores from fast thinking
        # 3. Consistency between predictions
        # 4. Key distinguishing features

        # Return ONLY a JSON object:
        # {{
        #     "predicted_category": "exact category name",
        #     "confidence": 0.0-1.0,
        #     "reasoning": "brief rationale",
        #     "key_features": "main visual features"
        # }}"""
        
        # response = self._cached_mllm_call(query_image_path, prompt, image)
        # json_match = re.search(r'\{.*\}', response, re.DOTALL)
        # if json_match:
        #     json_str = json_match.group()
        #     result = json.loads(json_str)
        #     predicted_category = result.get("predicted_category", "unknown")
        #     confidence = float(result.get("confidence", 0.5))
        #     reasoning = result.get("reasoning", "no rationale")
        #     key_features = result.get("key_features", "no features")
        #     info = f"{reasoning} | Features: {key_features}"
        #     used_experience = bool(experience_context)
        # else:
        #     # 解析失败：使用快思考的Top-1
        #     fallback = candidates[0][0] if candidates else "unknown"
        #     predicted_category = fallback
        #     confidence = float(candidates[0][1]) if candidates else 0.5
        #     info = "JSON parsing failed; fallback to fast thinking"
        #     used_experience = False
        prompt += """\nPlease analyze the image step by step and provide:
            1. Your reasoning chain (CoT) following the steps above
            2. Your final prediction (only the category name)
            Format your response as:
            Reasoning: [your step-by-step reasoning]
            Prediction: [category name]"""
        reply, response = self.mllm_bot.describe_attribute(image, prompt)
        if isinstance(response, list):
            response = " ".join(response)
        
        # 解析CoT和预测
        cot = ""
        prediction = "unknown"
        
        # 提取推理链
        reasoning_match = re.search(r'Reasoning[:\s]+(.*?)(?=Prediction|$)', response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            cot = reasoning_match.group(1).strip()
        else:
            # 如果没有明确标记，尝试提取整个响应作为推理链
            cot = response
        
        # 提取预测
        prediction_match = re.search(r'Prediction[:\s]+([^\n]+)', response, re.IGNORECASE)
        if prediction_match:
            # print(f"解析成功! 预测: {prediction_match.group(1).strip()}")
            prediction = prediction_match.group(1).strip()
        else:
            # 尝试从响应末尾提取类别名称
            lines = response.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                # 检查是否是候选类别之一
                for cat in top_k_candidates:
                    if cat.lower() in last_line.lower() or last_line.lower() in cat.lower():
                        prediction = cat
                        break
                if prediction == "unknown":
                    # 使用第一个候选作为fallback
                    prediction = top_k_candidates[0] if top_k_candidates else "unknown"
        # 类别名称修正
        predicted_category = self._correct_category_name(prediction)
        
        return {
            "predicted_category": predicted_category,
            "reasoning": cot,
            "fast_result": fast_result,
            "top_k_candidates": top_k_candidates,
            "simplified": True
        }
            
    
    def simplified_reasoning_with_enhanced(self, query_image_path: str, fast_result: Dict,
                                          enhanced_results: List[Tuple[str, float]], 
                                          top_k: int = 5) -> Dict:
        """
        结合增强检索结果的简化推理流程 - 提高准确率
        """
        image = Image.open(query_image_path).convert("RGB")
        
        # 收集候选类别：fast结果 + 增强检索结果
        candidates = []
        
        # 添加fast结果的Top-K
        fused_results = fast_result.get("fused_results", [])
        for cat, score in fused_results[:top_k]:
            candidates.append((str(cat), float(score)))
        
        # 添加增强检索结果
        for cat, score in (enhanced_results or [])[:top_k]:
            candidates.append((str(cat), float(score)))
        
        # 去重并按分数降序
        dedup = {}
        for cat, sc in candidates:
            if cat not in dedup or sc > dedup[cat]:
                dedup[cat] = sc
        merged = sorted([(c, s) for c, s in dedup.items()], key=lambda x: x[1], reverse=True)
        
        # 组织候选展示文本
        candidate_text = ""
        for i, (cat, sc) in enumerate(merged[:top_k * 2], start=1):  # 显示更多候选
            candidate_text += f"{i}. {cat} (score: {sc:.4f})\n"
        
        # 改进的提示 - 提供更多上下文信息
        prompt = f"""You are an expert in fine-grained visual recognition. Look at the image and determine the most likely subclass from the following candidates:

        {candidate_text}

        Fast thinking analysis:
        - Image-based prediction: {fast_result.get('img_category', 'unknown')} (confidence: {fast_result.get('img_confidence', 0):.3f})
        - Text-based prediction: {fast_result.get('text_category', 'unknown')} (confidence: {fast_result.get('text_confidence', 0):.3f})
        - Fused prediction: {fast_result.get('fused_top1', 'unknown')} (confidence: {fast_result.get('fused_top1_prob', 0):.3f})
        - Fused margin: {fast_result.get('fused_margin', 0):.3f}

        Enhanced retrieval results:
        - Top candidate: {enhanced_results[0][0] if enhanced_results else 'unknown'} (score: {(enhanced_results[0][1] if enhanced_results else 0.0):.4f})

        Please analyze the image carefully and consider:
        1. The visual characteristics of the object (size, shape, color, texture, distinctive features)
        2. The similarity scores from both fast thinking and enhanced retrieval
        3. The consistency between different predictions
        4. Any distinctive features that help distinguish between similar categories
        5. The confidence margins between top candidates

        Pay special attention to:
        - Breed-specific features (ear shape, tail, muzzle, body proportions)
        - Color patterns and markings
        - Size and overall appearance
        - Any unique identifying traits

        Return ONLY a JSON object with the following fields:
        {{
            "predicted_category": "exact category name",
            "confidence": 0.0-1.0,
            "reasoning": "brief but detailed rationale for your decision",
            "key_features": "main visual features supporting your decision",
            "alternative_candidates": ["category1", "category2"] (if applicable)
        }}"""
        
        try:
            response = self._cached_mllm_call(query_image_path, prompt, image)
            
            # 解析JSON响应
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    predicted_category = result.get("predicted_category", "unknown")
                    confidence = float(result.get("confidence", 0.5))
                    reasoning = result.get("reasoning", "no rationale")
                    key_features = result.get("key_features", "no features")
                    alternative_candidates = result.get("alternative_candidates", [])
                    info = f"{reasoning} | Key Features: {key_features}"
                    if alternative_candidates:
                        info += f" | Alternatives: {', '.join(alternative_candidates)}"
                else:
                    # 解析失败：使用增强检索的Top-1
                    fallback = enhanced_results[0][0] if enhanced_results else (merged[0][0] if merged else "unknown")
                    predicted_category = fallback
                    confidence = float(enhanced_results[0][1]) if enhanced_results else (float(merged[0][1]) if merged else 0.5)
                    info = "JSON parsing failed; fallback to enhanced retrieval top candidate"
            except (json.JSONDecodeError, ValueError):
                # 解析失败：使用增强检索的Top-1
                fallback = enhanced_results[0][0] if enhanced_results else (merged[0][0] if merged else "unknown")
                predicted_category = fallback
                confidence = float(enhanced_results[0][1]) if enhanced_results else (float(merged[0][1]) if merged else 0.5)
                info = "JSON parsing failed; fallback to enhanced retrieval top candidate"
            
            # 类别名称修正
            predicted_category = self._correct_category_name(predicted_category)
            
            return {
                "predicted_category": predicted_category,
                "confidence": confidence,
                "reasoning": info,
                "enhanced_results": enhanced_results,
                "fast_result": fast_result,
                "simplified": True
            }
            
        except Exception as e:
            print(f"简化推理失败: {e}")
            # 回退到增强检索结果
            fallback = enhanced_results[0][0] if enhanced_results else (merged[0][0] if merged else "unknown")
            conf = float(enhanced_results[0][1]) if enhanced_results else (float(merged[0][1]) if merged else 0.0)
            return {
                "predicted_category": self._correct_category_name(fallback),
                "confidence": conf,
                "reasoning": f"Simplified reasoning failed: {str(e)}",
                "enhanced_results": enhanced_results,
                "fast_result": fast_result,
                "simplified": True
            }
    
    def simplified_reasoning_pipeline(self, query_image_path: str, fast_result: Dict, 
                                     top_k: int = 5) -> Dict:
        """
        简化的推理流程 - 减少MLLM调用次数
        只进行一次MLLM调用,直接给出最终预测
        """
        image = Image.open(query_image_path).convert("RGB")
        
        # 收集候选类别
        candidates = []
        fast_top1 = fast_result.get("fused_top1") or fast_result.get("predicted_category")
        if isinstance(fast_top1, str) and len(fast_top1) > 0:
            candidates.append((fast_top1, float(fast_result.get("fused_top1_prob", 1.0))))
        
        # 添加fast结果的Top-K
        fused_results = fast_result.get("fused_results", [])
        for cat, score in fused_results[:top_k]:
            if cat not in [c[0] for c in candidates]:
                candidates.append((str(cat), float(score)))
        
        # 去重并按分数降序
        dedup = {}
        for cat, sc in candidates:
            if cat not in dedup or sc > dedup[cat]:
                dedup[cat] = sc
        merged = sorted([(c, s) for c, s in dedup.items()], key=lambda x: x[1], reverse=True)
        
        # 组织候选展示文本
        candidate_text = ""
        for i, (cat, sc) in enumerate(merged, start=1):
            candidate_text += f"{i}. {cat} (score: {sc:.4f})\n"
        
        # 改进的提示 - 提供更多上下文信息和详细分析要求
        prompt = f"""You are an expert in fine-grained visual recognition. Look at the image and determine the most likely subclass from the following candidates:

{candidate_text}

Fast thinking analysis:
- Image-based prediction: {fast_result.get('img_category', 'unknown')} (confidence: {fast_result.get('img_confidence', 0):.3f})
- Text-based prediction: {fast_result.get('text_category', 'unknown')} (confidence: {fast_result.get('text_confidence', 0):.3f})
- Fused prediction: {fast_result.get('fused_top1', 'unknown')} (confidence: {fast_result.get('fused_top1_prob', 0):.3f})
- Fused margin: {fast_result.get('fused_margin', 0):.3f}

Please analyze the image carefully and consider:
1. The visual characteristics of the object (size, shape, color, texture, distinctive features)
2. The similarity scores from fast thinking
3. The consistency between different predictions
4. Any distinctive features that help distinguish between similar categories
5. The confidence margins between top candidates

Pay special attention to:
- Breed-specific features (ear shape, tail, muzzle, body proportions)
- Color patterns and markings
- Size and overall appearance
- Any unique identifying traits

Return ONLY a JSON object with the following fields:
{{
    "predicted_category": "exact category name",
    "confidence": 0.0-1.0,
    "reasoning": "brief but detailed rationale for your decision",
    "key_features": "main visual features supporting your decision",
    "alternative_candidates": ["category1", "category2"] (if applicable)
}}"""
        
        try:
            response = self._cached_mllm_call(query_image_path, prompt, image)
            
            # 解析JSON响应
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    predicted_category = result.get("predicted_category", "unknown")
                    confidence = float(result.get("confidence", 0.5))
                    reasoning = result.get("reasoning", "no rationale")
                    key_features = result.get("key_features", "no features")
                    alternative_candidates = result.get("alternative_candidates", [])
                    info = f"{reasoning} | Key Features: {key_features}"
                    if alternative_candidates:
                        info += f" | Alternatives: {', '.join(alternative_candidates)}"
                else:
                    # 解析失败：回退到分数最高的候选
                    fallback = merged[0][0] if merged else (fast_top1 or "unknown")
                    predicted_category = fallback
                    confidence = float(merged[0][1]) if merged else 0.5
                    info = "JSON parsing failed; fallback to top candidate"
            except (json.JSONDecodeError, ValueError):
                fallback = merged[0][0] if merged else (fast_top1 or "unknown")
                predicted_category = fallback
                confidence = float(merged[0][1]) if merged else 0.5
                info = "JSON parsing failed; fallback to top candidate"
            
            # 类别名称修正
            predicted_category = self._correct_category_name(predicted_category)
            
            return {
                "predicted_category": predicted_category,
                "confidence": confidence,
                "reasoning": info,
                "simplified": True,
                "fast_result": fast_result
            }
            
        except Exception as e:
            print(f"简化推理失败: {e}")
            # 回退到fast结果
            fallback = merged[0][0] if merged else (fast_top1 or "unknown")
            conf = float(merged[0][1]) if merged else 0.0
            return {
                "predicted_category": self._correct_category_name(fallback),
                "confidence": conf,
                "reasoning": f"Simplified reasoning failed: {str(e)}",
                "simplified": True,
                "fast_result": fast_result
            }
    
    def _correct_category_name(self, predicted_category: str) -> str:
        """修正类别名称"""
        if predicted_category not in DOG_STATS['class_names']:
            # 标准化名称
            norm_pred = self.normalize_name(predicted_category)
            
            # 精确匹配
            if norm_pred in self.normalized_class_names:
                corrected_category = self.normalized_to_original[norm_pred]
                print(f"精确标准化匹配类别修正: '{predicted_category}' -> '{corrected_category}'")
                return corrected_category
            else:
                # 模糊匹配
                close_matches = get_close_matches(norm_pred, self.normalized_class_names, n=1, cutoff=0.3)
                if close_matches:
                    best_match_norm = close_matches[0]
                    corrected_category = self.normalized_to_original[best_match_norm]
                    print(f"模糊匹配类别修正: '{predicted_category}' -> '{corrected_category}'")
                    return corrected_category
                else:
                    print(f'发现新类别:{predicted_category}')
                    return predicted_category
        return predicted_category
    
    def enhanced_retrieval_only(self, query_image_path: str, fast_result: Dict, 
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """
        改进的增强检索 - 结合多种检索策略提高准确率
        基于fast结果、图像特征和文本特征进行多模态检索
        """
        try:
            # 提取图像特征
            img_feat = self.kb_builder.retrieval.extract_image_feat(query_image_path)
            
            # 1. 图像-图像检索
            img_img_similarities = {}
            for category, kb_feat in self.kb_builder.image_knowledge_base.items():
                sim = np.dot(img_feat, kb_feat)
                img_img_similarities[category] = float(sim)
            
            # 2. 图像-文本检索
            img_text_similarities = {}
            for category, text_feat in self.kb_builder.text_knowledge_base.items():
                sim = np.dot(img_feat, text_feat)
                img_text_similarities[category] = float(sim)
            
            # 3. 使用fast结果的Top-K类别进行加权检索
            fused_results = fast_result.get("fused_results", [])
            fast_scores = {}
            for category, score in fused_results[:top_k * 3]:  # 扩大候选范围
                fast_scores[category] = float(score)
            
            # 4. 融合多种检索结果 - 改进的融合策略
            weighted_similarities = {}
            for category in set(list(img_img_similarities.keys()) + 
                               list(img_text_similarities.keys()) + 
                               list(fast_scores.keys())):
                img_img_score = img_img_similarities.get(category, 0.0)
                img_text_score = img_text_similarities.get(category, 0.0)
                fast_score = fast_scores.get(category, 0.0)
                
                # 改进的加权策略：
                # - 如果fast结果中有该类别，给予更高权重
                # - 图像-图像和图像-文本检索各占一定权重
                if category in fast_scores:
                    # fast结果中有：fast权重0.4, 图像-图像0.35, 图像-文本0.25
                    weighted_sim = (fast_score * 0.4 + 
                                   img_img_score * 0.35 + 
                                   img_text_score * 0.25)
                else:
                    # fast结果中没有：图像-图像0.55, 图像-文本0.45
                    weighted_sim = (img_img_score * 0.55 + 
                                   img_text_score * 0.45)
                
                weighted_similarities[category] = weighted_sim
            
            # 排序并返回top-k
            sorted_results = sorted(weighted_similarities.items(), key=lambda x: x[1], reverse=True)
            return sorted_results[:top_k]
            
        except Exception as e:
            print(f"增强检索失败: {e}")
            return []
    
    def slow_thinking_pipeline_optimized(self, query_image_path: str, fast_result: Dict, 
                                        top_k: int = 5, save_dir: str = None) -> Dict:
        """
        优化后的慢思考流程 - 使用经验库指导MLLM推理
        移除增强检索，直接使用快思考结果 + 经验库进行MLLM推理
        """
        print(f"开始优化后的慢思考流程，查询图像: {query_image_path}")
        
        # 步骤1: 从快思考结果中提取top-k候选类别
        print("步骤1: 提取快思考候选类别...")
        fused_results = fast_result.get("fused_results", [])
        top_k_candidates = [cat for cat, score in fused_results]
        
        print(f"快思考Top-{top_k}候选类别: {top_k_candidates}")
        
        # 步骤2: 注入Self-Belief策略上下文
        experience_context = ""
        fused_top1_prob = float(fast_result.get("fused_top1_prob", 0.0))
        fused_margin = float(fast_result.get("fused_margin", 0.0))
        
        policy_context = self._get_self_belief_context()
        
        experience_context = policy_context
        
        # 步骤3: 使用MLLM进行最终推理 - 使用经验库作为上下文
        print("步骤3: 使用MLLM进行最终推理（基于经验库）...")
        result = self.reasoning_with_experience_base(
            query_image_path=query_image_path,
            fast_result=fast_result,
            top_k_candidates=top_k_candidates,
            experience_context=experience_context,
            top_k=top_k
        )
        
        # 更新统计
        predicted_category = result.get("predicted_category", "unknown")
        confidence = result.get("confidence", 0.0)
        # self._update_stats(predicted_category, confidence, fast_result, result, save_dir)
        
        return result
    
    def _update_stats(self, predicted_category: str, confidence: float, 
                     fast_result: Dict, slow_result: Dict, save_dir: str = None):
        """更新统计量"""
        # 获取快思考结果
        fused_top1_prob = float(fast_result.get("fused_top1_prob", 0.0))
        fused_margin = float(fast_result.get("fused_margin", 0.0))
        fused_top1 = str(fast_result.get("fused_top1", "unknown"))
        fast_slow_consistent = is_similar(fused_top1, predicted_category, threshold=0.5)
        
        # 获取LCB值
        lcb_map = fast_result.get('lcb_map', {}) or {}
        lcb_value = float(lcb_map.get(predicted_category, 0.5)) if isinstance(lcb_map, dict) else 0.5
        
        # 改进的多重置信度检查 - 更严格的标准
        confidence_checks = [
            float(confidence) >= 0.85,  # 提高阈值
            (fused_top1_prob >= 0.88 and fused_margin >= 0.18),  # 提高阈值
            (fast_slow_consistent and lcb_value >= 0.75 and float(confidence) >= 0.75),  # 更严格
            (float(confidence) >= 0.75 and lcb_value >= 0.70)  # 更严格
        ]
        
        is_confident_for_stats = any(confidence_checks)
        
        # 更新统计
        self.fast_thinking.update_stats(predicted_category, is_confident_for_stats, used_slow_thinking=True)
        
        used_experience = slow_result.get("used_experience_base", False)
        print(f"统计更新: 类别={predicted_category}, 置信度={confidence:.3f}, LCB={lcb_value:.3f}, "
              f"一致性={fast_slow_consistent}, 使用经验库={used_experience}, 更新m={is_confident_for_stats}")


