"""
慢思考模块
实现基于MLLM+CLIP的VQA+retrieval任务

功能：
1. 让MLLM承认能力不足，识别需要关注的细节区域
2. 提取关键区域特征
3. 生成结构化描述
4. 重新检索和最终推理
"""

import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Dict, Optional
import json
import re
from collections import Counter

from agents.mllm_bot import MLLMBot
from knowledge_base_builder import KnowledgeBaseBuilder
from fast_thinking import FastThinking
from utils.util import is_similar


class SlowThinking:
    """慢思考模块"""
    
    def __init__(self, mllm_bot: MLLMBot, knowledge_base_builder: KnowledgeBaseBuilder,
                 fast_thinking: FastThinking):
        """
        初始化慢思考模块
        
        Args:
            mllm_bot: MLLM模型
            knowledge_base_builder: 知识库构建器
            fast_thinking: 快思考模块
        """
        self.mllm_bot = mllm_bot
        self.kb_builder = knowledge_base_builder
        self.fast_thinking = fast_thinking
        
    def analyze_difficulty(self, query_image_path: str, fast_result: Dict) -> Dict:
        """
        让MLLM分析图像分类的困难点
        
        Args:
            query_image_path: 查询图像路径
            fast_result: 快思考结果
            
        Returns:
            Dict: 包含困难点分析和需要关注的区域
        """
        image = Image.open(query_image_path).convert("RGB")
        
        # 构建分析提示
        prompt = f"""You are an expert in fine-grained visual recognition. I need you to analyze why this image might be difficult to classify accurately.
        Current fast analysis results:
        - Image-based prediction: {fast_result.get('img_category', 'unknown')} (confidence: {fast_result.get('img_confidence', 0):.3f})
        - Text-based prediction: {fast_result.get('text_category', 'unknown')} (confidence: {fast_result.get('text_confidence', 0):.3f})

        Please analyze:
        1. What specific visual characteristics make this image challenging to classify?
        2. What discriminative regions should I focus on to improve classification accuracy?
        3. What additional information would help distinguish this from similar categories?

        Respond in JSON format:
        {{
            "difficulty_reasons": ["reason1", "reason2", "reason3"],
            "key_regions": ["region1", "region2", "region3"],
            "additional_info_needed": "description of what additional information would help"
        }}"""
        
        try:
            reply, response = self.mllm_bot.describe_attribute(image, prompt)
            if isinstance(response, list):
                response = " ".join(response)
            
            # 解析JSON响应
            try:
                # 提取JSON部分
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    analysis = json.loads(json_str)
                else:
                    # 如果无法解析JSON，使用默认值
                    analysis = {
                        "difficulty_reasons": ["Unable to parse response"],
                        "key_regions": ["overall appearance", "distinctive features"],
                        "additional_info_needed": "More detailed visual analysis needed"
                    }
            except json.JSONDecodeError:
                print(f"json.JSONDecodeError")
                analysis = {
                    "difficulty_reasons": ["JSON parsing failed"],
                    "key_regions": ["overall appearance", "distinctive features"],
                    "additional_info_needed": "More detailed visual analysis needed"
                }
            
            return analysis
        except Exception as e:
            print(f"困难点分析失败: {e}")
            return {
                "difficulty_reasons": [f"Analysis failed: {str(e)}"],
                "key_regions": ["overall appearance", "distinctive features"],
                "additional_info_needed": "More detailed visual analysis needed"
            }
    
    def extract_key_regions(self, query_image_path: str, key_regions: List[str]) -> List[str]:
        """
        基于MLLM识别的关键区域提取特征描述
        
        Args:
            query_image_path: 查询图像路径
            key_regions: 关键区域列表
            
        Returns:
            List[str]: 区域特征描述列表
        """
        image = Image.open(query_image_path).convert("RGB")
        region_descriptions = []
        
        for region in key_regions:
            prompt = f"""Focus on the {region} in this image and provide a detailed description of its visual characteristics that would help distinguish this object from similar categories.
            Be specific about:
            - Color, texture, and patterns
            - Shape and size relative to the object
            - Unique or distinctive features
            - Any variations or special characteristics

            Provide a concise but informative description."""
            
            try:
                reply, description = self.mllm_bot.describe_attribute(image, prompt)
                if isinstance(description, list):
                    description = " ".join(description)
                region_descriptions.append(description)
            except Exception as e:
                print(f"区域描述失败 {region}: {e}")
                region_descriptions.append(f"Unable to describe {region}")
        
        return region_descriptions
    
    def generate_structured_description(self, query_image_path: str, 
                                      region_descriptions: List[str],
                                      key_regions: List[str],
                                      difficulty_analysis: Dict) -> str:
        """
        生成结构化的图像描述
        
        Args:
            query_image_path: 查询图像路径
            region_descriptions: 区域描述列表
            key_regions: 关键区域列表
            difficulty_analysis: 困难点分析
            
        Returns:
            str: 结构化描述
        """
        image = Image.open(query_image_path).convert("RGB")
        
        # 构建区域信息
        region_info = "\n".join([f"**{region}**: {desc}" for region, desc in zip(key_regions, region_descriptions)])
        
        prompt = f"""Based on the detailed analysis of this image, generate a comprehensive and discriminative description that captures the key visual characteristics for fine-grained classification.
        Region-specific descriptions:
        {region_info}

        Difficulty analysis:
        - Reasons: {', '.join(difficulty_analysis.get('difficulty_reasons', []))}
        - Additional info needed: {difficulty_analysis.get('additional_info_needed', '')}

        Please provide a structured description that:
        1. Summarizes the overall appearance
        2. Highlights distinctive features from each key region
        3. Emphasizes characteristics that differentiate this from similar categories
        4. Is concise but informative for classification

        Format the output as a clear, structured description suitable for visual recognition."""
        
        try:
            reply, description = self.mllm_bot.describe_attribute(image, prompt)
            if isinstance(description, list):
                description = " ".join(description)
            return description
        except Exception as e:
            print(f"结构化描述生成失败: {e}")
            return f"Structured description generation failed: {str(e)}"
    
    def enhanced_retrieval(self, query_image_path: str, structured_description: str, 
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """
        使用增强的图像描述进行检索
        
        Args:
            query_image_path: 查询图像路径
            structured_description: 结构化描述
            top_k: 返回top-k结果
            
        Returns:
            List[Tuple[str, float]]: 检索结果
        """
        try:
            # 提取图像特征
            img_feat = self.kb_builder.retrieval.extract_image_feat(query_image_path)
            
            # 提取文本特征
            text_feat = self.kb_builder.retrieval.extract_text_feat(structured_description)
            
            # 融合特征
            if self.kb_builder.retrieval.fusion_method == "concat":
                query_feat = np.concatenate([img_feat, text_feat])
            elif self.kb_builder.retrieval.fusion_method == "average":
                query_feat = (img_feat + text_feat) / 2
            elif self.kb_builder.retrieval.fusion_method == "weighted":
                alpha = 0.7
                query_feat = alpha * text_feat + (1 - alpha) * img_feat
            else:
                query_feat = img_feat  # 默认使用图像特征
            
            # 归一化
            norm = np.linalg.norm(query_feat) + 1e-12
            query_feat = query_feat / norm
            
            # 与知识库比较
            similarities = []
            for category, feat in self.kb_builder.image_knowledge_base.items():
                sim = np.dot(query_feat, feat)
                similarities.append((category, sim))
            
            # 排序并返回top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            print(f"增强检索失败: {e}")
            return []
    
    def multi_modal_retrieval(self, query_image_path: str, structured_description: str, 
                             top_k: int = 5) -> List[Tuple[str, float]]:
        """
        多模态检索：结合图像-图像和图像-文本检索
        
        Args:
            query_image_path: 查询图像路径
            structured_description: 结构化描述
            top_k: 返回top-k结果
            
        Returns:
            List[Tuple[str, float]]: 检索结果
        """
        try:
            # 1. 图像-图像检索
            img_img_results = self.kb_builder.image_retrieval(query_image_path, top_k)
            
            # 2. 图像-文本检索（使用结构化描述）
            img_feat = self.kb_builder.retrieval.extract_image_feat(query_image_path)
            img_text_similarities = []
            for category, text_feat in self.kb_builder.text_knowledge_base.items():
                sim = np.dot(img_feat, text_feat)
                img_text_similarities.append((category, sim))
            img_text_similarities.sort(key=lambda x: x[1], reverse=True)
            img_text_results = img_text_similarities[:top_k]
            
            # 3. 文本-文本检索（使用结构化描述）
            text_feat = self.kb_builder.retrieval.extract_text_feat(structured_description)
            text_text_similarities = []
            for category, text_kb_feat in self.kb_builder.text_knowledge_base.items():
                sim = np.dot(text_feat, text_kb_feat)
                text_text_similarities.append((category, sim))
            text_text_similarities.sort(key=lambda x: x[1], reverse=True)
            text_text_results = text_text_similarities[:top_k]
            
            # 4. 融合三种检索结果
            all_candidates = {}
            
            # 收集所有候选类别
            for category, score in img_img_results:
                all_candidates[category] = all_candidates.get(category, []) + [score * 0.4]  # 图像-图像权重
            
            for category, score in img_text_results:
                all_candidates[category] = all_candidates.get(category, []) + [score * 0.3]  # 图像-文本权重
            
            for category, score in text_text_results:
                all_candidates[category] = all_candidates.get(category, []) + [score * 0.3]  # 文本-文本权重
            
            # 计算融合分数（取最大值）
            fused_results = []
            for category, scores in all_candidates.items():
                fused_score = max(scores)  # 使用最大值策略
                fused_results.append((category, fused_score))
            
            # 排序并返回top-k
            fused_results.sort(key=lambda x: x[1], reverse=True)
            return fused_results[:top_k]
            
        except Exception as e:
            print(f"多模态检索失败: {e}")
            return []
    
    def final_reasoning(self, query_image_path: str, enhanced_results: List[Tuple[str, float]],
                       structured_description: str) -> Tuple[str, float]:
        """
        最终推理：让MLLM基于检索结果进行最终分类
        
        Args:
            query_image_path: 查询图像路径
            enhanced_results: 增强检索结果
            structured_description: 结构化描述
            top_k: 候选数量
            
        Returns:
            Tuple[str, float]: (最终预测类别, 置信度)
        """
        image = Image.open(query_image_path).convert("RGB")
        
        # 构建候选类别列表
        candidates_text = ""
        for i, (category, score) in enumerate(enhanced_results):
            candidates_text += f"{i+1}. {category} (similarity: {score:.4f})\n"
        
        prompt = f"""You are an expert in fine-grained visual recognition. Based on the detailed visual analysis and similarity search results, determine the most accurate category for this image.

        Structured description of the image:
        {structured_description}

        Top candidate categories from similarity search:
        {candidates_text}

        Please analyze the image carefully and consider:
        1. The detailed visual characteristics described above
        2. The similarity scores to each candidate category
        3. The distinctive features that differentiate between similar categories
        4. The consistency between visual evidence and similarity rankings
        5. Any unique or unusual characteristics that might affect classification

        Pay special attention to:
        - Breed-specific features (size, shape, color patterns, facial structure)
        - Distinguishing characteristics that separate similar breeds
        - Overall confidence in the visual evidence

        Provide your final prediction in JSON format:
        {{
            "predicted_category": "exact category name",
            "confidence": 0.0-1.0,
            "reasoning": "detailed explanation of your decision process and key evidence",
            "key_features": "main visual features supporting your decision",
            "uncertainty_factors": "any factors that make this classification uncertain"
        }}"""
        
        try:
            reply, response = self.mllm_bot.describe_attribute(image, prompt)
            if isinstance(response, list):
                response = " ".join(response)
            
            # 解析JSON响应
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    predicted_category = result.get("predicted_category", "unknown")
                    confidence = float(result.get("confidence", 0.5))
                    reasoning = result.get("reasoning", "No reasoning provided")
                    key_features = result.get("key_features", "No key features identified")
                    uncertainty_factors = result.get("uncertainty_factors", "No uncertainty factors")
                    
                    # 增强推理信息
                    enhanced_reasoning = f"{reasoning} | Key Features: {key_features} | Uncertainty: {uncertainty_factors}"
                else:
                    # 如果无法解析JSON，使用最高相似度的候选
                    predicted_category = enhanced_results[0][0] if enhanced_results else "unknown"
                    confidence = enhanced_results[0][1] if enhanced_results else 0.0
                    reasoning = "JSON parsing failed, using top similarity result"
                    enhanced_reasoning = reasoning
            except (json.JSONDecodeError, ValueError):
                predicted_category = enhanced_results[0][0] if enhanced_results else "unknown"
                confidence = enhanced_results[0][1] if enhanced_results else 0.0
                reasoning = "JSON parsing failed, using top similarity result"
                enhanced_reasoning = reasoning
            
            return predicted_category, confidence, enhanced_reasoning
        except Exception as e:
            print(f"最终推理失败: {e}")
            # 返回最高相似度的候选
            if enhanced_results:
                return enhanced_results[0][0], enhanced_results[0][1], f"Reasoning failed: {str(e)}"
            else:
                return "unknown", 0.0, f"Reasoning failed: {str(e)}"

    def final_reasoning_simple(self, query_image_path: str, fast_result: Dict,
                               enhanced_results: List[Tuple[str, float]], top_k: int = 5) -> Tuple[str, float, str]:
        """
        - 将 fast_result 的预测（融合Top-1）与 enhanced_results（Top-K）一起提供给 MLLM
        - 让 MLLM 结合图片与候选分数，选择“最可能子类”；
        - 若都不合适，允许给出新的类别。

        Args:
            query_image_path: 查询图像路径
            fast_result: 快思考结果字典（需包含 fused_top1 与 fused_results 可选）
            enhanced_results: 慢思考检索得到的候选 [(cat, score), ...]
            top_k: 传入给 MLLM 的增强候选个数

        Returns:
            Tuple[str, float, str]: (最终预测类别, 置信度, 推理/信息)
        """
        image = Image.open(query_image_path).convert("RGB")

        # 收集候选：fast 融合Top-1 + 慢思考Top-K
        candidates = []
        fast_top1 = fast_result.get("fused_top1") or fast_result.get("predicted_category")
        if isinstance(fast_top1, str) and len(fast_top1) > 0:
            candidates.append((fast_top1, float(fast_result.get("fused_top1_prob", 1.0))))

        for cat, score in (enhanced_results or [])[:max(0, top_k)]:
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

        prompt = f"""You are an expert in fine-grained visual recognition. Look at the image and pick the most likely subclass in{candidate_text},
        Return ONLY a JSON object with the following fields:
        {{
        "predicted_category": "exact category name",
        "confidence": 0.0-1.0,
        "reasoning": "very brief rationale"
        }}"""

        try:
            reply, response = self.mllm_bot.describe_attribute(image, prompt)
            if isinstance(response, list):
                response = " ".join(response)

            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result = json.loads(json_str)
                    predicted_category = result.get("predicted_category", "unknown")
                    confidence = float(result.get("confidence", 0.5))
                    reasoning = result.get("reasoning", "no rationale")
                    info = f"{reasoning}"
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

            return predicted_category, confidence, info

        except Exception as e:
            print(f"final_reasoning_simple failed: {e}")
            fallback = merged[0][0] if merged else (fast_top1 or "unknown")
            conf = float(merged[0][1]) if merged else 0.0
            return fallback, conf, f"final_reasoning_simple exception: {str(e)}"
    
    def slow_thinking_pipeline(self, query_image_path: str, fast_result: Dict, 
                             top_k: int = 5) -> Dict:
        """
        慢思考完整流程
        
        Args:
            query_image_path: 查询图像路径
            fast_result: 快思考结果
            top_k: 检索top-k结果
            
        Returns:
            Dict: 包含最终预测结果和详细分析
        """
        print(f"开始慢思考流程，查询图像: {query_image_path}")
        
        # 1. 分析困难点
        print("步骤1: 调用模型去分析识别困难点...")
        difficulty_analysis = self.analyze_difficulty(query_image_path, fast_result)
        print(f"困难点分析: {difficulty_analysis}")
        
        # 2. 提取关键区域特征
        print("步骤2: 提取关键区域特征...")
        key_regions = difficulty_analysis.get("key_regions", ["overall appearance"])
        region_descriptions = self.extract_key_regions(query_image_path, key_regions)
        print(f"区域描述: {region_descriptions}")
        
        # 3. 生成结构化描述
        print("步骤3: 生成结构化描述,去除冗余信息...")
        structured_description = self.generate_structured_description(
            query_image_path, region_descriptions, key_regions, difficulty_analysis
        )
        print(f"结构化描述: {structured_description}")
        
        # 4. 多模态检索
        print("步骤4: 执行多模态检索...")
        enhanced_results = self.multi_modal_retrieval(query_image_path, structured_description, top_k)
        print(f"多模态检索结果: {enhanced_results}")
        
        # 如果多模态检索失败，使用增强检索作为fallback
        if not enhanced_results:
            print("多模态检索失败，使用增强检索...")
            enhanced_results = self.enhanced_retrieval(query_image_path, structured_description, top_k)
            print(f"增强检索结果: {enhanced_results}")
        
        # 5. 最终推理
        print("步骤5: 最终推理...")
        predicted_category, confidence, reasoning = self.final_reasoning(
            query_image_path, enhanced_results, structured_description
        )
        # predicted_category, confidence, reasoning = self.final_reasoning_simple(
        #     query_image_path, fast_result, enhanced_results, top_k
        # )
        print(f"最终预测: {predicted_category} (置信度: {confidence:.4f})")
        print(f"推理过程: {reasoning}")
        
        result = {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "reasoning": reasoning,
            "difficulty_analysis": difficulty_analysis,
            "key_regions": key_regions,
            "region_descriptions": region_descriptions,
            "structured_description": structured_description,
            "enhanced_results": enhanced_results,
            "fast_result": fast_result
        }
        
        return result
    
    def slow_thinking_pipeline_update(self, query_image_path: str, fast_result: Dict, top_k: int = 5, save_dir = './experiments/dog120/knowledge_base') -> Dict:
        """
        慢思考完整流程
        
        Args:
            query_image_path: 查询图像路径
            fast_result: 快思考结果
            top_k: 检索top-k结果
            
        Returns:
            Dict: 包含最终预测结果和详细分析
        """
        print(f"开始慢思考流程，查询图像: {query_image_path}")
        
        # 1. 分析困难点
        print("步骤1: 调用模型去分析识别困难点...")
        difficulty_analysis = self.analyze_difficulty(query_image_path, fast_result)
        print(f"困难点分析: {difficulty_analysis}")
        
        # 2. 提取关键区域特征
        print("步骤2: 提取关键区域特征...")
        key_regions = difficulty_analysis.get("key_regions", ["overall appearance"])
        region_descriptions = self.extract_key_regions(query_image_path, key_regions)
        print(f"区域描述: {region_descriptions}")
        
        # 3. 生成结构化描述
        print("步骤3: 生成结构化描述,去除冗余信息...")
        structured_description = self.generate_structured_description(
            query_image_path, region_descriptions, key_regions, difficulty_analysis
        )
        print(f"结构化描述: {structured_description}")
        
        # 4. 多模态检索
        print("步骤4: 执行多模态检索...")
        enhanced_results = self.multi_modal_retrieval(query_image_path, structured_description, top_k)
        print(f"多模态检索结果: {enhanced_results}")
        
        # 如果多模态检索失败，使用增强检索作为fallback
        if not enhanced_results:
            print("多模态检索失败，使用增强检索...")
            enhanced_results = self.enhanced_retrieval(query_image_path, structured_description, top_k)
            print(f"增强检索结果: {enhanced_results}")
        
        # 5. 最终推理
        print("步骤5: 最终推理...")
        predicted_category, confidence, reasoning = self.final_reasoning(
            query_image_path, enhanced_results, structured_description
        )
        # predicted_category, confidence, reasoning = self.final_reasoning_simple(
        #     query_image_path, fast_result, enhanced_results, top_k
        # )
        print(f"最终预测: {predicted_category} (置信度: {confidence:.4f})")
        print(f"推理过程: {reasoning}")

        # 6. 统计更新 + 持续学习：基于最终预测进行自适应更新 上面代码没变
        # 更严格的置信标准：结合LCB、一致性、置信度等多重指标
        enhanced_top1_score = float((enhanced_results or [("", 0.0)])[0][1]) if enhanced_results else 0.0
        fused_top1_prob = float(fast_result.get("fused_top1_prob", 0.0))
        fused_margin = float(fast_result.get("fused_margin", 0.0))
        fused_top1 = str(fast_result.get("fused_top1", "unknown"))
        fast_slow_consistent = is_similar(fused_top1, predicted_category, threshold=0.5)
        
        # 获取LCB值
        lcb_map = fast_result.get('lcb_map', {}) or {}
        lcb_value = float(lcb_map.get(predicted_category, 0.5)) if isinstance(lcb_map, dict) else 0.5
        
        # 多重置信度检查
        confidence_checks = [
            float(confidence) >= 0.80,  # 慢思考高置信度
            enhanced_top1_score >= 0.75,  # 增强检索高分数
            (fused_top1_prob >= 0.85 and fused_margin >= 0.15),  # 融合结果高置信度+大margin
            (fast_slow_consistent and lcb_value >= 0.7),  # 一致性+高LCB
            (float(confidence) >= 0.70 and lcb_value >= 0.6)  # 中等置信度+中等LCB
        ]
        
        is_confident_for_stats = any(confidence_checks)
        
        # 更新统计：只有高置信度预测才更新m（正确次数）
        self.fast_thinking.update_stats(predicted_category, is_confident_for_stats)
        
        print(f"统计更新: 类别={predicted_category}, 置信度={confidence:.3f}, LCB={lcb_value:.3f}, " f"一致性={fast_slow_consistent}, 更新m={is_confident_for_stats}")

        # 仅当预测到具体类别名时进行更新
        if isinstance(predicted_category, str) and len(predicted_category) > 0 and predicted_category.lower() != 'unknown':
            # 组装持续学习的辅助信号（用于自适应权重）
            fast = fast_result
            fused_prob = float(fast.get('fused_top1_prob', 0.5))
            fused_margin = float(fast.get('fused_margin', 0.1))
            fast_slow_consistent = is_similar(predicted_category, fast.get('fused_top1', 'unknown'), threshold=0.5)

            # 获取 LCB 值：优先使用预测类别的LCB，否则使用融合Top-1的LCB
            lcb_map = fast.get('lcb_map', {})
            lcb_value = float(lcb_map.get(predicted_category, lcb_map.get(fused_top1, 0.5))) if isinstance(lcb_map, dict) else 0.5

            # 估计候选排名（越靠前越可信）
            rank_fast = len(fast.get('fused_results', [])) - 1  # 默认排名为最后
            rank_enh = len(enhanced_results) - 1   # 默认排名为最后
            # fused_results = fast.get('fused_results')
            # print(f'fused_results:{fused_results}')
            if isinstance(fast.get('fused_results'), list):
                for idx, (cat, _) in enumerate(fast['fused_results']):
                    if is_similar(cat, predicted_category, threshold=0.5):
                        rank_fast = idx
                        break
            if isinstance(enhanced_results, list):
                for idx, (cat, _) in enumerate(enhanced_results):
                    if is_similar(cat, predicted_category, threshold=0.5):
                        rank_enh = idx
                        break
            # print(f'候选排名: fast={rank_fast}, enhanced={rank_enh} ')
            signals = {
                'lcb': lcb_value,
                'fast_slow_consistent': fast_slow_consistent,
                'fused_top1_prob': fused_prob,
                'fused_margin': fused_margin,
                'rank_in_fast_topk': rank_fast,
                'rank_in_enhanced_topk': rank_enh
            }

            self.kb_builder.incremental_update(
                category=predicted_category,
                image_path=query_image_path,
                structured_description=structured_description,
                key_regions=key_regions,
                confidence=float(confidence),
                mllm_bot=self.mllm_bot,
                save_dir=save_dir,
                signals=signals
            )
            print(f"已对知识库进行增量更新: 类别={predicted_category}, 置信度={confidence:.4f}, signals={signals}")
        else:
            print("跳过增量更新：预测类别未知或为空")
        
        result = {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "reasoning": reasoning,
            "difficulty_analysis": difficulty_analysis,
            "key_regions": key_regions,
            "region_descriptions": region_descriptions,
            "structured_description": structured_description,
            "enhanced_results": enhanced_results,
            "fast_result": fast_result
        }
        
        return result
    
    def batch_slow_thinking(self, query_image_paths: List[str], fast_results: List[Dict], 
                           top_k: int = 5) -> List[Dict]:
        """
        批量慢思考处理
        
        Args:
            query_image_paths: 查询图像路径列表
            fast_results: 对应的快思考结果列表
            top_k: 检索top-k结果
            
        Returns:
            List[Dict]: 每个图像的慢思考结果
        """
        results = []
        for img_path, fast_result in zip(query_image_paths, fast_results):
            try:
                result = self.slow_thinking_pipeline(img_path, fast_result, top_k)
                results.append(result)
            except Exception as e:
                print(f"慢思考处理失败 {img_path}: {e}")
                results.append({
                    "predicted_category": "error",
                    "confidence": 0.0,
                    "reasoning": f"Processing failed: {str(e)}",
                    "error": str(e)
                })
        return results


# 示例使用
if __name__ == "__main__":
    # 初始化组件
    mllm_bot = MLLMBot(model_tag="Qwen2.5-VL-7B", model_name="Qwen2.5-VL-7B", device="cuda")
    kb_builder = KnowledgeBaseBuilder()
    kb_builder.load_knowledge_base("./knowledge_base")
    
    fast_thinking = FastThinking(kb_builder)
    slow_thinking = SlowThinking(mllm_bot, kb_builder, fast_thinking)
    
    # 测试慢思考
    query_image = "path/to/test_image.jpg"
    fast_result = fast_thinking.fast_thinking_pipeline(query_image)
    
    if fast_result["need_slow_thinking"]:
        slow_result = slow_thinking.slow_thinking_pipeline(query_image, fast_result)
        print("慢思考结果:")
        print(json.dumps(slow_result, indent=2, ensure_ascii=False))
    else:
        print("快思考结果足够，无需慢思考")
