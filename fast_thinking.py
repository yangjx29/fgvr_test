"""
快思考模块
实现基于CLIP的双模态检索和触发器机制

功能：
1. 图像-图像检索
2. 图像-文本检索  
3. 触发器机制判断是否需要慢思考
4. 结果融合和归一化
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from collections import Counter
import json

from knowledge_base_builder import KnowledgeBaseBuilder
from utils.util import is_similar


class FastThinking:
    """快思考模块"""
    
    def __init__(self, knowledge_base_builder: KnowledgeBaseBuilder, 
                 confidence_threshold: float = 0.8,
                 similarity_threshold: float = 0.7,
                 # 这两个值调过了，应该是最优区间
                 fusion_weight: float = 0.05,
                 softmax_temp: float = 0.07,
                 fused_conf_threshold: float = 0.6,
                 fused_margin_threshold: float = 0.12,
                 per_modality_conf_threshold: float = 0.5,
                 consider_topk_overlap: bool = True,
                 topk_for_overlap: int = 3):
        """
        初始化快思考模块
        
        Args:
            knowledge_base_builder: 知识库构建器
            confidence_threshold: 置信度阈值，超过此值直接返回结果
            similarity_threshold: 相似度阈值，用于判断两个模态结果是否一致
        """
        self.kb_builder = knowledge_base_builder
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.fusion_weight = fusion_weight
        # 温度越小，分布越尖锐；对Top-1更友好
        self.softmax_temp = max(1e-6, softmax_temp)
        # 触发器增强参数
        self.fused_conf_threshold = fused_conf_threshold
        self.fused_margin_threshold = fused_margin_threshold
        self.per_modality_conf_threshold = per_modality_conf_threshold
        self.consider_topk_overlap = consider_topk_overlap
        self.topk_for_overlap = max(1, topk_for_overlap)
        
    def image_to_image_retrieval(self, query_image_path: str, top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        图像到图像检索
        
        Args:
            query_image_path: 查询图像路径
            top_k: 返回top-k结果
            
        Returns:
            Tuple[str, float, List[Tuple[str, float]]]: (最佳匹配类别, 最高相似度, 所有结果)
        """
        try:
            results = self.kb_builder.image_retrieval(query_image_path, top_k)
            if not results:
                return "unknown", 0.0, []
            
            best_category, best_score = results[0]
            return best_category, best_score, results
        except Exception as e:
            print(f"图像检索失败: {e}")
            return "unknown", 0.0, []
    
    def image_to_text_retrieval(self, query_image_path: str, top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        图像到文本检索
        通过图像生成描述，然后检索文本知识库
        
        Args:
            query_image_path: 查询图像路径
            top_k: 返回top-k结果
            
        Returns:
            Tuple[str, float, List[Tuple[str, float]]]: (最佳匹配类别, 最高相似度, 所有结果)
        """
        try:
            # 使用CLIP提取图像特征，然后与文本知识库比较
            query_img_feat = self.kb_builder.retrieval.extract_image_feat(query_image_path)
            
            # 计算与文本知识库的相似度
            similarities = []
            for category, text_feat in self.kb_builder.text_knowledge_base.items():
                # 使用图像特征与文本特征计算相似度
                sim = np.dot(query_img_feat, text_feat)
                similarities.append((category, sim))
            
            # 排序并返回top-k
            similarities.sort(key=lambda x: x[1], reverse=True)
            results = similarities[:top_k]
            
            if not results:
                return "unknown", 0.0, []
            
            best_category, best_score = results[0]
            return best_category, best_score, results
        except Exception as e:
            print(f"图像到文本检索失败: {e}")
            return "unknown", 0.0, []
    
    def normalize_scores(self, img_results: List[Tuple[str, float]], 
                        text_results: List[Tuple[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        归一化两个模态的相似度分数
        
        Args:
            img_results: 图像检索结果
            text_results: 文本检索结果
            
        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: (归一化图像分数, 归一化文本分数)
        """
        # 提取分数
        img_scores = [score for _, score in img_results]
        text_scores = [score for _, score in text_results]
        
        # 归一化到[0, 1]
        if img_scores:
            img_min, img_max = min(img_scores), max(img_scores)
            if img_max > img_min:
                img_scores = [(score - img_min) / (img_max - img_min) for score in img_scores]
            else:
                img_scores = [1.0] * len(img_scores)
        else:
            img_scores = []
        
        if text_scores:
            text_min, text_max = min(text_scores), max(text_scores)
            if text_max > text_min:
                text_scores = [(score - text_min) / (text_max - text_min) for score in text_scores]
            else:
                text_scores = [1.0] * len(text_scores)
        else:
            text_scores = []
        
        # 构建类别到分数的映射
        img_scores_dict = {category: score for (category, _), score in zip(img_results, img_scores)}
        text_scores_dict = {category: score for (category, _), score in zip(text_results, text_scores)}
        
        return img_scores_dict, text_scores_dict

    def _to_probs(self, results: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        将相似度分数转换为概率（按类聚合后softmax）。
        """
        if not results:
            return {}
        # 先收集并聚合同类（取最大分数）
        class_to_score = {}
        for cname, score in results:
            if cname not in class_to_score:
                class_to_score[cname] = score
            else:
                class_to_score[cname] = max(class_to_score[cname], score)
        # 温度缩放softmax
        scores = np.array(list(class_to_score.values()), dtype=np.float32)
        scores = scores / self.softmax_temp
        # 数值稳定
        scores = scores - scores.max()
        exps = np.exp(scores)
        probs = exps / (exps.sum() + 1e-12)
        return {c: float(p) for c, p in zip(class_to_score.keys(), probs)}

    def _rrf(self, results: List[Tuple[str, float]], k: int = 60) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion，对一个排序列表生成RRF分数。
        
        k: 常数，通常取60。
        """
        rrf = {}
        for rank, (cname, _) in enumerate(results, start=1):
            rrf[cname] = rrf.get(cname, 0.0) + 1.0 / (k + rank)
        return rrf
    
    def fuse_results(self, img_results: List[Tuple[str, float]], 
                    text_results: List[Tuple[str, float]], 
                    fusion_weight: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        融合两个模态的检索结果
        
        Args:
            img_results: 图像检索结果
            text_results: 文本检索结果
            fusion_weight: 融合权重
            
        Returns:
            List[Tuple[str, float]]: 融合后的结果
        """
        # 使用概率融合 + RRF 融合，稳健性更好
        alpha = self.fusion_weight if fusion_weight is None else fusion_weight
        img_probs = self._to_probs(img_results)
        text_probs = self._to_probs(text_results)
        img_rrf = self._rrf(img_results)
        text_rrf = self._rrf(text_results)

        categories = set(img_probs.keys()) | set(text_probs.keys()) | set(img_rrf.keys()) | set(text_rrf.keys())
        fused = []
        for c in categories:
            p_img = img_probs.get(c, 0.0)
            p_txt = text_probs.get(c, 0.0)
            rrf_img = img_rrf.get(c, 0.0)
            rrf_txt = text_rrf.get(c, 0.0)
            # 概率融合（主）+ RRF 辅助
            score = alpha * p_img + (1 - alpha) * p_txt + 0.1 * (rrf_img + rrf_txt)
            fused.append((c, float(score)))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused
    
    def trigger_mechanism(self, img_category: str, text_category: str, 
                         img_confidence: float, text_confidence: float,
                         fused_top1: str, fused_top1_prob: float, fused_margin: float,
                         topk_overlap: bool, name_soft_agree: bool) -> Tuple[bool, str, float]:
        """
        触发器机制：判断是否需要进入慢思考
        
        Args:
            img_category: 图像检索结果
            text_category: 文本检索结果
            img_confidence: 图像检索置信度
            text_confidence: 文本检索置信度
            
        Returns:
            Tuple[bool, str, float]: (是否需要慢思考, 预测类别, 置信度)
        """
        # 名称软一致性（语义近似）或Top-K 重叠，弱冲突判定
        categories_match_soft = is_similar(img_category, text_category, threshold=self.similarity_threshold) or name_soft_agree

        # 若融合Top-1置信度足够高且与次高差距足够大，则直接信任融合结果
        if fused_top1_prob >= self.fused_conf_threshold and fused_margin >= self.fused_margin_threshold:
            return False, fused_top1, fused_top1_prob

        # 若两个模态较为一致（软）且各自置信度均不低，则无需慢思考
        if categories_match_soft and img_confidence >= self.per_modality_conf_threshold and text_confidence >= self.per_modality_conf_threshold:
            return False, fused_top1, float(max(img_confidence, text_confidence))

        # 若Top-K结果存在重叠且融合Top-1落在重叠集合中（通过上层传入的布尔），提高信任
        if self.consider_topk_overlap and topk_overlap and fused_top1_prob >= (self.fused_conf_threshold * 0.9):
            return False, fused_top1, fused_top1_prob

        # 其余情况进入慢思考
        avg_confidence = (img_confidence + text_confidence) / 2
        return True, "conflict", avg_confidence
    
    def fast_thinking_pipeline(self, query_image_path: str, top_k: int = 5) -> Dict:
        """
        快思考完整流程
        
        Args:
            query_image_path: 查询图像路径
            top_k: 检索top-k结果
            
        Returns:
            Dict: 包含预测结果、置信度、是否需要慢思考等信息
        """
        print(f"开始快思考流程，查询图像: {query_image_path}")
        
        # 1. 图像到图像检索
        img_category, img_confidence, img_results = self.image_to_image_retrieval(query_image_path, top_k)
        print(f"图像检索结果: {img_category} (置信度: {img_confidence:.4f})")
        
        # 2. 图像到文本检索
        text_category, text_confidence, text_results = self.image_to_text_retrieval(query_image_path, top_k)
        print(f"文本检索结果: {text_category} (置信度: {text_confidence:.4f})")
        
        # 3. 融合结果（概率+RRF）
        fused_results = self.fuse_results(img_results, text_results)
        print(f"融合结果: {fused_results[:3]}")  # 显示前3个结果
        fused_top1 = fused_results[0][0] if fused_results else img_category
        # 计算融合的softmax概率与margin
        fused_scores = np.array([s for _, s in fused_results], dtype=np.float32) if fused_results else np.array([1.0], dtype=np.float32)
        fused_scaled = fused_scores / self.softmax_temp
        fused_scaled = fused_scaled - fused_scaled.max()
        fused_exps = np.exp(fused_scaled)
        fused_probs = fused_exps / (fused_exps.sum() + 1e-12)
        fused_top1_prob = float(fused_probs[0]) if fused_probs.size > 0 else 1.0
        fused_margin = float(fused_probs[0] - fused_probs[1]) if fused_probs.size > 1 else fused_top1_prob

        # 各模态的top-k类别集合与softmax置信度
        img_topk = [c for c, _ in img_results[:self.topk_for_overlap]]
        text_topk = [c for c, _ in text_results[:self.topk_for_overlap]]
        topk_overlap = any(c in text_topk for c in img_topk)
        # 名称软一致：任一top-k之间近似
        name_soft_agree = False
        for ci in img_topk:
            for ct in text_topk:
                if is_similar(ci, ct, threshold=self.similarity_threshold):
                    name_soft_agree = True
                    break
            if name_soft_agree:
                break
        # 重新定义各自置信度为各自softmax顶一概率
        img_probs = self._to_probs(img_results)
        text_probs = self._to_probs(text_results)
        img_confidence = float(max(img_probs.values())) if img_probs else 0.0
        text_confidence = float(max(text_probs.values())) if text_probs else 0.0
        
        # 4. 触发器机制
        need_slow_thinking, predicted_category, confidence = self.trigger_mechanism(
            img_category, text_category, img_confidence, text_confidence,
            fused_top1, fused_top1_prob, fused_margin, topk_overlap, name_soft_agree
        )

        # 对于fast-only流程，返回融合Top-1作为首选预测
        predicted_fast = fused_top1
        
        result = {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "need_slow_thinking": need_slow_thinking,
            "fused_top1": fused_top1,
            "predicted_fast": predicted_fast,
            "img_category": img_category,
            "text_category": text_category,
            "img_confidence": img_confidence,
            "text_confidence": text_confidence,
            "fused_results": fused_results,
            "fused_top1_prob": fused_top1_prob,
            "fused_margin": fused_margin,
            "topk_overlap": topk_overlap,
            "name_soft_agree": name_soft_agree,
            "img_results": img_results,
            "text_results": text_results
        }
        
        print(f"快思考结果: {predicted_category} (置信度: {confidence:.4f})")
        print(f"需要慢思考: {need_slow_thinking}")
        
        return result
    
    def batch_fast_thinking(self, query_image_paths: List[str], top_k: int = 5) -> List[Dict]:
        """
        批量快思考处理
        
        Args:
            query_image_paths: 查询图像路径列表
            top_k: 检索top-k结果
            
        Returns:
            List[Dict]: 每个图像的快思考结果
        """
        results = []
        for img_path in query_image_paths:
            try:
                result = self.fast_thinking_pipeline(img_path, top_k)
                results.append(result)
            except Exception as e:
                print(f"处理图像失败 {img_path}: {e}")
                results.append({
                    "predicted_category": "error",
                    "confidence": 0.0,
                    "need_slow_thinking": True,
                    "error": str(e)
                })
        return results


# 示例使用
if __name__ == "__main__":
    # 初始化知识库构建器
    kb_builder = KnowledgeBaseBuilder()
    
    # 加载知识库
    kb_builder.load_knowledge_base("./knowledge_base")
    
    # 初始化快思考模块
    fast_thinking = FastThinking(kb_builder)
    
    # 测试单张图像
    query_image = "path/to/test_image.jpg"
    result = fast_thinking.fast_thinking_pipeline(query_image)
    
    print("快思考结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
