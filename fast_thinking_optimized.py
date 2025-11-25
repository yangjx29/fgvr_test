"""
优化后的快思考模块
主要优化:
1. 更严格的触发机制,减少不必要的慢思考
2. 多级阈值判断,更早地拒绝慢思考
3. 动态LCB阈值调整
4. 缓存机制
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import json
import os
import hashlib
from functools import lru_cache

from knowledge_base_builder import KnowledgeBaseBuilder
from utils.util import is_similar


class FastThinkingOptimized:
    """优化后的快思考模块"""
    
    def __init__(self, knowledge_base_builder: KnowledgeBaseBuilder, 
                 confidence_threshold: float = 0.8,
                 similarity_threshold: float = 0.4,
                 # 优化的融合参数
                 fusion_weight: float = 0.05,
                 softmax_temp: float = 0.07,
                 # 优化的触发阈值 - 更激进地减少慢思考
                 fused_conf_threshold: float = 0.50, 
                 fused_margin_threshold: float = 0.08,  
                 per_modality_conf_threshold: float = 0.50,  
                 consider_topk_overlap: bool = True,
                 topk_for_overlap: int = 3,
                 # LCB 相关参数 - 优化
                 stats_file: str = "/data/yjx/MLLM/Try_again/experiments/dog120/knowledge_base/stats.json",
                 lcb_threshold: float = 0.40,  # 进一步降低到0.40，让更多样本通过
                 lcb_threshold_adaptive: bool = True,  # 启用自适应阈值
                 lcb_threshold_min: float = 0.35,  # 最小阈值
                 lcb_threshold_max: float = 0.55,  # 最大阈值
                 prior_strength: float = 3.0,  # 增加先验强度，让冷启动时LCB更高
                 prior_p: float = 0.65,  # 提高先验正确率
                 lcb_eta: float = 0.8,  # 降低置信区间系数，使LCB更宽松
                 lcb_alpha: float = 0.3,  # 降低熵的影响
                 lcb_epsilon: float = 1e-6,
                 # 缓存相关
                 enable_cache: bool = True,
                 cache_size: int = 1000):
        """
        初始化优化后的快思考模块
        """
        self.kb_builder = knowledge_base_builder
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.fusion_weight = fusion_weight
        self.softmax_temp = max(1e-6, softmax_temp)
        
        # 优化的触发阈值
        self.fused_conf_threshold = fused_conf_threshold
        self.fused_margin_threshold = fused_margin_threshold
        self.per_modality_conf_threshold = per_modality_conf_threshold
        self.consider_topk_overlap = consider_topk_overlap
        self.topk_for_overlap = max(1, topk_for_overlap)
        
        # LCB 参数
        self.lcb_threshold = lcb_threshold
        self.lcb_threshold_adaptive = lcb_threshold_adaptive
        self.lcb_threshold_min = lcb_threshold_min
        self.lcb_threshold_max = lcb_threshold_max
        self.prior_strength = prior_strength
        self.prior_p = prior_p
        self.lcb_eta = lcb_eta
        self.lcb_alpha = lcb_alpha
        self.lcb_epsilon = lcb_epsilon
        self.stats_file = stats_file
        self.total_predictions = 1
        self.category_stats = defaultdict(lambda: {"n": 0, "m": 0})
        self.load_stats()
        
        # 缓存机制
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._similarity_cache = {}  # 相似度缓存
        self._lcb_cache = {}  # LCB值缓存
        
        # 性能统计
        self.performance_stats = {
            "fast_path_count": 0,
            "slow_path_count": 0,
            "fast_path_correct": 0,
            "slow_path_correct": 0
        }
    
    def load_stats(self):
        """加载历史统计量"""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.category_stats = defaultdict(lambda: {"n": 0, "m": 0}, data.get("category_stats", {}))
                self.total_predictions = data.get("total_predictions", 0)
                # 加载性能统计
                if "performance_stats" in data:
                    self.performance_stats = data["performance_stats"]
            print(f"已加载历史统计量: {self.total_predictions} 次预测")
        else:
            print("未找到历史统计量文件，将从头开始")
    
    def save_stats(self):
        """保存统计量到文件"""
        data = {
            "category_stats": dict(self.category_stats),
            "total_predictions": self.total_predictions,
            "performance_stats": self.performance_stats
        }
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _get_cache_key(self, query_image_path: str, top_k: int) -> str:
        """生成缓存键"""
        if not self.enable_cache:
            return None
        # 使用图像路径和top_k生成缓存键
        cache_key = hashlib.md5(f"{query_image_path}_{top_k}".encode()).hexdigest()
        return cache_key
    
    def _get_similarity_cache_key(self, img_path: str, category: str) -> str:
        """生成相似度缓存键"""
        if not self.enable_cache:
            return None
        return hashlib.md5(f"{img_path}_{category}".encode()).hexdigest()
    
    def image_to_image_retrieval(self, query_image_path: str, top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
        """图像到图像检索"""
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
        """图像到文本检索"""
        query_img_feat = self.kb_builder.retrieval.extract_image_feat(query_image_path)
        similarities = []
        for category, text_feat in self.kb_builder.text_knowledge_base.items():
            sim = np.dot(query_img_feat, text_feat)
            similarities.append((category, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]
        if not results:
            return "unknown", 0.0, []
        best_category, best_score = results[0]
        return best_category, best_score, results
    
    def _to_probs(self, results: List[Tuple[str, float]]) -> Dict[str, float]:
        """将相似度分数转换为概率"""
        if not results:
            return {}
        class_to_score = {}
        for cname, score in results:
            if cname not in class_to_score:
                class_to_score[cname] = score
            else:
                class_to_score[cname] = max(class_to_score[cname], score)
        scores = np.array(list(class_to_score.values()), dtype=np.float32)
        scores = scores / self.softmax_temp
        scores = scores - scores.max()
        exps = np.exp(scores)
        probs = exps / (exps.sum() + 1e-12)
        return {c: float(p) for c, p in zip(class_to_score.keys(), probs)}
    
    def _rrf(self, results: List[Tuple[str, float]], k: int = 60) -> Dict[str, float]:
        """Reciprocal Rank Fusion"""
        rrf = {}
        for rank, (cname, _) in enumerate(results, start=1):
            rrf[cname] = rrf.get(cname, 0.0) + 1.0 / (k + rank)
        return rrf
    
    def fuse_results(self, img_results: List[Tuple[str, float]], 
                    text_results: List[Tuple[str, float]], 
                    fusion_weight: Optional[float] = None) -> List[Tuple[str, float]]:
        """融合两个模态的检索结果"""
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
            score = alpha * p_img + (1 - alpha) * p_txt + 0.1 * (rrf_img + rrf_txt)
            fused.append((c, float(score)))
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused
    
    def calculate_lcb(self, category: str, confidence_scores: List[float]) -> float:
        """计算类别的LCB"""
        import math
        stats = self.category_stats[category]
        n_raw = stats["n"]
        m_raw = stats["m"]
        
        # Beta先验平滑
        n = n_raw + self.prior_strength
        m = m_raw + self.prior_p * self.prior_strength
        p_hat = m / (n + self.lcb_epsilon)
        
        # 计算置信度分布熵
        if len(confidence_scores) > 1:
            probs = np.array(confidence_scores)
            probs = probs / (probs.sum() + 1e-12)
            entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs) + 1e-12)
        else:
            entropy = 0.0
        
        # 置信区间项
        if n_raw > 0:
            confidence_term = self.lcb_eta * math.sqrt(math.log(max(1, self.total_predictions)) / (2 * n + 1))
        else:
            confidence_term = self.lcb_eta * math.sqrt(math.log(max(1, self.total_predictions)))
        
        # LCB公式
        lcb = p_hat - confidence_term - self.lcb_alpha * entropy
        return max(0.0, min(1.0, lcb))
    
    def _get_adaptive_lcb_threshold(self) -> float:
        """
        根据历史性能自适应调整LCB阈值
        优化: 更倾向于降低阈值,减少慢思考触发率
        """
        if not self.lcb_threshold_adaptive:
            return self.lcb_threshold
        
        # 计算快速路径的正确率
        fast_path_acc = 0.0
        if self.performance_stats["fast_path_count"] > 0:
            fast_path_acc = self.performance_stats["fast_path_correct"] / self.performance_stats["fast_path_count"]
        
        # 计算慢速路径的正确率
        slow_path_acc = 0.0
        if self.performance_stats["slow_path_count"] > 0:
            slow_path_acc = self.performance_stats["slow_path_correct"] / self.performance_stats["slow_path_count"]
        
        # 计算慢思考触发率
        total_count = self.performance_stats["fast_path_count"] + self.performance_stats["slow_path_count"]
        slow_trigger_rate = 0.0
        if total_count > 0:
            slow_trigger_rate = self.performance_stats["slow_path_count"] / total_count
        
        # 优化策略: 更激进地降低阈值以减少慢思考触发率
        # 如果慢思考触发率太高(>0.85),大幅降低阈值
        if slow_trigger_rate > 0.85:
            adaptive_threshold = max(self.lcb_threshold_min, self.lcb_threshold - 0.12)
        # 如果慢思考触发率较高(>0.75),降低阈值
        elif slow_trigger_rate > 0.75:
            adaptive_threshold = max(self.lcb_threshold_min, self.lcb_threshold - 0.08)
        # 如果快速路径正确率很高(>0.85),可以稍微提高阈值
        elif fast_path_acc > 0.85:
            adaptive_threshold = min(self.lcb_threshold_max, self.lcb_threshold + 0.02)
        # 如果快速路径正确率较高(>0.75),保持阈值
        elif fast_path_acc > 0.75:
            adaptive_threshold = self.lcb_threshold
        # 如果快速路径正确率中等(0.65-0.75),稍微降低阈值
        elif fast_path_acc >= 0.65:
            adaptive_threshold = max(self.lcb_threshold_min, self.lcb_threshold - 0.03)
        # 如果快速路径正确率较低(<0.65),降低阈值
        else:
            adaptive_threshold = max(self.lcb_threshold_min, self.lcb_threshold - 0.06)
        
        return adaptive_threshold
    
    def trigger_lcb_optimized(self, img_category: str, text_category: str, 
                             img_confidence: float, text_confidence: float,
                             fused_top1: str, fused_top1_prob: float, fused_margin: float,
                             topk_overlap: bool, name_soft_agree: bool) -> Tuple[bool, str, float, Dict]:
        """
        优化后的触发器机制 - 多级阈值判断,更早地拒绝慢思考
        返回: (是否需要慢思考, 预测类别, 置信度, 触发原因)
        """
        trigger_reason = {}
        
        # === 第一级: 高置信度快速路径 (最优先判断) ===
        # 1. 融合Top-1置信度足够高且margin足够大
        if fused_top1_prob >= self.fused_conf_threshold and fused_margin >= self.fused_margin_threshold:
            trigger_reason["type"] = "high_confidence_margin"
            trigger_reason["fused_prob"] = fused_top1_prob
            trigger_reason["margin"] = fused_margin
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        
        # 3. 两个模态高度一致且各自置信度都较高
        categories_match_soft = is_similar(img_category, text_category, threshold=self.similarity_threshold) or name_soft_agree
        if categories_match_soft:
            # 放宽条件: 只要模态一致,且至少一个置信度较高
            if img_confidence >= self.per_modality_conf_threshold or text_confidence >= self.per_modality_conf_threshold:
                trigger_reason["type"] = "modality_consistency"
                trigger_reason["img_conf"] = img_confidence
                trigger_reason["text_conf"] = text_confidence
                return False, fused_top1, float(max(img_confidence, text_confidence)), trigger_reason
        
        
        # 5. 两个模态的Top-1完全一致,即使置信度不高也信任（进一步放宽）
        if img_category == text_category or is_similar(img_category, text_category, threshold=0.4):
            if fused_top1_prob >= 0.2:  # 进一步降低到0.50
                trigger_reason["type"] = "exact_modality_match"
                trigger_reason["fused_prob"] = fused_top1_prob
                return False, fused_top1, fused_top1_prob, trigger_reason
        
        # 5.5. 两个模态的Top-1软一致且置信度中等
        if categories_match_soft and fused_top1_prob >= 0.3:
            trigger_reason["type"] = "soft_modality_match"
            trigger_reason["fused_prob"] = fused_top1_prob
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        # === 第二级: 中等置信度快速路径 ===
        # 6. 融合Top-1置信度中等(>=0.60)且margin中等(>=0.06)（进一步放宽）
        if fused_top1_prob >= 0.40 and fused_margin >= 0.06:
            trigger_reason["type"] = "medium_confidence_margin"
            trigger_reason["fused_prob"] = fused_top1_prob
            trigger_reason["margin"] = fused_margin
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        # 7. 融合Top-1置信度>=0.65,即使margin很小也信任（从0.70降低到0.65）
        if fused_top1_prob >= 0.4:
            trigger_reason["type"] = "high_prob_low_margin"
            trigger_reason["fused_prob"] = fused_top1_prob
            trigger_reason["margin"] = fused_margin
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        # 8. 融合Top-1置信度>=0.55且margin>=0.04,且模态一致（进一步放宽）
        if fused_top1_prob >= 0.4 and fused_margin >= 0.04:
            if categories_match_soft:
                trigger_reason["type"] = "relaxed_confidence_modality"
                trigger_reason["fused_prob"] = fused_top1_prob
                trigger_reason["margin"] = fused_margin
                return False, fused_top1, fused_top1_prob, trigger_reason
        
        # 8.5. 融合Top-1置信度>=0.60且Top-K重叠（新增）
        if fused_top1_prob >= 0.4 and topk_overlap:
            trigger_reason["type"] = "medium_prob_overlap_early"
            trigger_reason["fused_prob"] = fused_top1_prob
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        # === 第三级: 基于置信度的预判断（在LCB之前） ===
        # 9. 如果融合Top-1置信度>=0.60且margin>=0.05,直接通过（新增，在LCB之前）
        if fused_top1_prob >= 0.4 and fused_margin >= 0.05:
            trigger_reason["type"] = "pre_lcb_confidence"
            trigger_reason["fused_prob"] = fused_top1_prob
            trigger_reason["margin"] = fused_margin
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        # === 第四级: LCB判断 (动态阈值) ===
        # 准备置信度分数
        confidence_scores = [
            max(0.0, min(1.0, float(img_confidence))),
            max(0.0, min(1.0, float(text_confidence))),
            max(0.0, min(1.0, float(fused_top1_prob)))
        ]
        category_for_lcb = fused_top1
        
        # 冷启动保护
        if category_for_lcb not in self.category_stats:
            self.category_stats[category_for_lcb] = {"n": 0, "m": 0}
        
        # 计算LCB
        lcb_value = self.calculate_lcb(category_for_lcb, confidence_scores)
        
        # 获取自适应阈值
        adaptive_threshold = self._get_adaptive_lcb_threshold()
        
        # LCB判断 - 使用自适应阈值
        if lcb_value >= adaptive_threshold:
            trigger_reason["type"] = "lcb_pass"
            trigger_reason["lcb_value"] = lcb_value
            trigger_reason["threshold"] = adaptive_threshold
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        # === 第五级: LCB失败后的宽松判断 ===
        # 10. 如果LCB接近阈值且置信度>=0.55,也通过（新增）
        if lcb_value >= (adaptive_threshold * 0.7) and fused_top1_prob >= 0.5:
            trigger_reason["type"] = "lcb_near_threshold"
            trigger_reason["lcb_value"] = lcb_value
            trigger_reason["threshold"] = adaptive_threshold
            trigger_reason["fused_prob"] = fused_top1_prob
            return False, fused_top1, fused_top1_prob, trigger_reason
        
        # === 第六级: 宽松的快速判断 (LCB失败后的最后机会) ===
        # 11. 如果融合Top-1置信度较高(>=0.60)且margin较大(>=0.06),即使LCB不够高也信任
        if fused_top1_prob >= 0.5 and fused_margin >= 0.05:
            # 检查两个模态的Top-1是否一致
            if is_similar(img_category, text_category, threshold=0.4):
                trigger_reason["type"] = "high_prob_modality_match"
                trigger_reason["fused_prob"] = fused_top1_prob
                trigger_reason["margin"] = fused_margin
                return False, fused_top1, fused_top1_prob, trigger_reason        
        # === 需要慢思考 ===
        trigger_reason["type"] = "need_slow_thinking"
        trigger_reason["lcb_value"] = lcb_value
        trigger_reason["threshold"] = adaptive_threshold
        trigger_reason["fused_prob"] = fused_top1_prob
        trigger_reason["margin"] = fused_margin
        avg_confidence = (img_confidence + text_confidence) / 2
        return True, "conflict", avg_confidence, trigger_reason
    
    def update_stats(self, category: str, is_correct: bool, used_slow_thinking: bool = False):
        """更新统计量"""
        self.category_stats[category]["n"] += 1
        if is_correct:
            self.category_stats[category]["m"] += 1
        
        self.total_predictions += 1
        
        # 更新性能统计
        if used_slow_thinking:
            self.performance_stats["slow_path_count"] += 1
            if is_correct:
                self.performance_stats["slow_path_correct"] += 1
        else:
            self.performance_stats["fast_path_count"] += 1
            if is_correct:
                self.performance_stats["fast_path_correct"] += 1
        
        self.save_stats()
    
    def fast_thinking_pipeline(self, query_image_path: str, top_k: int = 5) -> Dict:
        """优化后的快思考完整流程"""
        # 1. 图像到图像检索
        img_category, img_confidence, img_results = self.image_to_image_retrieval(query_image_path, top_k)
        
        # 2. 图像到文本检索
        text_category, text_confidence, text_results = self.image_to_text_retrieval(query_image_path, top_k)
        
        # 3. 融合结果
        fused_results = self.fuse_results(img_results, text_results)
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
        
        # 名称软一致
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
        
        # 4. 优化后的触发器机制
        need_slow_thinking, predicted_category, confidence, trigger_reason = self.trigger_lcb_optimized(
            img_category, text_category, img_confidence, text_confidence,
            fused_top1, fused_top1_prob, fused_margin, topk_overlap, name_soft_agree
        )
        
        # 5. 计算LCB值用于后续质量评估
        lcb_map = {}
        confidence_scores = [img_confidence, text_confidence, fused_top1_prob]
        lcb_value = self.calculate_lcb(fused_top1, confidence_scores)
        lcb_map[fused_top1] = lcb_value
        
        # 对于fast-only流程,返回融合Top-1作为首选预测
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
            "text_results": text_results,
            "lcb_map": lcb_map,
            "trigger_reason": trigger_reason  # 添加触发原因
        }
        
        return result


