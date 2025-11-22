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
import os
from collections import Counter, defaultdict

class FastThinking:
    """快思考模块"""
    
    def __init__(self, knowledge_base_builder: KnowledgeBaseBuilder, 
                 confidence_threshold: float = 0.8,
                 similarity_threshold: float = 0.7,
                 # 这两个值调过了，应该是最优区间
                 fusion_weight: float = 0.05,
                 softmax_temp: float = 0.07,
                 # 优化的触发阈值 - 平衡速度和准确率
                 fused_conf_threshold: float = 0.70,  # 从0.75降低到0.70,增加慢思考触发
                 fused_margin_threshold: float = 0.18,  # 从0.15提高到0.18,更严格
                 per_modality_conf_threshold: float = 0.68,  # 从0.65提高到0.68,更严格
                 consider_topk_overlap: bool = True,
                 topk_for_overlap: int = 3,
                 # —— LCB 相关默认参数 - 优化 ——
                 stats_file: str = None,  # 将从dataset_info自动推断
                 lcb_threshold: float = 0.68,  # 从0.65提高到0.68,增加慢思考触发
                 lcb_threshold_adaptive: bool = True,  # 启用自适应阈值
                 lcb_threshold_min: float = 0.60,  # 最小阈值
                 lcb_threshold_max: float = 0.78,  # 最大阈值
                 prior_strength: float = 2.0,
                 prior_p: float = 0.6,
                 lcb_eta: float = 1.0,
                 lcb_alpha: float = 0.5,
                 lcb_epsilon: float = 1e-6,
                 # 数据集信息
                 dataset_info: dict = None):
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
        # LCB 参数与状态
        self.lcb_threshold = lcb_threshold
        self.lcb_threshold_adaptive = lcb_threshold_adaptive
        self.lcb_threshold_min = lcb_threshold_min
        self.lcb_threshold_max = lcb_threshold_max
        self.prior_strength = prior_strength
        self.prior_p = prior_p
        self.lcb_eta = lcb_eta
        self.lcb_alpha = lcb_alpha
        self.lcb_epsilon = lcb_epsilon
        
        # 数据集信息 - 自动推断stats_file路径
        self.dataset_info = dataset_info or {}
        if stats_file is None:
            if not self.dataset_info or 'stats_file_full' not in self.dataset_info:
                raise ValueError(
                    "FastThinking初始化失败: 必须提供dataset_info或明确指定stats_file。"
                    "dataset_info应包含'stats_file_full'字段，以避免误污染知识库。"
                )
            # 从dataset_info构建stats_file路径
            self.stats_file = self.dataset_info['stats_file_full']
            print(f"使用数据集stats文件: {self.stats_file}")
        else:
            self.stats_file = stats_file
            print(f"使用指定的stats文件: {self.stats_file}")
        
        self.total_predictions = 1  # 避免对数为0
        # category -> {"n": 历史命中次数, "m": 历史正确次数}
        self.category_stats = defaultdict(lambda: {"n": 0, "m": 0})  # n: 检索命中次数, m: 预测正确次数
        # 性能统计
        self.performance_stats = {
            "fast_path_count": 0,
            "slow_path_count": 0,
            "fast_path_correct": 0,
            "slow_path_correct": 0
        }
        # 加载历史统计量
        self.load_stats()



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
    
    def trigger_lcb(self, img_category: str, text_category: str, 
                         img_confidence: float, text_confidence: float,
                         fused_top1: str, fused_top1_prob: float, fused_margin: float,
                         topk_overlap: bool, name_soft_agree: bool) -> Tuple[bool, str, float]:
        """
        优化后的触发器机制：多级阈值判断,更早地拒绝慢思考
        
        Args:
            img_category: 图像检索结果
            text_category: 文本检索结果
            img_confidence: 图像检索置信度
            text_confidence: 文本检索置信度
            
        Returns:
            Tuple[bool, str, float]: (是否需要慢思考, 预测类别, 置信度)
        """
        # === 第一级: 严格的快速路径判断 ===
        # 1. 融合Top-1置信度足够高且margin足够大 - 最严格条件
        # 提高要求：同时检查置信度和margin，并且要求两个模态都较高
        if (fused_top1_prob >= self.fused_conf_threshold and 
            fused_margin >= self.fused_margin_threshold and
            img_confidence >= 0.60 and text_confidence >= 0.60):
            return False, fused_top1, fused_top1_prob

        # 2. 两个模态高度一致且各自置信度都很高 - 更严格的条件
        categories_match_soft = is_similar(img_category, text_category, threshold=self.similarity_threshold) or name_soft_agree
        if (categories_match_soft and 
            img_confidence >= self.per_modality_conf_threshold and 
            text_confidence >= self.per_modality_conf_threshold and
            fused_top1_prob >= 0.65):  # 额外要求融合概率也较高
            return False, fused_top1, float(max(img_confidence, text_confidence))

        # 3. Top-K重叠且融合Top-1置信度较高 - 更严格的条件
        if (self.consider_topk_overlap and topk_overlap and 
            fused_top1_prob >= (self.fused_conf_threshold * 0.95) and  # 提高到0.95
            fused_margin >= (self.fused_margin_threshold * 0.8)):  # 要求margin也较高
            return False, fused_top1, fused_top1_prob

        # === 第二级: LCB判断 ===
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
        
        self.total_predictions = max(1, int(self.total_predictions) + 1)
        
        # 计算LCB
        lcb_value = self.calculate_lcb(category_for_lcb, confidence_scores)
        
        # 获取自适应阈值
        adaptive_threshold = self._get_adaptive_lcb_threshold()
        
        # LCB判断 - 使用自适应阈值
        if lcb_value >= adaptive_threshold:
            return False, fused_top1, fused_top1_prob

        # === 第三级: 额外的快速判断 - 更严格的条件 ===
        # 如果融合Top-1置信度很高(>=0.75)且margin很大(>=0.15),即使LCB不够高也信任
        # 同时要求两个模态都较高且一致
        if (fused_top1_prob >= 0.75 and 
            fused_margin >= 0.15 and
            img_confidence >= 0.65 and 
            text_confidence >= 0.65):
            # 检查两个模态的Top-1是否一致
            if is_similar(img_category, text_category, threshold=0.85):  # 提高相似度阈值
                return False, fused_top1, fused_top1_prob

        # === 需要慢思考 ===
        # 对于不确定的情况,倾向于使用慢思考以提高准确率
        avg_confidence = (img_confidence + text_confidence) / 2
        return True, "conflict", avg_confidence
    
    def calculate_lcb(self, category: str, confidence_scores: List[float]) -> float:
        """
        计算类别的LCB (Lower Confidence Bound)，用于判断是否需要触发慢思考。
        基于Beta先验 + 置信度分布熵调节。

        Args:
            category: 类别名称
            confidence_scores: 当前检索结果的置信度分数列表

        Returns:
            float: LCB值
        """
        import math
        stats = self.category_stats[category]
        n_raw = stats["n"]  # 历史命中次数
        m_raw = stats["m"]  # 历史正确次数

        # ---- Beta先验平滑（缓解冷启动）----
        n = n_raw + self.prior_strength
        m = m_raw + self.prior_p * self.prior_strength
        p_hat = m / (n + self.lcb_epsilon)

        # ---- 计算置信度分布熵（反映模型犹豫程度）----
        if len(confidence_scores) > 1:
            probs = np.array(confidence_scores)
            probs = probs / (probs.sum() + 1e-12)
            entropy = -np.sum(probs * np.log(probs + 1e-12)) / np.log(len(probs) + 1e-12)  # 归一化熵 ∈ [0,1]
        else:
            entropy = 0.0

        # ---- 置信区间项 ----
        if n_raw > 0:
            confidence_term = self.lcb_eta * math.sqrt(math.log(max(1, self.total_predictions)) / (2 * n + 1))
        else:
            confidence_term = self.lcb_eta * math.sqrt(math.log(max(1, self.total_predictions)))

        # ---- 最终LCB公式 ----
        # 熵越高（模型越犹豫），LCB越低，更可能触发慢思考
        lcb = p_hat - confidence_term - self.lcb_alpha * entropy

        return max(0.0, min(1.0, lcb))
    
    def update_stats(self, category: str, is_correct: bool, used_slow_thinking: bool = False):
        """
        更新统计量
        
        Args:
            category: 预测的类别
            is_correct: 预测是否正确
            used_slow_thinking: 是否使用了慢思考
            n 为命中次数， m为正确次数
        """
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
    
    def save_stats(self):
        """保存统计量到文件"""
        data = {
            "category_stats": dict(self.category_stats),
            "total_predictions": self.total_predictions,
            "performance_stats": self.performance_stats
        }
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _get_adaptive_lcb_threshold(self) -> float:
        """根据历史性能自适应调整LCB阈值"""
        if not self.lcb_threshold_adaptive:
            return self.lcb_threshold
        
        # 计算快速路径的正确率
        fast_path_acc = 0.0
        if self.performance_stats["fast_path_count"] > 0:
            fast_path_acc = self.performance_stats["fast_path_correct"] / self.performance_stats["fast_path_count"]
        
        # 如果快速路径正确率很高(>0.90),可以稍微提高阈值
        if fast_path_acc > 0.90:
            adaptive_threshold = min(self.lcb_threshold_max, self.lcb_threshold + 0.03)
        # 如果快速路径正确率较低(<0.75),降低阈值(更宽松,增加慢思考以提高准确率)
        elif fast_path_acc < 0.75:
            adaptive_threshold = max(self.lcb_threshold_min, self.lcb_threshold - 0.08)
        # 如果快速路径正确率中等,保持阈值或稍微降低
        elif fast_path_acc < 0.85:
            adaptive_threshold = max(self.lcb_threshold_min, self.lcb_threshold - 0.03)
        else:
            adaptive_threshold = self.lcb_threshold
        
        return adaptive_threshold
    
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
        # need_slow_thinking, predicted_category, confidence = self.trigger_mechanism(
        #     img_category, text_category, img_confidence, text_confidence,
        #     fused_top1, fused_top1_prob, fused_margin, topk_overlap, name_soft_agree
        # )
        need_slow_thinking, predicted_category, confidence = self.trigger_lcb(
            img_category, text_category, img_confidence, text_confidence,
            fused_top1, fused_top1_prob, fused_margin, topk_overlap, name_soft_agree
        )
        
        # 5. 计算LCB值用于后续质量评估
        lcb_map = {}
        confidence_scores = [img_confidence, text_confidence, fused_top1_prob]
        lcb_value = self.calculate_lcb(fused_top1, confidence_scores)
        lcb_map[fused_top1] = lcb_value

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
            "text_results": text_results,
            "lcb_map": lcb_map  # 添加LCB值用于质量评估
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
