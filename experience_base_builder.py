"""
经验库构建模块 - 模型自反思与优化
基于Self-Belief和World-Belief的细粒度视觉识别策略优化

核心思想：
- Self-Belief：对自身当前推理能力与策略的元认知
- World-Belief：对每个细粒度类别的通用视觉-语义描述（由knowledge_base_builder提供）
- 通过失败经验 → 反思信念偏差 → 提炼策略规则 → 优化行为指令 → 实现策略进化
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from PIL import Image
import re
from collections import defaultdict

from agents.mllm_bot import MLLMBot
from knowledge_base_builder import KnowledgeBaseBuilder
from utils.fileios import dump_json, load_json
from utils.util import is_similar


class ExperienceBaseBuilder:
    """经验库构建器 - 基于模型自反思与优化"""
    
    # 初始Self-Belief策略
    INITIAL_SELF_BELIEF = """You are an expert in fine-grained visual recognition. Please follow these steps:
    1. Observe: First, look at the overall object and identify its coarse category (e.g., bird, dog, car).
    2. Localize: Then, identify the most discriminative local parts.
    3. Compare: Recall the visual characteristics of the candidate subcategories from your knowledge.
    4. Decide: Based on the match between observed details and subcategory descriptions, choose the most likely class.
    Answer only with the final class name."""
    
    def __init__(self,
                 mllm_bot: MLLMBot,
                 knowledge_base_builder: KnowledgeBaseBuilder,
                 fast_thinking_module,
                 slow_thinking_module,
                 device: str = "cuda",
                 dataset_info: dict = None):
        """
        初始化经验库构建器
        
        Args:
            mllm_bot: MLLM模型
            knowledge_base_builder: 知识库构建器（提供World-Belief）
            fast_thinking_module: 快思考模块
            slow_thinking_module: 慢思考模块
            device: 设备
        """
        self.mllm_bot = mllm_bot
        self.kb_builder = knowledge_base_builder
        self.fast_thinking = fast_thinking_module
        self.slow_thinking = slow_thinking_module
        self.device = device
        self.dataset_info = dataset_info or {}
        
        # Self-Belief：当前推理策略
        self.max_strategy_rules = 8
        self.strategy_rules = []
        self.next_rule_id = 1
        self.self_belief_core = self.INITIAL_SELF_BELIEF
        self.self_belief = self._compose_self_belief_prompt()
        
        # 伪轨迹存储：记录推理过程
        self.pseudo_trajectories = []  # List[Dict]: {image_path, cot, prediction, label, top_k_candidates}
        
        # 反思历史：记录每次反思的策略改进
        self.reflection_history = []  # List[Dict]: {rule_id, rule, failure_pattern, added_at}
        
        # 经验库存储路径
        self.save_dir = None
    
    def initialize_self_belief(self, custom_belief: Optional[str] = None):
        """
        初始化Self-Belief
        
        Args:
            custom_belief: 自定义的初始信念，如果为None则使用默认值
        """
        self.strategy_rules = []
        self.next_rule_id = 1
        if custom_belief:
            self.self_belief_core = custom_belief
        else:
            self.self_belief_core = self.INITIAL_SELF_BELIEF
        self.self_belief = self._compose_self_belief_prompt()
        
        print("Self-Belief已初始化")
        print(f"初始策略:\n{self.self_belief}")
    
    def _compose_self_belief_prompt(self) -> str:
        """
        构造包含策略规则的Self-Belief提示词
        """
        if not self.strategy_rules:
            return self.self_belief_core
        
        rules_text = "\n\nPolicy Memory (learned strategy rules):\n"
        for idx, rule in enumerate(self.strategy_rules, start=1):
            condition = rule.get("applicability_signals", "applicable to challenging cases")
            rules_text += f"{idx}. {rule['rule']} (Trigger: {condition})\n"
        return f"{self.self_belief_core}{rules_text}"
    
    def _refresh_self_belief_prompt(self):
        """
        根据当前策略规则刷新Self-Belief
        """
        self.self_belief = self._compose_self_belief_prompt()
    
    def generate_cot_with_self_belief(self, 
                                      image_path: str,
                                      top_k_candidates: List[Tuple[str, float]],
                                      world_belief_context: Optional[str] = None) -> Tuple[str, str]:
        """
        使用当前Self-Belief生成CoT推理链和最终预测
        
        Args:
            image_path: 图像路径
            top_k_candidates: Top-K候选类别列表 [(category, score), ...]
            world_belief_context: World-Belief上下文（类别描述等）
            
        Returns:
            Tuple[str, str]: (CoT推理链, 最终预测类别)
        """
        image = Image.open(image_path).convert("RGB")
        
        # 构建候选类别文本
        candidate_text = ""
        for i, (cat, score) in enumerate(top_k_candidates, start=1):
            candidate_text += f"{i}. {cat} (similarity: {score:.4f})\n"
        
        # 构建完整提示词：Self-Belief + World-Belief + 候选类别
        prompt = f"""{self.self_belief}
        Candidate classes (highly likely to contain the correct option):
        {candidate_text}"""     
        # 添加World-Belief上下文（类别描述）
        if world_belief_context:
            prompt += f"\n\nCategory descriptions:\n{world_belief_context}\n"
        
        prompt += """\nPlease analyze the image step by step and provide:
            1. Your reasoning chain (CoT) following the steps above
            2. Your final prediction (only the category name)
            Format your response as:
            Reasoning: [your step-by-step reasoning]
            Prediction: [category name]"""
                
        # 调用MLLM生成推理链
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
                for cat, _ in top_k_candidates:
                    if cat.lower() in last_line.lower() or last_line.lower() in cat.lower():
                        prediction = cat
                        break
                if prediction == "unknown":
                    # 使用第一个候选作为fallback
                    prediction = top_k_candidates[0][0] if top_k_candidates else "unknown"
            
        print(f"推理链: {cot}")
        print(f"预测: {prediction}")
        return cot, prediction
    
    def build_pseudo_trajectories(self,
                                  validation_samples: Dict[str, List[str]],
                                  top_k: int = 5,
                                  max_samples_per_category: Optional[int] = None) -> List[Dict]:
        """
        构造伪轨迹：使用当前Self-Belief + World-Belief进行推理
        
        Args:
            validation_samples: {true_category: [image_paths]} 验证样本
            top_k: 检索top-k结果
            max_samples_per_category: 每个类别最多处理的样本数
            
        Returns:
            List[Dict]: 伪轨迹列表，每个轨迹包含 {image_path, cot, prediction, label, top_k_candidates}
        """
        print("开始构造伪轨迹...")
        self.pseudo_trajectories = []
        
        total_samples = sum(len(paths) for paths in validation_samples.values())
        processed = 0
        
        for true_category, image_paths in validation_samples.items():
            print(f"处理类别: {true_category} ({len(image_paths)} 张图像)")
            
            # 限制每个类别的样本数
            if max_samples_per_category:
                image_paths = image_paths[:max_samples_per_category]
            
            # 获取World-Belief：该类别的描述
            world_belief_context = self._get_world_belief_context(true_category)
            
            for img_path in tqdm(image_paths, desc=f"处理 {true_category}"):
                # 1. 快思考获取Top-K候选
                fast_result = self.fast_thinking.fast_thinking_pipeline(img_path, top_k)
                top_k_candidates = fast_result.get("fused_results", [])[:top_k]
                
                if not top_k_candidates:
                    print(f"警告: {img_path} 没有检索到候选类别")
                    continue
                
                # 2. 使用Self-Belief生成CoT和预测
                cot, prediction = self.generate_cot_with_self_belief(
                    img_path, top_k_candidates, world_belief_context
                )
                
                # 3. 记录伪轨迹
                trajectory = {
                    "image_path": img_path,
                    "cot": cot,
                    "prediction": prediction,
                    "label": true_category,
                    "top_k_candidates": top_k_candidates,
                    "is_correct": is_similar(prediction, true_category, threshold=0.3)
                }
                
                self.pseudo_trajectories.append(trajectory)
                processed += 1
        
        print(f"伪轨迹构造完成! 共 {len(self.pseudo_trajectories)} 条轨迹")
        print(f"正确预测: {sum(1 for t in self.pseudo_trajectories if t['is_correct'])}")
        print(f"错误预测: {sum(1 for t in self.pseudo_trajectories if not t['is_correct'])}")
        
        return self.pseudo_trajectories
    
    def _get_world_belief_context(self, category: str) -> str:
        """
        获取World-Belief上下文（类别描述）
        
        Args:
            category: 类别名称
            
        Returns:
            str: 类别描述
        """
        if hasattr(self.kb_builder, 'category_descriptions'):
            # category_descriptions 应为 {类别名: 描述} 的字典
            description = self.kb_builder.category_descriptions.get(category, "")
            if description:
                return f"{category}: {description}"
        return f"{category}: No description available"
    
    def _normalize_rule_text(self, text: str) -> str:
        """
        规范化策略文本用于去重
        """
        cleaned = text.lower().strip()
        cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    def _aggregate_rules(self, candidates: List[Dict]) -> List[Dict]:
        """
        合并重复或相似的策略规则
        """
        aggregated = {}
        for candidate in candidates:
            rule_text = candidate.get("rule", "").strip()
            if not rule_text:
                continue
            normalized = self._normalize_rule_text(rule_text)
            if not normalized:
                continue
            if normalized in aggregated:
                aggregated_rule = aggregated[normalized]
                aggregated_rule["support"] += candidate.get("support", 1)
                aggregated_rule["source_images"].extend(candidate.get("source_images", []))
                aggregated_rule["labels"].extend(candidate.get("labels", []))
                aggregated_rule["failure_patterns"].append(candidate.get("failure_pattern", ""))
            else:
                candidate["normalized"] = normalized
                candidate.setdefault("support", 1)
                candidate.setdefault("source_images", [])
                candidate.setdefault("labels", [])
                candidate.setdefault("failure_patterns", [])
                aggregated[normalized] = candidate
        return list(aggregated.values())
    
    def _add_strategy_rule(self, rule_entry: Dict) -> bool:
        """
        将新的策略规则写入缓存，并根据容量控制策略
        """
        rule_text = rule_entry.get("rule", "").strip()
        if not rule_text:
            return False
        
        normalized = self._normalize_rule_text(rule_text)
        if not normalized:
            return False
        
        added = False
        for existing in self.strategy_rules:
            if existing.get("normalized") == normalized:
                existing["support"] += rule_entry.get("support", 1)
                existing["source_images"].extend(rule_entry.get("source_images", []))
                existing["labels"].extend(rule_entry.get("labels", []))
                existing["failure_patterns"].extend(rule_entry.get("failure_patterns", []))
                existing["priority"] = rule_entry.get("priority", existing.get("priority", "medium"))
                added = True
                break
        
        if not added:
            rule_entry["normalized"] = normalized
            rule_entry["id"] = f"R{self.next_rule_id}"
            self.next_rule_id += 1
            rule_entry.setdefault("support", 1)
            rule_entry.setdefault("source_images", [])
            rule_entry.setdefault("labels", [])
            rule_entry.setdefault("failure_patterns", [])
            rule_entry.setdefault("priority", "medium")
            rule_entry.setdefault("applicability_signals", "challenging scenarios")
            rule_entry.setdefault("effectiveness", 0.0)
            rule_entry.setdefault("applicability", 0.0)
            self.strategy_rules.append(rule_entry)
            self.reflection_history.append({
                "rule_id": rule_entry["id"],
                "rule": rule_entry["rule"],
                "failure_pattern": rule_entry.get("failure_pattern", ""),
                "source_images": rule_entry.get("source_images", []),
                "added_at": len(self.reflection_history) + 1
            })
            added = True
        
        if len(self.strategy_rules) > self.max_strategy_rules:
            self._manage_rule_capacity()
        
        return added
    
    def _manage_rule_capacity(self):
        """
        当策略缓存超过容量时，调用评估机制保留最有价值的规则
        """
        if len(self.strategy_rules) <= self.max_strategy_rules:
            return
        
        ranked = self._rank_rules_for_retention()
        if ranked:
            keep_ids = set(ranked.get("keep_ids", []))
            if keep_ids:
                self.strategy_rules = [rule for rule in self.strategy_rules if rule["id"] in keep_ids]
        if len(self.strategy_rules) > self.max_strategy_rules:
            self.strategy_rules = self.strategy_rules[:self.max_strategy_rules]
    
    def _rank_rules_for_retention(self) -> Dict:
        """
        使用MLLM评估策略规则的保留优先级
        """
        summary_lines = []
        for rule in self.strategy_rules:
            summary_lines.append(
                f"{rule['id']}: {rule['rule']} | support={rule.get('support', 1)} | "
                f"priority={rule.get('priority', 'medium')} | effectiveness={rule.get('effectiveness', 0.0)} | "
                f"applicability={rule.get('applicability', 0.0)}"
            )
        
        prompt = f"""You are maintaining a policy rule cache for fine-grained recognition. Capacity is {self.max_strategy_rules}.
Rules:
{chr(10).join(summary_lines)}

Please decide which rules to keep to maximize generalization coverage. Return ONLY a JSON object:
{{
    "keep_ids": ["R1", "R2", ...],  // exactly {self.max_strategy_rules} IDs if possible
    "drop_ids": ["R3", ...]
}}"""
        
        dummy_image = Image.new('RGB', (224, 224), color='white')
        reply, response = self.mllm_bot.describe_attribute(dummy_image, prompt)
        if isinstance(response, list):
            response = " ".join(response)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            decision = json.loads(json_str)
            return decision
        return {}
    
    def reflect_on_failed_samples(self,
                                  failed_trajectories: List[Dict],
                                  max_reflections: int = 5) -> List[Dict]:
        """
        对失败的样本进行反思，聚焦于推理策略缺陷
        
        Args:
            failed_trajectories: 失败的轨迹列表
            max_reflections: 最多反思的样本数
            
        Returns:
            List[Dict]: 新生成的策略规则
        """
        if not failed_trajectories:
            return []
        
        print(f"开始反思 {len(failed_trajectories)} 个失败样本...")
        
        # 选择最多max_reflections个样本进行反思
        selected_samples = failed_trajectories[:max_reflections]
        
        rule_candidates = []
        
        for i, traj in enumerate(selected_samples, start=1):
            print(f"反思样本 {i}/{len(selected_samples)}: {traj['image_path']}")
            
            # 获取World-Belief上下文
            world_belief_context = self._get_world_belief_context(traj['label'])
            
            # 构建候选类别文本
            candidate_text = ""
            for j, (cat, score) in enumerate(traj['top_k_candidates'], start=1):
                candidate_text += f"{j}. {cat} (similarity: {score:.4f})\n"
            
            # 构建反思提示词
            reflection_prompt = f"""You are reviewing a failed fine-grained recognition case.
Candidate classes (highly likely to contain the correct option):
{candidate_text}
Your reasoning: "{traj['cot']}"
Prediction: {traj['prediction']}
Ground truth: {traj['label']}

Please reflect at the strategy level:
1. Correctness: Did your reasoning correctly interpret visual evidence?
2. Consistency: Was your step-by-step logic consistent with known class characteristics?
3. Rationality: Did you focus on the right discriminative parts? Did you ignore key details?
4. Generalization: What general lesson can be learned to avoid similar mistakes on other images?
Then, propose a domain-general rule of thumb that applies to future FGVR cases.

Return ONLY a JSON object:
{{
    "failure_pattern": "concise description of the challenging scenario",
    "root_cause": "brief diagnosis of the reasoning flaw",
    "improved_rule": "Actionable instruction starting with a verb",
    "applicability_signals": "When should this rule be recalled?",
    "priority": "high|medium|low"
}}"""
            
            # 调用MLLM进行反思
            image = Image.open(traj['image_path']).convert("RGB")
            reply, response = self.mllm_bot.describe_attribute(image, reflection_prompt)
            if isinstance(response, list):
                response = " ".join(response)
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
            else:
                parsed = {}
            
            improved_rule = parsed.get("improved_rule", "").strip()
            failure_pattern = parsed.get("failure_pattern", "").strip()
            applicability_signals = parsed.get("applicability_signals", "").strip()
            priority = parsed.get("priority", "medium").strip()
            
            candidate = {
                "rule": improved_rule,
                "failure_pattern": failure_pattern,
                "applicability_signals": applicability_signals or "similar ambiguous cases",
                "priority": priority or "medium",
                "support": 1,
                "source_images": [traj['image_path']],
                "labels": [traj['label']],
                "root_cause": parsed.get("root_cause", "").strip(),
                "source_reasoning": traj['cot']
            }
            rule_candidates.append(candidate)
        
        refined_rules = self._aggregate_rules(rule_candidates)
        return refined_rules
    
    def update_self_belief(self, refined_rules: List[Dict]) -> bool:
        """
        将提炼的策略改进合并到Self-Belief中
        
        Args:
            refined_rules: 提炼后的策略规则列表
            
        Returns:
            bool: 是否成功更新
        """
        if not refined_rules:
            print("警告: 策略改进为空，跳过更新")
            return False
        
        updated = False
        for rule_entry in refined_rules:
            added = self._add_strategy_rule(rule_entry)
            updated = updated or added
            if added:
                print(f"新增策略规则: {rule_entry.get('rule', '').strip()}")
        
        if updated:
            self._refresh_self_belief_prompt()
            print("Self-Belief已更新并包含最新策略规则")
        else:
            print("未添加新策略规则，Self-Belief保持不变")
        
        return updated
    
    def evaluate_with_current_belief(self,
                                     validation_samples: Dict[str, List[str]],
                                     top_k: int = 5) -> Dict:
        """
        使用当前Self-Belief在验证集上评估性能
        
        Args:
            validation_samples: {true_category: [image_paths]} 验证样本
            top_k: 检索top-k结果
            
        Returns:
            Dict: 评估结果 {accuracy, correct_count, total_count, ...}
        """
        print("使用当前Self-Belief评估性能...")
        
        correct_count = 0
        total_count = 0
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for true_category, image_paths in validation_samples.items():
            world_belief_context = self._get_world_belief_context(true_category)
            
            for img_path in image_paths:
                # 快思考获取Top-K
                fast_result = self.fast_thinking.fast_thinking_pipeline(img_path, top_k)
                top_k_candidates = fast_result.get("fused_results", [])[:top_k]

                
                # 使用当前Self-Belief生成预测
                cot, prediction = self.generate_cot_with_self_belief(
                    img_path, top_k_candidates, world_belief_context
                )
                
                # 判断是否正确
                is_correct = is_similar(prediction, true_category, threshold=0.4)
                print(f"evaluate_with_current_belief, 预测: {prediction}, 真实: {true_category}, 是否正确: {is_correct}")
                if is_correct:
                    correct_count += 1
                    category_stats[true_category]["correct"] += 1
                
                total_count += 1
                category_stats[true_category]["total"] += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        result = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "category_stats": dict(category_stats)
        }
        
        print(f"评估完成: 准确率 = {accuracy:.4f} ({correct_count}/{total_count})")
        
        return result
    
    def build_experience_base(self,
                              validation_samples: Dict[str, List[str]],
                              max_iterations: int = 3,
                              top_k: int = 5,
                              max_reflections_per_iter: int = 5,
                              min_improvement: float = 0.01,
                              max_samples_per_category: Optional[int] = None) -> Dict:
        """
        构建经验库：迭代优化Self-Belief
        
        Args:
            validation_samples: {true_category: [image_paths]} 验证样本
            max_iterations: 最大迭代次数
            top_k: 检索top-k结果
            max_reflections_per_iter: 每次迭代最多反思的样本数
            min_improvement: 最小性能提升阈值（如果提升小于此值，停止迭代）
            max_samples_per_category: 每个类别最多处理的样本数
            
        Returns:
            Dict: 构建结果
        """
        print("=" * 60)
        print("开始构建经验库（模型自反思与优化）")
        print("=" * 60)
        
        # 初始化Self-Belief
        self.initialize_self_belief()
        
        # 记录初始性能
        initial_performance = self.evaluate_with_current_belief(validation_samples, top_k)
        best_accuracy = initial_performance["accuracy"]
        best_self_belief = self.self_belief
        
        print(f"\n初始准确率: {best_accuracy:.4f}")
        
        iteration_results = []
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'=' * 60}")
            print(f"迭代 {iteration}/{max_iterations}")
            print(f"{'=' * 60}")
            
            # 1. 构造伪轨迹
            trajectories = self.build_pseudo_trajectories(
                validation_samples, top_k, max_samples_per_category
            )
            
            # 2. 识别失败样本
            failed_trajectories = [t for t in trajectories if not t['is_correct']]
            
            if not failed_trajectories:
                print("所有样本都预测正确，无需进一步优化！")
                break
            
            print(f"失败样本数: {len(failed_trajectories)}")
            
            # 3. 对失败样本进行反思
            refined_rules = self.reflect_on_failed_samples(
                failed_trajectories, max_reflections_per_iter
            )
            
            if not refined_rules:
                print("未能提炼出策略改进，跳过本次迭代")
                continue
            else:
                print(f"提炼出 {len(refined_rules)} 条策略改进")
            # 4. 更新Self-Belief
            updated = self.update_self_belief(refined_rules)
            if not updated:
                print("策略规则未发生变化，继续下一轮迭代")
                continue
            
            # 5. 在验证集上重新评估
            new_performance = self.evaluate_with_current_belief(validation_samples, top_k)
            new_accuracy = new_performance["accuracy"]
            
            performance_change = new_accuracy - best_accuracy
            
            print(f"\n性能变化: {best_accuracy:.4f} → {new_accuracy:.4f} (变化: {performance_change:+.4f})")
            
            # 6. 判断是否保留更新
            if performance_change >= min_improvement:
                print(f"性能提升 {performance_change:.4f} >= {min_improvement:.4f}，保留策略更新")
                best_accuracy = new_accuracy
                best_self_belief = self.self_belief
            else:
                print(f"性能提升 {performance_change:.4f} < {min_improvement:.4f}，回退策略更新")
                # 回退Self-Belief
                self.self_belief = best_self_belief
            
            # 记录迭代结果
            iteration_result = {
                "iteration": iteration,
                "failed_samples_count": len(failed_trajectories),
                "refined_rules": [rule.get("rule", "") for rule in refined_rules],
                "accuracy_before": best_accuracy - performance_change if performance_change >= min_improvement else best_accuracy,
                "accuracy_after": new_accuracy,
                "performance_change": performance_change,
                "strategy_accepted": performance_change >= min_improvement
            }
            iteration_results.append(iteration_result)
            
            # 如果性能不再提升，提前停止
            if performance_change < min_improvement and iteration > 1:
                print("性能不再提升，提前停止迭代")
                break
        
        # 使用最佳Self-Belief
        self.self_belief = best_self_belief
        
        # 最终评估
        final_performance = self.evaluate_with_current_belief(validation_samples, top_k)
        
        result = {
            "initial_accuracy": initial_performance["accuracy"],
            "final_accuracy": final_performance["accuracy"],
            "improvement": final_performance["accuracy"] - initial_performance["accuracy"],
            "best_self_belief": best_self_belief,
            "iteration_results": iteration_results,
            "total_iterations": len(iteration_results)
        }
        
        print(f"\n{'=' * 60}")
        print("经验库构建完成!")
        print(f"初始准确率: {initial_performance['accuracy']:.4f}")
        print(f"最终准确率: {final_performance['accuracy']:.4f}")
        print(f"性能提升: {result['improvement']:+.4f}")
        print(f"{'=' * 60}")
        
        return result
    
    def save_experience_base(self, save_dir: str):
        """
        保存经验库到文件
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        
        # 保存Self-Belief
        belief_path = os.path.join(save_dir, "self_belief.txt")
        with open(belief_path, 'w', encoding='utf-8') as f:
            f.write(self.self_belief)
        
        # 保存伪轨迹
        trajectories_path = os.path.join(save_dir, "pseudo_trajectories.json")
        dump_json(trajectories_path, self.pseudo_trajectories)
        
        # 保存反思历史
        history_path = os.path.join(save_dir, "reflection_history.json")
        dump_json(history_path, self.reflection_history)
        
        # 保存策略规则
        rules_path = os.path.join(save_dir, "strategy_rules.json")
        dump_json(rules_path, self.strategy_rules)
        
        print(f"经验库已保存到: {save_dir}")
    
    def load_experience_base(self, load_dir: str):
        """
        从文件加载经验库
        
        Args:
            load_dir: 加载目录
        """
        self.save_dir = load_dir
        
        # 加载Self-Belief
        belief_path = os.path.join(load_dir, "self_belief.txt")
        if os.path.exists(belief_path):
            with open(belief_path, 'r', encoding='utf-8') as f:
                self.self_belief = f.read()
            # print(f"Self-Belief已从 {belief_path} 加载")
            print(f'加载self_belief:{self.self_belief}')
        
        # # 加载伪轨迹
        # trajectories_path = os.path.join(load_dir, "pseudo_trajectories.json")
        # if os.path.exists(trajectories_path):
        #     self.pseudo_trajectories = load_json(trajectories_path)
        #     print(f"伪轨迹已从 {trajectories_path} 加载 ({len(self.pseudo_trajectories)} 条)")
        
        # # 加载反思历史
        # history_path = os.path.join(load_dir, "reflection_history.json")
        # if os.path.exists(history_path):
        #     self.reflection_history = load_json(history_path)
        #     print(f"反思历史已从 {history_path} 加载 ({len(self.reflection_history)} 条记录)")
        
        # 加载策略规则
        # rules_path = os.path.join(load_dir, "strategy_rules.json")
        # if os.path.exists(rules_path):
        #     self.strategy_rules = load_json(rules_path)
        #     print(f"策略规则已从 {rules_path} 加载 ({len(self.strategy_rules)} 条)")
        #     print(f'加载的strategy_rules:{self.strategy_rules}')
        # else:
        #     self.strategy_rules = []
        # if self.strategy_rules:
        #     last_numeric_id = max(int(rule["id"].lstrip("R")) for rule in self.strategy_rules if "id" in rule)
        #     self.next_rule_id = last_numeric_id + 1
        # else:
        #     self.next_rule_id = 1
        
        # 刷新Self-Belief 暂时不用
        # self.self_belief = self._compose_self_belief_prompt()
        
        print(f"经验库已从 {load_dir} 加载完成")
    
    def get_self_belief(self) -> str:
        """获取当前Self-Belief"""
        return self.self_belief
    
    def get_pseudo_trajectories(self) -> List[Dict]:
        """获取伪轨迹列表"""
        return self.pseudo_trajectories


# 示例使用
if __name__ == "__main__":
    from agents.mllm_bot import MLLMBot
    from knowledge_base_builder import KnowledgeBaseBuilder
    from fast_thinking_optimized import FastThinkingOptimized
    from slow_thinking_optimized import SlowThinkingOptimized
    
    # 初始化组件
    mllm_bot = MLLMBot(model_tag="Qwen2.5-VL-7B", model_name="Qwen2.5-VL-7B", device="cuda")
    kb_builder = KnowledgeBaseBuilder(device="cuda")
    fast_thinking = FastThinkingOptimized(kb_builder, device="cuda")
    slow_thinking = SlowThinkingOptimized(mllm_bot, kb_builder, fast_thinking, device="cuda")
    
    # 初始化经验库构建器
    exp_builder = ExperienceBaseBuilder(
        mllm_bot=mllm_bot,
        knowledge_base_builder=kb_builder,
        fast_thinking_module=fast_thinking,
        slow_thinking_module=slow_thinking,
        device="cuda"
    )
    
    # 示例验证样本
    validation_samples = {
        "Chihuahua": ["path/to/chihuahua1.jpg", "path/to/chihuahua2.jpg"],
        "Shiba Inu": ["path/to/shiba1.jpg", "path/to/shiba2.jpg"]
    }
    
    # 构建经验库
    result = exp_builder.build_experience_base(validation_samples, max_iterations=1)
    
    # 保存经验库
    exp_builder.save_experience_base("./experience_base")
    
    print("经验库构建完成!")

