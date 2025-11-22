"""
快慢思考细粒度图像识别系统
整合快思考和慢思考模块，实现完整的FGVR流程

功能：
1. 知识库构建和管理
2. 快思考流程（CLIP双模态检索）
3. 慢思考流程（MLLM+CLIP深度分析）
4. 结果融合和评估
5. 批量处理接口
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import torch
from collections import defaultdict
import threading
from datetime import datetime

from agents.mllm_bot import MLLMBot
from knowledge_base_builder import KnowledgeBaseBuilder
from experience_base_builder import ExperienceBaseBuilder
from fast_thinking import FastThinking
from fast_thinking_optimized import FastThinkingOptimized
from slow_thinking_optimized import SlowThinkingOptimized
from slow_thinking import SlowThinking
from utils.fileios import dump_json, load_json
from utils.util import is_similar


class FastSlowThinkingSystem:
    """快慢思考细粒度图像识别系统"""
    
    def __init__(self, 
                 model_tag: str = "Qwen2.5-VL-7B",
                 model_name: str = "Qwen2.5-VL-7B",
                 image_encoder_name: str = "./models/Clip/clip-vit-base-patch32",
                 text_encoder_name: str = "./models/Clip/clip-vit-base-patch32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 cfg: Optional[Dict] = None,
                 dataset_info: Optional[Dict] = None):
        """
        初始化快慢思考系统
        
        Args:
            model_tag: MLLM模型标签
            model_name: MLLM模型名称
            image_encoder_name: 图像编码器名称
            text_encoder_name: 文本编码器名称
            device: 设备
            cfg: 配置参数
        """
        self.device = device
        self.cfg = cfg or {}
        self.dataset_info = dataset_info or {}

        print("\n================= 快慢思考系统初始化 =================")
        print(f"🖥️ 设备: {self.device}")
        print(f"🤖 MLLM 模型标签: {model_tag}")
        print(f"🤖 MLLM 模型名称: {model_name}")
        print(f"🖼️ 图像编码器: {image_encoder_name}")
        print(f"📄 文本编码器: {text_encoder_name}")
        print("📚 数据集信息:")
        print(json.dumps(self.dataset_info, indent=4, ensure_ascii=False))
        print("====================================================\n")
        
        # 初始化MLLM
        print("🚀初始化MLLM模型...")
        self.mllm_bot = MLLMBot(
            model_tag=model_tag,
            model_name=model_name,
            device=device
        )
        print(f"✅ MLLM 初始化完成, 当前使用精度: {getattr(self.mllm_bot, 'dtype_used', '未知')}")
        
        # 初始化知识库构建器
        print("🚀 初始化知识库构建器...")
        self.kb_builder = KnowledgeBaseBuilder(
            image_encoder_name=image_encoder_name,
            text_encoder_name=text_encoder_name,
            device=device,
            cfg=cfg,
            dataset_info=self.dataset_info
        )
        print("✅ 知识库构建器初始化完成")
        
        # 初始化快思考模块
        print("🚀 初始化快思考模块...")
        self.fast_thinking = FastThinkingOptimized(
            knowledge_base_builder=self.kb_builder,
            confidence_threshold=self.cfg.get('confidence_threshold', 0.8),
            similarity_threshold=self.cfg.get('similarity_threshold', 0.7),
            dataset_info=self.dataset_info
        )
        print("✅ 快思考模块初始化完成")
        
        # 初始化慢思考模块
        print("🚀 初始化慢思考模块...")
        self.slow_thinking = SlowThinkingOptimized(
            mllm_bot=self.mllm_bot,
            knowledge_base_builder=self.kb_builder,
            fast_thinking=self.fast_thinking,
            dataset_info=self.dataset_info
        )
        print("✅ 慢思考模块初始化完成")
        
        # 初始化经验库构建器（可选）
        self.exp_builder = None
        
        # 启动显存监控线程
        print("⏱️ 启动显存监控线程...")
        self.memory_monitor_stop = threading.Event()
        self.memory_monitor_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True
        )
        self.memory_monitor_thread.start()

        print("🎉快慢思考系统初始化完成!")
        print("====================================================\n")
        
    def __del__(self):
        """析构函数：清理系统资源"""
        self.cleanup()
        
    def _monitor_memory(self):
        """后台线程：每3秒记录一次显存使用情况"""
        while not self.memory_monitor_stop.is_set():
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_free = memory_total - memory_allocated
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] [显存监控] 已分配={memory_allocated:.2f}GB, 已保留={memory_reserved:.2f}GB, 空闲={memory_free:.2f}GB")
            self.memory_monitor_stop.wait(3)
    
    def cleanup(self):
        """手动清理系统资源"""
        try:
            # 停止显存监控线程
            if hasattr(self, 'memory_monitor_stop'):
                self.memory_monitor_stop.set()
            if hasattr(self, 'memory_monitor_thread') and self.memory_monitor_thread.is_alive():
                self.memory_monitor_thread.join(timeout=2)
            
            if hasattr(self, 'mllm_bot') and self.mllm_bot:
                self.mllm_bot.cleanup()
                del self.mllm_bot
            if hasattr(self, 'kb_builder'):
                del self.kb_builder
            if hasattr(self, 'fast_thinking'):
                del self.fast_thinking
            if hasattr(self, 'slow_thinking'):
                del self.slow_thinking
            if hasattr(self, 'exp_builder'):
                del self.exp_builder
            torch.cuda.empty_cache()
            print("FastSlowThinkingSystem资源已清理")
        except Exception as e:
            print(f"清理FastSlowThinkingSystem资源时出错: {e}")
    
    def build_knowledge_base(self, train_samples: Dict[str, List[str]], 
                           save_dir: str = "./knowledge_base",
                           augmentation: bool = True) -> Tuple[Dict, Dict]:
        """
        构建知识库
        
        Args:
            train_samples: {category: [image_paths]} 训练样本
            save_dir: 知识库保存目录
            augmentation: 是否进行数据增强
            
        Returns:
            Tuple[Dict, Dict]: (image_kb, text_kb)
        """
        print("构建知识库...")
        print(f"训练样本包含 {len(train_samples)} 个类别")
        
        # 构建知识库
        image_kb, text_kb = self.kb_builder.build_knowledge_base(
            self.mllm_bot, train_samples, augmentation
        )
        
        # 保存知识库
        self.kb_builder.save_knowledge_base(save_dir)

        self.initialize_experience_base()
        self.exp_builder.build_experience_base(train_samples,max_iterations=1, max_reflections_per_iter=3, top_k=5)
        self.exp_builder.save_experience_base(save_dir)
        # 使用训练集初始化LCB统计参数
        print("\n开始初始化LCB统计参数...")
        stats_summary = self.kb_builder.initialize_lcb_stats_with_labels(
            train_samples, self.fast_thinking, top_k=5
        )
        print(f"初始化准确率: {stats_summary['initialization_accuracy']:.4f}")

        print("知识库构建完成!")
        return image_kb, text_kb
    
    def load_knowledge_base(self, load_dir: str = "./knowledge_base"):
        """
        加载知识库
        
        Args:
            load_dir: 知识库加载目录
        """
        print(f"从 {load_dir} 加载知识库...")
        self.kb_builder.load_knowledge_base(load_dir)
        print("知识库加载完成!")
    
    def initialize_experience_base(self):
        """
        初始化经验库构建器
        """
        if self.exp_builder is None:
            print("初始化经验库构建器...")
            self.exp_builder = ExperienceBaseBuilder(
                mllm_bot=self.mllm_bot,
                knowledge_base_builder=self.kb_builder,
                fast_thinking_module=self.fast_thinking,
                slow_thinking_module=self.slow_thinking,
                device=self.device,
                dataset_info=self.dataset_info
            )
            print("经验库构建器初始化完成!")
        return self.exp_builder
    
    def build_experience_base(self,
                              validation_samples: Dict[str, List[str]],
                              max_iterations: int = 3,
                              top_k: int = 5,
                              max_reflections_per_iter: int = 5,
                              min_improvement: float = 0.01,
                              max_samples_per_category: Optional[int] = None,
                              save_dir: str = "./experience_base") -> Dict:
        """
        构建经验库（Self-Belief优化）
        
        Args:
            validation_samples: {true_category: [image_paths]} 验证样本
            max_iterations: 最大迭代次数
            top_k: 检索top-k结果
            max_reflections_per_iter: 每次迭代最多反思的样本数
            min_improvement: 最小性能提升阈值
            max_samples_per_category: 每个类别最多处理的样本数
            save_dir: 保存目录
            
        Returns:
            Dict: 构建结果
        """
        # 初始化经验库构建器
        exp_builder = self.initialize_experience_base()
        
        # 构建经验库
        result = exp_builder.build_experience_base(
            validation_samples=validation_samples,
            max_iterations=max_iterations,
            top_k=top_k,
            max_reflections_per_iter=max_reflections_per_iter,
            min_improvement=min_improvement,
            max_samples_per_category=max_samples_per_category
        )
        
        # 保存经验库
        exp_builder.save_experience_base(save_dir)
        
        return result
    
    def load_experience_base(self, load_dir: str = "./experience_base"):
        """
        加载经验库
        
        Args:
            load_dir: 经验库加载目录
        """
        # 初始化经验库构建器
        exp_builder = self.initialize_experience_base()
        
        # 加载经验库
        exp_builder.load_experience_base(load_dir)
        
        # 将Self-Belief传递给慢思考模块（如果需要）
        if hasattr(self.slow_thinking, 'set_experience_base'):
            self.slow_thinking.set_experience_base(exp_builder)
        
        print("经验库加载完成!")   
        
    def classify_single_image(self, query_image_path: str, 
                            use_slow_thinking: bool = None,
                            top_k: int = 5) -> Dict:
        """
        对单张图像进行分类
        
        Args:
            query_image_path: 查询图像路径
            use_slow_thinking: 是否强制使用慢思考（None表示自动判断）
            top_k: 检索top-k结果
            
        Returns:
            Dict: 分类结果
        """
        print(f"开始分类图像: {query_image_path}")
        start_time = time.time()
        
        # 1. 快思考
        print("执行快思考...")
        fast_result = self.fast_thinking.fast_thinking_pipeline(query_image_path, top_k)
        fast_time = time.time() - start_time
        
        result = {
            "query_image": query_image_path,
            "fast_result": fast_result,
            "fast_time": fast_time,
            "total_time": fast_time
        }
        
        # 2. 判断是否需要慢思考
        need_slow_thinking = use_slow_thinking if use_slow_thinking is not None else fast_result["need_slow_thinking"]
        
        if need_slow_thinking:
            print("快思考结果不确定，执行慢思考...")
            slow_start_time = time.time()
            
            # 3. 慢思考
            slow_result = self.slow_thinking.slow_thinking_pipeline_optimized(
                query_image_path, fast_result, top_k
            )
            # slow_result = self.slow_thinking.slow_thinking_pipeline_update(
            #     query_image_path, fast_result, top_k
            # )
            slow_time = time.time() - slow_start_time
            
            # 4. 改进的最终决策：智能融合快慢思考结果
            fast_pred = fast_result.get("fused_top1", fast_result.get("predicted_category", "unknown"))
            slow_pred = slow_result["predicted_category"]
            fast_conf = fast_result.get("fused_top1_prob", fast_result.get("confidence", 0.0))
            slow_conf = slow_result.get("confidence", 0.0)
            
            # 检查结果一致性
            fast_slow_consistent = is_similar(fast_pred, slow_pred, threshold=0.5)
            
            # 改进的融合策略：
            # 1. 如果结果一致，优先使用慢思考结果（通常更准确）
            # 2. 如果结果不一致，根据置信度和LCB值进行决策
            # 3. 如果慢思考置信度很高，优先使用慢思考结果
            # 4. 如果快思考置信度很高且LCB值很高，可以考虑使用快思考结果
            if fast_slow_consistent:
                # 结果一致，使用慢思考结果（更详细的分析）
                final_prediction = slow_pred
                final_confidence = slow_conf
                final_reasoning = slow_result["reasoning"]
            else:
                # 结果不一致，进行智能决策
                lcb_map = fast_result.get('lcb_map', {}) or {}
                lcb_value = float(lcb_map.get(slow_pred, lcb_map.get(fast_pred, 0.5))) if isinstance(lcb_map, dict) else 0.5
                
                # 决策规则：
                # 1. 如果慢思考置信度很高(>=0.80)，优先使用慢思考
                # 2. 如果快思考置信度很高(>=0.75)且LCB值很高(>=0.70)，使用快思考
                # 3. 如果慢思考置信度中等(>=0.70)且快思考置信度较低(<0.70)，使用慢思考
                # 4. 其他情况，使用慢思考（更详细的分析）
                if slow_conf >= 0.80:
                    # 慢思考置信度很高，优先使用
                    final_prediction = slow_pred
                    final_confidence = slow_conf
                    final_reasoning = slow_result["reasoning"] + " | Fast prediction: " + fast_pred
                elif fast_conf >= 0.75 and lcb_value >= 0.70:
                    # 快思考置信度很高且LCB值很高，使用快思考
                    final_prediction = fast_pred
                    final_confidence = fast_conf
                    final_reasoning = f"Fast thinking with high confidence (LCB: {lcb_value:.3f}) | Slow prediction: {slow_pred}"
                elif slow_conf >= 0.70 and fast_conf < 0.70:
                    # 慢思考置信度中等且快思考置信度较低，使用慢思考
                    final_prediction = slow_pred
                    final_confidence = slow_conf
                    final_reasoning = slow_result["reasoning"] + " | Fast prediction: " + fast_pred
                else:
                    # 其他情况，使用慢思考（更详细的分析）
                    print("快慢思考结果不一致，进行最终融合决策...")
                    final_prediction, final_confidence, final_reasoning = self._final_decision(
                        query_image_path, fast_result, slow_result, top_k
                    )
            
            result.update({
                "slow_result": slow_result,
                "slow_time": slow_time,
                "total_time": fast_time + slow_time,
                "final_prediction": final_prediction,
                "final_confidence": final_confidence,
                "final_reasoning": final_reasoning,
                "used_slow_thinking": True,
                "fast_slow_consistent": is_similar(fast_pred, slow_pred, threshold=0.5)
            })
        else:
            print("快思考结果确定，直接返回...")
            final_prediction = fast_result["predicted_category"]
            final_confidence = fast_result["confidence"]
            
            result.update({
                "final_prediction": final_prediction,
                "final_confidence": final_confidence,
                "final_reasoning": "Fast thinking result",
                "used_slow_thinking": False
            })
        
        total_time = time.time() - start_time
        result["total_time"] = total_time
        
        print(f"分类完成: {final_prediction} (置信度: {final_confidence:.4f})")
        print(f"总耗时: {total_time:.2f}秒")
        
        return result
    
    def classify_batch_images(self, query_image_paths: List[str],
                            use_slow_thinking: bool = None,
                            top_k: int = 5) -> List[Dict]:
        """
        批量分类图像
        
        Args:
            query_image_paths: 查询图像路径列表
            use_slow_thinking: 是否强制使用慢思考
            top_k: 检索top-k结果
            
        Returns:
            List[Dict]: 每个图像的分类结果
        """
        print(f"开始批量分类 {len(query_image_paths)} 张图像...")
        results = []
        
        for i, img_path in enumerate(tqdm(query_image_paths, desc="分类进度")):
            try:
                result = self.classify_single_image(img_path, use_slow_thinking, top_k)
                results.append(result)
            except Exception as e:
                print(f"分类失败 {img_path}: {e}")
                results.append({
                    "query_image": img_path,
                    "final_prediction": "error",
                    "final_confidence": 0.0,
                    "error": str(e)
                })
        
        print("批量分类完成!")
        return results
    
    def evaluate_on_dataset(self, test_samples: Dict[str, List[str]],
                          use_slow_thinking: bool = None,
                          top_k: int = 5) -> Dict:
        """
        在测试数据集上评估系统性能
        
        Args:
            test_samples: {true_category: [image_paths]} 测试样本
            use_slow_thinking: 是否强制使用慢思考
            top_k: 检索top-k结果
            
        Returns:
            Dict: 评估结果
        """
        print("开始数据集评估...")
        
        all_results = []
        correct_count = 0
        total_count = 0
        fast_thinking_count = 0
        slow_thinking_count = 0
        
        for true_category, image_paths in test_samples.items():
            print(f"评估类别: {true_category} ({len(image_paths)} 张图像)")
            
            for img_path in image_paths:
                try:
                    result = self.classify_single_image(img_path, use_slow_thinking, top_k)
                    
                    # 检查预测是否正确
                    predicted = result["final_prediction"]
                    is_correct = is_similar(predicted, true_category, threshold=0.5)
                    
                    if is_correct:
                        correct_count += 1
                    
                    total_count += 1
                    
                    if result.get("used_slow_thinking", False):
                        slow_thinking_count += 1
                    else:
                        fast_thinking_count += 1
                    
                    result["true_category"] = true_category
                    result["is_correct"] = is_correct
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"评估失败 {img_path}: {e}")
                    total_count += 1
                    all_results.append({
                        "query_image": img_path,
                        "true_category": true_category,
                        "final_prediction": "error",
                        "is_correct": False,
                        "error": str(e)
                    })
        
        # 计算指标
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        fast_thinking_ratio = fast_thinking_count / total_count if total_count > 0 else 0.0
        slow_thinking_ratio = slow_thinking_count / total_count if total_count > 0 else 0.0
        
        # 计算平均时间
        total_times = [r.get("total_time", 0) for r in all_results if "total_time" in r]
        avg_time = np.mean(total_times) if total_times else 0.0
        
        # 计算平均置信度
        confidences = [r.get("final_confidence", 0) for r in all_results if "final_confidence" in r]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        evaluation_result = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "fast_thinking_count": fast_thinking_count,
            "slow_thinking_count": slow_thinking_count,
            "fast_thinking_ratio": fast_thinking_ratio,
            "slow_thinking_ratio": slow_thinking_ratio,
            "avg_time": avg_time,
            "avg_confidence": avg_confidence,
            "detailed_results": all_results
        }
        
        print("评估完成!")
        print(f"准确率: {accuracy:.4f}")
        print(f"快思考比例: {fast_thinking_ratio:.4f}")
        print(f"慢思考比例: {slow_thinking_ratio:.4f}")
        print(f"平均时间: {avg_time:.2f}秒")
        print(f"平均置信度: {avg_confidence:.4f}")
        
        return evaluation_result
    
    def save_results(self, results: List[Dict], save_path: str):
        """
        保存结果到文件
        
        Args:
            results: 结果列表
            save_path: 保存路径
        """
        # 清理结果，移除不能序列化的对象
        cleaned_results = []
        for result in results:
            cleaned_result = {}
            for key, value in result.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    cleaned_result[key] = value
                else:
                    cleaned_result[key] = str(value)
            cleaned_results.append(cleaned_result)
        
        dump_json(save_path, cleaned_results)
        print(f"结果已保存到: {save_path}")
    
    def get_system_stats(self) -> Dict:
        """
        获取系统统计信息
        
        Returns:
            Dict: 系统统计信息
        """
        stats = {
            "image_kb_size": len(self.kb_builder.image_knowledge_base),
            "text_kb_size": len(self.kb_builder.text_knowledge_base),
            "device": self.device,
            "model_tag": self.mllm_bot.model_tag if hasattr(self.mllm_bot, 'model_tag') else "unknown",
            "confidence_threshold": self.fast_thinking.confidence_threshold,
            "similarity_threshold": self.fast_thinking.similarity_threshold
        }
        return stats
    
    def _final_decision(self, query_image_path: str, fast_result: Dict, slow_result: Dict, top_k: int = 5) -> Tuple[str, float, str]:
        """
        最终决策：当快慢思考结果不一致时，让MLLM基于候选类别进行最终选择
        
        Args:
            query_image_path: 查询图像路径
            fast_result: 快思考结果
            slow_result: 慢思考结果
            top_k: 候选类别数量
            
        Returns:
            Tuple[str, float, str]: (最终预测类别, 置信度, 推理过程)
        """
        from PIL import Image
        import json
        import re
        
        image = Image.open(query_image_path).convert("RGB")
        
        # 获取快思考的候选类别（融合结果）
        fast_candidates = fast_result.get("fused_results", [])[:top_k]
        fast_candidates_text = ""
        for i, (category, score) in enumerate(fast_candidates):
            fast_candidates_text += f"{i+1}. {category} (fast similarity: {score:.4f})\n"
        
        # 获取慢思考的候选类别（增强检索结果）
        slow_candidates = slow_result.get("enhanced_results", [])[:top_k]
        slow_candidates_text = ""
        for i, (category, score) in enumerate(slow_candidates):
            slow_candidates_text += f"{i+1}. {category} (slow similarity: {score:.4f})\n"
        
        # 获取慢思考的结构化描述
        structured_description = slow_result.get("structured_description", "")
        
        # 构建最终决策提示
        prompt = f"""You are an expert in fine-grained visual recognition. I need you to make a final decision between two different analysis approaches.

Fast Thinking Analysis (CLIP-based retrieval):
{fast_candidates_text}

Slow Thinking Analysis (MLLM + detailed analysis):
{slow_candidates_text}

Detailed visual analysis from slow thinking:
{structured_description}

Please carefully analyze the image and consider:
1. The visual characteristics described in the detailed analysis
2. The similarity scores from both approaches
3. The consistency between fast and slow thinking results
4. Which approach provides more reliable evidence for classification

Make your final prediction in JSON format:
{{
    "predicted_category": "exact category name",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of your decision process",
    "chosen_approach": "fast" or "slow" or "hybrid",
    "key_evidence": "main visual evidence supporting your decision"
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
                    chosen_approach = result.get("chosen_approach", "hybrid")
                    key_evidence = result.get("key_evidence", "No evidence provided")
                    
                    # 增强推理信息
                    enhanced_reasoning = f"Final Decision - {chosen_approach.upper()}: {reasoning} | Key Evidence: {key_evidence}"
                    
                else:
                    # 如果无法解析JSON，使用慢思考结果作为fallback
                    predicted_category = slow_result["predicted_category"]
                    confidence = slow_result["confidence"]
                    reasoning = "JSON parsing failed, using slow thinking result as fallback"
                    enhanced_reasoning = reasoning
                    
            except (json.JSONDecodeError, ValueError):
                predicted_category = slow_result["predicted_category"]
                confidence = slow_result["confidence"]
                reasoning = "JSON parsing failed, using slow thinking result as fallback"
                enhanced_reasoning = reasoning
            
            return predicted_category, confidence, enhanced_reasoning
            
        except Exception as e:
            print(f"最终决策失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 使用慢思考结果作为fallback，带默认值
            predicted_category = slow_result.get("predicted_category", slow_result.get("category", "unknown"))
            confidence = slow_result.get("confidence", slow_result.get("similarity", 0.0))
            
            # 如果仍然没有有效值，使用快思考结果
            if not predicted_category or predicted_category == "unknown":
                if fast_result and "predicted_category" in fast_result:
                    predicted_category = fast_result["predicted_category"]
                    confidence = fast_result.get("confidence", 0.0)
                else:
                    # 从融合结果中获取
                    fused_results = fast_result.get("fused_results", [])
                    if fused_results:
                        predicted_category = fused_results[0][0]
                        confidence = fused_results[0][1]
                    else:
                        predicted_category = "unknown"
                        confidence = 0.0
            
            return predicted_category, confidence, f"Final decision failed: {str(e)}"


# 示例使用
if __name__ == "__main__":
    # 初始化系统
    system = FastSlowThinkingSystem(
        model_tag="Qwen2.5-VL-7B",
        model_name="Qwen2.5-VL-7B",
        device="cuda"
    )
    
    # 构建知识库
    train_samples = {
        "Chihuahua": ["path/to/chihuahua1.jpg", "path/to/chihuahua2.jpg"],
        "Shiba Inu": ["path/to/shiba1.jpg", "path/to/shiba2.jpg"]
    }
    
    # 构建知识库
    image_kb, text_kb = system.build_knowledge_base(train_samples)
    
    # 测试单张图像
    query_image = "path/to/test_image.jpg"
    result = system.classify_single_image(query_image)
    print("分类结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 测试批量图像
    test_images = ["path/to/test1.jpg", "path/to/test2.jpg"]
    batch_results = system.classify_batch_images(test_images)
    
    # 保存结果
    system.save_results(batch_results, "classification_results.json")
    
    print("系统测试完成!")
