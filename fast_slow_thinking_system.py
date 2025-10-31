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

from agents.mllm_bot import MLLMBot
from knowledge_base_builder import KnowledgeBaseBuilder
from fast_thinking import FastThinking
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
                 enable_mllm_intermediate_judge: bool = True):
        """
        初始化快慢思考系统
        
        Args:
            model_tag: MLLM模型标签
            model_name: MLLM模型名称
            image_encoder_name: 图像编码器名称
            text_encoder_name: 文本编码器名称
            device: 设备
            cfg: 配置参数
            enable_mllm_intermediate_judge: 是否启用MLLM中间判断（消融实验开关）
        """
        self.device = device
        self.cfg = cfg or {}
        self.enable_mllm_intermediate_judge = enable_mllm_intermediate_judge
        
        # 先初始化知识库构建器（仅CLIP，显存占用小）
        print("初始化知识库构建器...")
        self.kb_builder = KnowledgeBaseBuilder(
            image_encoder_name=image_encoder_name,
            text_encoder_name=text_encoder_name,
            device=device,
            cfg=cfg
        )
        
        # 使用MLLM单例，避免重复加载（显存优化）
        print("获取MLLM模型实例（单例模式）...")
        from utils.mllm_singleton import get_mllm_bot
        self.mllm_bot = get_mllm_bot(
            model_tag=model_tag,
            device=device
        )
        
        # 初始化快思考模块
        print("初始化快思考模块...")
        # 设置一个默认的stats文件路径，后续会在load_knowledge_base中更新
        default_stats_file = "./experiments/temp/stats.json"
        self.fast_thinking = FastThinking(
            knowledge_base_builder=self.kb_builder,
            confidence_threshold=self.cfg.get('confidence_threshold', 0.8),
            similarity_threshold=self.cfg.get('similarity_threshold', 0.7),
            stats_file=default_stats_file  # 使用默认路径，避免None错误
        )
        
        # 初始化慢思考模块
        print("初始化慢思考模块...")
        self.slow_thinking = SlowThinking(
            mllm_bot=self.mllm_bot,
            knowledge_base_builder=self.kb_builder,
            fast_thinking=self.fast_thinking,
            knowledge_base_dir=getattr(self, 'knowledge_base_dir', None)
        )
        
        print("快慢思考系统初始化完成!")
    
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
        
        # 初始化返回值
        image_kb = None
        text_kb = None
        
        try:
            # 构建知识库
            image_kb, text_kb = self.kb_builder.build_knowledge_base(
                self.mllm_bot, train_samples, augmentation
            )
            
            # 保存知识库
            self.kb_builder.save_knowledge_base(save_dir)
            
        except Exception as e:
            print(f"构建知识库时出错: {e}")
            # 如果构建失败，尝试从已有的知识库获取
            if hasattr(self.kb_builder, 'image_knowledge_base') and hasattr(self.kb_builder, 'text_knowledge_base'):
                image_kb = self.kb_builder.image_knowledge_base
                text_kb = self.kb_builder.text_knowledge_base
                print("使用已加载的知识库")
            else:
                raise e
        
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
        self.knowledge_base_dir = load_dir  # 存储知识库路径
        self.kb_builder.load_knowledge_base(load_dir)
        
        # 设置快思考模块的stats文件路径
        import os
        stats_file = os.path.join(load_dir, "stats.json")
        if hasattr(self, 'fast_thinking') and self.fast_thinking:
            self.fast_thinking.stats_file = stats_file
        
        # 更新慢思考模块的知识库路径
        if hasattr(self, 'slow_thinking'):
            self.slow_thinking.knowledge_base_dir = load_dir
        
        print("知识库加载完成!")   
    
    def mllm_intermediate_judge(self, query_image_path: str, fast_result: Dict, top_k: int = 5) -> Tuple[bool, str, float]:
        """
        MLLM中间判断：分析快思考的top-k结果，判断是否需要进入慢思考
        
        Args:
            query_image_path: 查询图像路径
            fast_result: 快思考结果
            top_k: 要分析的top-k结果数量
            
        Returns:
            Tuple[bool, str, float]: (是否需要慢思考, 预测类别, 置信度)
        """
        print("执行MLLM中间判断...")
        
        try:
            # 获取快思考的融合结果
            fused_results = fast_result.get("fused_results", [])
            if not fused_results:
                # 如果没有融合结果，使用图像检索结果
                fused_results = fast_result.get("img_results", [])
            
            if len(fused_results) == 0:
                print("警告：没有可用的检索结果，跳过MLLM判断")
                return True, "unknown", 0.0
                
            # 构造候选类别列表
            candidates = []
            for i, (category, score) in enumerate(fused_results[:top_k]):
                candidates.append(f"{i+1}. {category} (置信度: {score:.4f})")
            
            candidates_text = "\n".join(candidates)
            
            # 构造MLLM判断提示
            prompt = f"""你是一个专业的图像分类专家。请分析这张图像和以下候选分类结果，判断你是否有足够的信心做出最终分类决定。

候选类别（按相似度排序）：
{candidates_text}

请仔细观察图像特征，分析以上候选结果，然后回答：

1. 你是否有足够信心确定这张图像的类别？
2. 如果有信心，你认为最可能的类别是什么？
3. 你的信心水平如何？（0-1之间的数值）

请按以下格式回答：
决策：[有信心/没信心]
预测类别：[类别名称]
置信度：[0-1的数值]
推理：[简要说明你的判断理由]
"""
            
            # 调用MLLM进行判断
            response = self.mllm_bot.text_image_response_multimodal(query_image_path, prompt)
            
            print(f"MLLM中间判断响应: {response}")
            
            # 解析MLLM回复
            need_slow_thinking, predicted_category, confidence = self._parse_mllm_judge_response(response, fused_results)
            
            return need_slow_thinking, predicted_category, confidence
            
        except Exception as e:
            print(f"MLLM中间判断出错: {e}")
            # 出错时默认进入慢思考
            return True, "error", 0.0
    
    def _parse_mllm_judge_response(self, response: str, fused_results: List[Tuple[str, float]]) -> Tuple[bool, str, float]:
        """
        解析MLLM判断响应
        
        Args:
            response: MLLM的回复文本
            fused_results: 融合检索结果，用于fallback
            
        Returns:
            Tuple[bool, str, float]: (是否需要慢思考, 预测类别, 置信度)
        """
        try:
            # 提取决策
            decision_confident = False
            predicted_category = fused_results[0][0] if fused_results else "unknown"  # 默认值
            confidence = 0.5  # 默认置信度
            
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('决策：') or line.startswith('决策:'):
                    decision = line.split('：')[1] if '：' in line else line.split(':')[1]
                    decision_confident = '有信心' in decision
                    
                elif line.startswith('预测类别：') or line.startswith('预测类别:'):
                    predicted_category = line.split('：')[1] if '：' in line else line.split(':')[1]
                    predicted_category = predicted_category.strip()
                    
                elif line.startswith('置信度：') or line.startswith('置信度:'):
                    conf_str = line.split('：')[1] if '：' in line else line.split(':')[1]
                    try:
                        confidence = float(conf_str.strip())
                    except:
                        confidence = 0.5
            
            # 如果MLLM有信心且置信度较高，则不需要慢思考
            need_slow_thinking = not (decision_confident and confidence >= 0.6)
            
            print(f"MLLM判断解析结果 - 需要慢思考: {need_slow_thinking}, 类别: {predicted_category}, 置信度: {confidence}")
            
            return need_slow_thinking, predicted_category, confidence
            
        except Exception as e:
            print(f"解析MLLM响应时出错: {e}")
            # 出错时默认进入慢思考
            return True, fused_results[0][0] if fused_results else "unknown", 0.0
        
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
        if use_slow_thinking is not None:
            # 强制指定是否使用慢思考
            need_slow_thinking = use_slow_thinking
            mllm_judge_result = None
        elif self.enable_mllm_intermediate_judge:
            # 启用MLLM中间判断
            print("启用MLLM中间判断模式...")
            mllm_need_slow, mllm_predicted, mllm_confidence = self.mllm_intermediate_judge(query_image_path, fast_result, top_k)
            need_slow_thinking = mllm_need_slow
            mllm_judge_result = {
                "predicted_category": mllm_predicted,
                "confidence": mllm_confidence,
                "need_slow_thinking": mllm_need_slow
            }
        else:
            # 使用传统的快思考触发机制
            need_slow_thinking = fast_result["need_slow_thinking"]
            mllm_judge_result = None
        
        if need_slow_thinking:
            print("快思考结果不确定，执行慢思考...")
            slow_start_time = time.time()
            
            # 3. 慢思考
            # slow_result = self.slow_thinking.slow_thinking_pipeline(
            #     query_image_path, fast_result, top_k
            # )
            slow_result = self.slow_thinking.slow_thinking_pipeline_update(
                query_image_path, fast_result, top_k
            )
            slow_time = time.time() - slow_start_time
            
            # 4. 最终决策：如果快慢思考结果不一致，进行融合决策
            fast_pred = fast_result.get("fused_top1", fast_result.get("predicted_category", "unknown"))
            slow_pred = slow_result["predicted_category"]
            
            if fast_pred != slow_pred and not is_similar(fast_pred, slow_pred, threshold=0.5):
                print("快慢思考结果不一致，进行最终融合决策...")
                final_prediction, final_confidence, final_reasoning = self._final_decision(
                    query_image_path, fast_result, slow_result, top_k
                )
            else:
                # 结果一致，使用慢思考结果
                final_prediction = slow_pred
                final_confidence = slow_result["confidence"]
                final_reasoning = slow_result["reasoning"]
            
            result.update({
                "slow_result": slow_result,
                "slow_time": slow_time,
                "total_time": fast_time + slow_time,
                "final_prediction": final_prediction,
                "final_confidence": final_confidence,
                "final_reasoning": final_reasoning,
                "used_slow_thinking": True,
                "fast_slow_consistent": is_similar(fast_pred, slow_pred, threshold=0.5),
                "mllm_judge_result": mllm_judge_result
            })
        else:
            if mllm_judge_result is not None and not mllm_judge_result["need_slow_thinking"]:
                print("MLLM中间判断有信心，直接返回...")
                final_prediction = mllm_judge_result["predicted_category"]
                final_confidence = mllm_judge_result["confidence"]
                final_reasoning = "MLLM intermediate judge result"
            else:
                print("快思考结果确定，直接返回...")
                final_prediction = fast_result["predicted_category"]
                final_confidence = fast_result["confidence"]
                final_reasoning = "Fast thinking result"
            
            result.update({
                "final_prediction": final_prediction,
                "final_confidence": final_confidence,
                "final_reasoning": final_reasoning,
                "used_slow_thinking": False,
                "mllm_judge_result": mllm_judge_result
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
            # 使用慢思考结果作为fallback
            return slow_result["predicted_category"], slow_result["confidence"], f"Final decision failed: {str(e)}"


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
