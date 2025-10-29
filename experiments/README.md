## experiments 目录数据说明

本说明聚焦于 `experiments` 下与快慢思考分类直接相关的两个子目录：
- `infer/` 推理阶段的中间结果
- `knowledge_base/` 知识库（类别级的特征与辅助信息）

其他子目录此处不作描述。

### 目录结构示例

```text
experiments/
  dog120/
    infer/                # 推理阶段保存的每图JSON（按图片粒度）
      <safe_cat>_<img>.json
      ...
    knowledge_base/       # 知识库（按类别粒度）
      image_knowledge_base.json
      text_knowledge_base.json
      category_descriptions.json
      stats.json
```

## infer/ 推理阶段保存内容

推理阶段按“每张图片”保存一份 JSON，文件名形如：
- `<真实类别>_<图片基名>.json`（类别与文件名中的空格会替换为下划线）

常见字段如下（不同版本会有少量差异，以下为核心与推荐字段）：

```json
{
  "query_image": "./datasets/.../xxx.jpg",               
  "true_category": "Pomeranian",                         

  "fast_result": {                                         
    "predicted_category": "Pomeranian",                  
    "confidence": 0.62,                                    
    "need_slow_thinking": true,                            
    "img_results": [["Pomeranian", 0.83], ["Keeshond", 0.78], ...],
    "text_results": [["Pomeranian", 0.85], ["Japanese Spaniel", 0.80], ...],
    "fused_results": [["Pomeranian", 0.2254], ["Keeshond", 0.2186], ...],
    "fused_top1": "Pomeranian",
    "fused_top1_prob": 0.61,
    "fused_margin": 0.12,
    "topk_overlap": true
  },

  "need_slow_thinking": true,                              

  "slow_result": {                                         
    "predicted_category": "Pomeranian",
    "confidence": 0.78,
    "structured_description": "... 结构化描述文本 ...",
    "enhanced_results": [["Pomeranian", 0.91], ["Chihuahua", 0.74], ...]
  },

  "fast_top_k": [["Pomeranian", 0.83], ["Keeshond", 0.78], ...],
  "fast_fused_results": [["Pomeranian", 0.2254], ["Keeshond", 0.2186], ...],
  "slow_top_k": [["Pomeranian", 0.91], ["Chihuahua", 0.74], ...],

  "timestamp": 1730090000.123
}
```

- "fast_result": 快思考阶段的检索与融合输出，包含图→图、图→文 Top‑K，融合Top‑K、Top‑1与置信度、边际、Top‑K重叠等触发相关量。
- "need_slow_thinking": 触发标志，指示是否进入慢思考。
- "slow_result": 慢思考阶段的结构化描述、增强检索Top‑K与最终预测；仅在触发时存在。
- "fast_top_k"/"fast_fused_results"/"slow_top_k": 为分类/裁决与后续消融准备的候选与融合结果快照。

说明：推理阶段默认不单独落盘“查询图像的特征向量”。若需要，可在推理阶段扩展将查询特征另存为 `.npy` 至 `experiments/<dataset><num>/infer/features/`。

## knowledge_base/ 知识库存储内容

知识库为“类别级别”的静态数据，供检索与触发判断使用：

### 1) image_knowledge_base.json
- 结构：`{ "<CategoryName>": [f1, f2, ..., fD], ... }`
- 含义：每个类别的图像特征向量（通常为CLIP图像编码聚合后得到，D≈512）。

### 2) text_knowledge_base.json
- 结构：`{ "<CategoryName>": [f1, f2, ..., fD], ... }`
- 含义：每个类别的文本特征向量（来自类别描述或模板文本经CLIP文本编码）。

### 3) category_descriptions.json
- 结构：`{ "<CategoryName>": "类别描述文本...", ... }`
- 含义：面向检索/解释的类别描述汇总（可能来自CDV-Captioner或人工整理）。

### 4) stats.json
- 结构示例：
```json
{
  "Pomeranian": {"n": 667, "m": 420},
  "Chihuahua": {"n": 650, "m": 388}
}
```
- 含义：历史预测统计（例如 n=预测次数，m=正确次数），用于动态触发（LCB/UCB）等策略的先验与自适应阈值计算。

## 快速对照
- 推理阶段（`infer/`）：按“图片粒度”保存中间信息，便于后续分类策略与裁决的消融实验，无需重复推理。
- 知识库（`knowledge_base/`）：按“类别粒度”保存图像/文本特征与描述、统计信息，供快/慢思考检索与触发机制使用。


