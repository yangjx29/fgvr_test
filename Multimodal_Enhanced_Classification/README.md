# Multimodal Enhanced Classification: 多模态增强分类框架

本文面向 `Multimodal_Enhanced_Classification` 目录，系统性概括多模态增强分类框架的推理流程与算法实现，便于快速理解与复现。

## 背景与目标

**核心思想**：实现**待测试[图-文]与检索到的[图-文]进行匹配**的分类策略。

该框架使用CLIP提取图像和文本的多模态特征，通过特征拼接、基于熵的权重计算和加权相似度匹配来实现精准的细粒度视觉识别。与传统的图文匹配不同，本方法直接对比两个多模态样本（待测试样本和检索样本），每个样本都包含图像和对应的文本描述。

### 核心改进（基于AWT思想）
- **A: Augmentation** 对两组图像分别进行多视图增强，提升稳健性；
- **W: Weighting** 以不确定性（熵）自适应估计多模态特征的权重；
- **T: Transportation** 将两组多模态特征进行加权相似度计算，实现多模态匹配。

---

## 数据与使用方式概览

### 数据要求
- 两组图像数据集：
  - `{dataset}_retrieved`：检索到的图像（作为参考库）
  - `{dataset}_test`：待测试的图像（需要分类的图像）
- 对应的文本描述文件：
  - `./descriptions/{dataset}_retrieved_descriptions.json`：检索图像的描述
  - `./descriptions/{dataset}_test_descriptions.json`：测试图像的描述

### 使用流程
```bash
# 1. 预提取多模态特征（同时处理检索和测试图像）
python pre_extract.py [data_path] --test_set [dataset_name]

# 2. 运行多模态增强分类评估
python evaluate.py --test_set [dataset_name]
```

---

## 流水线细节（与实现文件对应）

### 1) 预提取多模态特征（pre_extract.py）

- 使用 CLIP 对两组图像分别生成 **n_views** 个增强视图特征，并结合对应的文本描述：
  - 基础变换：Resize→CenterCrop；
  - 归一化：与 CLIP 预处理一致；
  - 增强器 `Augmenter(base_transform, preprocess, n_views)` 产出多视图；
  - 对每张图像编码其对应的文本描述；
  - 将文本特征和图像特征拼接：`[Ti, Ii]`（检索）和 `[T'j, I'j]`（测试）；
  - 编码后对每个向量做 L2 归一化；
  - 分别保存到 `./pre_extracted_feat/{arch}/seed{seed}/{dataset}_retrieved.pth` 和 `{dataset}_test.pth`。

形状约定：
- 单样本多模态特征 `multimodal_features ∈ R^{n_views × 2d}`，已 L2-Norm。

### 2) 多模态增强分类评估（evaluate.py）

- 加载两组预提取的多模态特征：
  - `retrieved_data`: 检索图像的[图-文]特征 `[(multimodal_features, target), ...]`
  - `test_data`: 待测试图像的[图-文]特征 `[(multimodal_features, target), ...]`
- 对每个待测试样本，计算与所有检索样本的多模态相似度。

### 3) 权重计算与多模态相似度匹配

- 使用基于熵的权重计算：
  - 对每组多模态特征计算熵：`entropy = calculate_batch_entropy(features)`
  - 权重计算：`weights = F.softmax(-entropy / temperature, dim=0)`
  - 加权平均：`weighted_features = (features * weights.unsqueeze(-1)).sum(dim=0)`
- 计算加权相似度：
  - L2归一化：`weighted_features = weighted_features / weighted_features.norm()`
  - 余弦相似度：`similarity = logit_scale.exp() * torch.dot(weighted_test, weighted_retrieved)`

---

## Weighting：多模态特征权重估计

在多模态增强分类中，我们对两组多模态特征分别计算权重，熵定义为 \( H(p) = -\sum_i p_i \log p_i \)。

### 多模态特征权重计算

- **待测试样本权重**：
  - 计算每个视图的熵：`entropy = calculate_batch_entropy(test_features)`
  - `test_weights = F.softmax(-entropy / temperature, dim=0)`，熵越低权重越大。

- **检索样本权重**：
  - 计算每个视图的熵：`entropy = calculate_batch_entropy(retrieved_features)`
  - `retrieved_weights = F.softmax(-entropy / temperature, dim=0)`，熵越低权重越大。

- **加权平均**：
  - `weighted_test = (test_features * test_weights.unsqueeze(-1)).sum(dim=0)`
  - `weighted_retrieved = (retrieved_features * retrieved_weights.unsqueeze(-1)).sum(dim=0)`

超参：`temperature ∈ (0, +∞)` 控制 softmax 平滑度，默认为 0.5。

---

## Transportation：加权多模态相似度计算

在多模态增强分类中，我们使用加权相似度计算来实现[图-文]特征匹配。

### 加权多模态相似度匹配流程

1. **多模态特征加权**：
   - 对两组多模态特征分别计算权重并加权平均
   - `weighted_test = (test_features * test_weights.unsqueeze(-1)).sum(dim=0)`
   - `weighted_retrieved = (retrieved_features * retrieved_weights.unsqueeze(-1)).sum(dim=0)`

2. **L2归一化**：
   - `weighted_test = weighted_test / weighted_test.norm()`
   - `weighted_retrieved = weighted_retrieved / weighted_retrieved.norm()`

3. **多模态相似度计算**：
   - `similarity = logit_scale.exp() * torch.dot(weighted_test, weighted_retrieved)`

4. **匹配决策**：
   - 找到相似度最高的检索样本：`max_idx = torch.argmax(similarities)`
   - 使用该样本的标签作为预测：`predicted_label = retrieved_labels[max_idx]`

---

## 多模态增强分类算法特点

### 核心改进

1. **[图-文]对[图-文]匹配**：
   - 实现待测试图像文本对与检索图像文本对的直接匹配
   - 每个样本都包含图像和对应的文本描述信息

2. **多模态特征融合**：
   - 将图像特征和对应文本描述特征拼接：`[Ti, Ii]` 和 `[T'j, I'j]`
   - 特征维度从 `d` 扩展到 `2d`，包含更丰富的语义信息

3. **双向权重计算**：
   - 对两组多模态特征分别计算基于熵的权重
   - 权重反映每个视图的不确定性，熵越低权重越高

4. **端到端匹配**：
   - 直接在多模态特征空间进行相似度计算
   - 无需预定义类别模板，更加灵活

5. **检索增强分类**：
   - 利用检索到的相关样本进行分类决策
   - 提高分类准确性和鲁棒性

---

## 关键超参与形状速查

### 多模态特征形状
- `image_features`: `(n_views, d)` - 多视图图像特征
- `text_features`: `(1, d)` - 单个文本描述特征
- `multimodal_features`: `(n_views, 2*d)` - 拼接后的[图-文]特征
- `test_weights`: `(n_views,)` - 待测试样本的视图权重
- `retrieved_weights`: `(n_views,)` - 检索样本的视图权重
- `weighted_test`: `(2*d,)` - 加权后的待测试[图-文]特征
- `weighted_retrieved`: `(2*d,)` - 加权后的检索[图-文]特征

### 关键超参数
- `n_views`：每图像的增强视图数量（默认 50）
- `temperature`：权重 softmax 温度（默认 0.5）
- `resolution`：输入分辨率（默认 224）

---

## 复现实用提醒

1. **数据准备**：确保检索图像集、测试图像集和对应的描述JSON文件准备完整
2. **特征提取**：先运行 `pre_extract.py` 提取并保存多模态[图-文]特征
3. **分类评估**：再运行 `evaluate.py` 进行多模态增强分类评估
4. **参数调整**：可以调整温度参数（默认0.5）来控制权重分布的平滑度
5. **GPU内存**：注意多视图多模态特征可能占用较多GPU内存，可适当调整batch_size
6. **描述质量**：文本描述的质量直接影响多模态特征的表达能力和最终分类效果

### 数据文件结构
```
descriptions/
├── {dataset_name}_retrieved_descriptions.json  # 检索图像的文本描述
└── {dataset_name}_test_descriptions.json       # 测试图像的文本描述

data/
├── {dataset_name}_retrieved/                   # 检索图像数据集
└── {dataset_name}_test/                        # 测试图像数据集
```

---

## 引用

若本实现或文档对您的研究有帮助，欢迎引用论文：

Zhu, Yuhan; Ji, Yuyang; Zhao, Zhiyu; Wu, Gangshan; Wang, Limin. AWT: Transferring Vision-Language Models via Augmentation, Weighting, and Transportation. NeurIPS 2024.