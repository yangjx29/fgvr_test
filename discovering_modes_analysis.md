# discovering.py 各模式功能分析

## 模式分类

脚本支持 12 种模式，分为**快慢思考系统模式**和**传统发现模式**两大类：

## 一、快慢思考系统模式（已完整实现）

### 1. `build_knowledge_base` - 构建知识库
**功能：** 为快慢思考系统构建完整的知识检索库

**核心流程：**
- 初始化FastSlowThinkingSystem
- 加载训练样本（从data_discovery获取）
- 构建图像知识库和文本知识库
- 支持数据增强（augmentation=True）
- 保存到指定目录

**使用示例：**
```bash
CUDA_VISIBLE_DEVICES=3 python discovering.py --mode=build_knowledge_base \
  --config_file_env=./configs/env_machine.yml \
  --config_file_expt=./configs/expts/dog120_all.yml \
  --num_per_category=10 \
  --knowledge_base_dir=/path/to/knowledge_base
```

### 2. `classify` - 单张图像分类
**功能：** 使用完整的快慢思考系统对单张图像进行分类

**核心流程：**
- 加载预构建的知识库
- 执行快慢思考分类流程
- 根据置信度自动决定是否触发慢思考
- 保存分类结果和详细信息

**关键参数：**
- `--query_image`: 待分类的图像路径
- `--use_slow_thinking`: 强制使用慢思考（None表示自动）

### 3. `evaluate` - 批量评估系统
**功能：** 在测试数据集上评估快慢思考系统的整体性能

**核心流程：**
- 从测试目录构建测试样本
- 批量运行快慢思考系统
- 统计准确率、快慢思考比例等指标
- 生成详细的评估报告

**输出指标：**
- 总体准确率
- 快思考使用比例
- 慢思考使用比例

### 4. `fastonly` - 仅快思考评估
**功能：** 仅使用快思考模块进行评估，用于分析快思考性能

**核心流程：**
- 只运行FastThinking模块
- 统计快思考的预测准确性
- 分析需要慢思考的样本分布
- 提供快思考性能基线

**统计信息：**
- 正确预测中需要/不需要慢思考的分布
- 错误预测中需要/不需要慢思考的分布
- 快思考准确率基线

### 5. `slowonly` - 仅慢思考评估
**功能：** 仅使用慢思考模块进行评估（需要快思考结果作为输入）

**核心流程：**
- 先运行快思考获取初始结果
- 强制运行慢思考进行深度分析
- 统计慢思考的最终预测性能
- 分析慢思考的改进效果

**注意：** 慢思考依然需要快思考结果作为输入上下文

### 6. `fast_slow` - 完整系统评估
**功能：** 使用完整的快慢思考系统进行评估，提供最全面的性能分析

**核心流程：**
- 运行完整的快慢思考决策流程
- 自动判断是否需要触发慢思考
- 统计各种场景下的性能表现
- 提供系统级别的性能分析

**详细统计：**
- 总体准确率
- 仅快思考正确的数量和比例
- 慢思考触发次数和准确率
- 快慢思考的性能对比分析

## 二、多模态检索模式（已完整实现）

### 7. `build_gallery` - 构建多模态模板库
**功能：** 为基于检索的多模态分类方法构建类别模板库

**核心流程：**
- 初始化MLLM、CDVCaptioner、MultimodalRetrieval
- 加载K-shot训练样本
- 为每个类别生成多模态特征模板
- 支持多种特征融合方法（concat/average/weighted/cross_attention）
- 保存为JSON格式的gallery文件

**关键参数：**
- `--kshot`: 每类样本数量
- `--region_num`: 区域选择数量  
- `--superclass`: 超类名称（如dog、bird等）
- `--fusion_method`: 特征融合方法
- `--gallery_out`: 输出路径

## 三、传统发现模式（函数已定义，主流程未实现）

### 8. `identify` - 超类识别
**函数：** `main_identify()`
**设计功能：** 识别图像的超类（如car、bird、flower、dog、cat、Pokemon等）

### 9. `describe` - 属性描述生成
**函数：** `main_describe()`
**设计功能：** 
- 使用VQA模型为每个属性生成描述
- 生成LLM推理用的prompt
- 处理宠物数据集的特殊逻辑（区分狗和猫）

### 10. `guess` - 类别推测
**函数：** `main_guess()`
**设计功能：** 基于属性描述使用LLM推理出可能的类别名称

### 11. `howto` - 区分策略分析
**函数：** `how_to_distinguish()`
**设计功能：** 询问LLM如何区分不同类别

### 12. `postprocess` - 后处理
**函数：** `post_process()`
**设计功能：** 清理和整理LLM推理结果，提取类别名称

## 模式使用建议

### 典型工作流程：

1. **构建阶段：**
   ```bash
   # 构建知识库（用于快慢思考系统）
   python discovering.py --mode=build_knowledge_base
   
   # 或构建gallery（用于传统多模态检索）
   python discovering.py --mode=build_gallery
   ```

2. **测试阶段：**
   ```bash
   # 单张图像测试
   python discovering.py --mode=classify --query_image=test.jpg
   
   # 系统性能评估
   python discovering.py --mode=fast_slow --test_data_dir=./test_data
   ```

3. **性能分析：**
   ```bash
   # 快思考基线性能
   python discovering.py --mode=fastonly
   
   # 慢思考上限性能
   python discovering.py --mode=slowonly
   ```

### 主要区别：

- **快慢思考模式**：现代化的自适应系统，结合效率和准确性
- **多模态检索模式**：基于特征检索的传统方法
- **传统发现模式**：早期的属性驱动发现方法（未完全实现）

## 技术架构对比

| 模式类别 | 核心技术 | 适用场景 | 实现状态 |
|---------|----------|----------|----------|
| 快慢思考系统 | MLLM + 双模态检索 | 高精度分类，自适应计算 | ✅ 完整实现 |
| 多模态检索 | CLIP + CDV-Captioner | 高效检索，模板匹配 | ✅ 完整实现 |
| 传统发现 | VQA + LLM推理 | 属性驱动发现 | ⚠️ 部分实现 |
