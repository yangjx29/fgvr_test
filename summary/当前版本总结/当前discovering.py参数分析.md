# discovering.py 参数分析

## 概述

`discovering.py` 是项目的主入口脚本，支持多种运行模式和丰富的命令行参数配置。

## 超参数配置

### 超参数对象

超参数通过全局 `Hyperparameters` 类管理，不再依赖 YAML 配置文件：

```python
class Hyperparameters:
    """超参数配置类，用于存储和管理运行时超参数"""
    def __init__(self, experience_number: int = 8, classify_top_k: int = 10):
        self.experience_number = experience_number  # 经验库最大经验条数
        self.classify_top_k = classify_top_k        # 分类时的top_k候选数

# 全局超参数实例
hyperparams = Hyperparameters()
```

### 超参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `experience_number` | int | 8 | 经验库最大经验条数，控制 `ExperienceBaseBuilder` 中的 `max_strategy_rules` |
| `classify_top_k` | int | 10 | 分类时的 top_k 候选数，用于 `classify_single_image` 方法 |

### 命令行传入方式

```bash
python discovering.py --mode fast_slow --experience_number 12 --classify_top_k 15
```

## 命令行参数完整列表

### 运行模式参数

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `--mode` | str | `build_knowledge_base` | `build_knowledge_base`, `classify`, `evaluate`, `fastonly`, `slowonly`, `fast_slow`, `pipeline` | 运行模式 |

### 配置文件参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config_file_env` | str | `./configs/env_machine.yml` | 环境配置文件路径 |
| `--config_file_expt` | str | `./configs/expts/bird200_all.yml` | 实验配置文件路径 |

### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_per_category` | str | `3` | 每个类别的样本数量，可选 1-10 或 random |
| `--knowledge_base_dir` | str | `./knowledge_base` | 知识库目录 |
| `--query_image` | str | None | 查询图像路径 |
| `--test_data_dir` | str | None | 测试数据目录 |
| `--use_test_data` | flag | False | 使用 images_test 目录进行测试 |
| `--test_percentage` | float | 100.0 | 使用的测试图像百分比 (0-100) |
| `--results_out` | str | `./results.json` | 结果输出路径 |

### 快慢思考系统参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_slow_thinking` | bool | None | 强制使用慢思考（None 为自动） |
| `--confidence_threshold` | float | 0.8 | 快思考置信度阈值 |
| `--similarity_threshold` | float | 0.7 | 触发机制相似度阈值 |

### 超参数配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--experience_number` | int | 8 | 经验库最大经验条数 |
| `--classify_top_k` | int | 10 | 分类时的 top_k 候选数 |

## 超参数使用位置

### 1. experience_number

在 `experience_base_builder.py` 中使用：

```python
# experience_base_builder.py
def get_experience_number():
    """从discovering模块获取experience_number超参数"""
    from discovering import hyperparams
    return hyperparams.experience_number

class ExperienceBaseBuilder:
    def __init__(self, ...):
        self.max_strategy_rules = get_experience_number()  # 经验条数上限
```

### 2. classify_top_k

在 `discovering.py` 的 `fast_slow` 模式中使用：

```python
# discovering.py - fast_slow 模式
result = system.classify_single_image(
    path, 
    use_slow_thinking=None, 
    top_k=hyperparams.classify_top_k  # 使用超参数
)
```

## 运行示例

### 构建知识库

```bash
python discovering.py --mode build_knowledge_base
```

### 快慢思考评估（自定义超参数）

```bash
python discovering.py \
    --mode fast_slow \
    --experience_number 12 \
    --classify_top_k 15 \
    --test_percentage 50.0
```

### 完整 Pipeline

```bash
python discovering.py \
    --mode pipeline \
    --config_file_expt ./configs/expts/dog120_all.yml \
    --experience_number 10 \
    --classify_top_k 20
```

## 变更历史

- **2024-12-01**: 移除 `configs/hyperparameters.yaml` 依赖，改为命令行参数传入
  - 新增 `Hyperparameters` 类管理超参数
  - 新增 `--experience_number` 和 `--classify_top_k` 命令行参数
  - `experience_base_builder.py` 改为从 `discovering.hyperparams` 读取配置
