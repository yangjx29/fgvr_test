# 脚本使用说明

本目录包含了快慢思考系统的各种运行脚本，支持五个数据集，**所有配置参数都在 `config.yaml` 文件中统一管理**。

## 支持的数据集

| 数据集 | DATASET | 类别数 | 配置文件 | 数据目录 |
|---------|---------|---------|-------------|-------------|
| 狗类 | dog | 120 | dog120_all.yml | dogs_120 |
| 鸟类 | bird | 200 | bird200_all.yml | CUB_200_2011/CUB_200_2011* |
| 花类 | flower | 102 | flower102_all.yml | flowers_102 |
| 宠物 | pet | 37 | pet37_all.yml | pet_37 |
| 车类 | car | 196 | car196_all.yml | car_196 |

*注：鸟类数据集(CUB-200-2011)有特殊的双层目录结构

## 脚本列表

### 1. run_discovery.sh ⭐ **全面增强**
- **功能**: 通用发现脚本，支持所有19种发现模式
- **支持模式**（分为5大类）: 
  - **传统VQA流程**: `identify`, `howto`, `describe`, `guess`, `postprocess`
  - **快慢思考系统**: `build_knowledge_base`, `classify`, `evaluate`, `fastonly`, `slowonly`, `fast_slow`
  - **分离式推理分类**: `fast_slow_infer`, `fast_slow_classify`
  - **并行分类**: `fast_classify`, `slow_classify`, `terminal_decision`
  - **多模态增强**: `fast_classify_enhanced`, `slow_classify_enhanced`, `terminal_decision_enhanced`
- **用法**: `bash run_discovery.sh`
- **说明**: 通过修改`config.yaml`中的`discovery_mode`来选择运行模式

### 2. run_build_knowledge_base.sh
- **功能**: 专门用于构建知识库
- **模式**: 仅支持 `build_knowledge_base`
- **用法**: `bash run_build_knowledge_base.sh`
- **说明**: 构建快慢思考系统所需的知识库文件

### 3. run_fast_slow.sh
- **功能**: 专门用于快慢思考系统评估
- **模式**: 仅支持 `fast_slow`
- **用法**: `bash run_fast_slow.sh`
- **说明**: 需要先运行知识库构建脚本


## 使用方法

### 第零步：测试配置（可选）
```bash
# 测试配置文件是否正常
bash test_config.sh
```

### 第一步：修改配置文件
所有参数都在 `config.yaml` 文件中配置：

```yaml
# 修改数据集
dataset:
  name: "dog"                 # 改为: dog, bird, flower, pet, car
  test_data_suffix: "10"      # 测试数据后缀

# 修改GPU
gpu:
  cuda_visible_devices: "4"  # 修改GPU编号

# 修改超参数
hyperparameters:
  kshot: 3                    # K-shot learning

# 修改运行模式 (run_discovery.sh支持19种发现模式)
modes:
  # 发现模式选择 - 支持19种模式，分为5大类：
  # 1. 传统VQA流程: identify, howto, describe, guess, postprocess
  # 2. 快慢思考系统: build_knowledge_base, classify, evaluate, fastonly, slowonly, fast_slow  
  # 3. 分离式推理分类: fast_slow_infer, fast_slow_classify
  # 4. 并行分类: fast_classify, slow_classify, terminal_decision
  # 5. 多模态增强: fast_classify_enhanced, slow_classify_enhanced, terminal_decision_enhanced
  discovery_mode: "build_knowledge_base"  
  eval_mode: "fast_slow"                  # 评估模式
```

### 第二步：运行脚本

#### 传统方式：单独运行
```bash
# 1. 构建知识库
bash run_build_knowledge_base.sh

# 2. 进行快慢思考评估
bash run_fast_slow.sh
```

#### 传统方式：完整流程
```bash
# 一键运行完整流程
bash run_full_pipeline.sh
```

#### ⭐ **新方式：分离式运行（推荐用于消融实验）**
```bash
# 方式1: 分步执行
# 1. 构建知识库
bash run_build_knowledge_base.sh

# 2. 执行推理阶段（保存中间结果）
bash run_fast_slow_infer.sh

# 3. 执行分类阶段（基于中间结果）
bash run_fast_slow_classify.sh

# 方式2: 一键分离流程
bash run_fast_slow_pipeline.sh
```

#### 🔥 **新增：全模式支持使用**
```bash
# 使用通用脚本进行所有19种发现模式
bash run_discovery.sh

# 传统VQA流程示例
# 修改 config.yaml 中的 discovery_mode: "describe"
bash run_discovery.sh

# 并行分类示例 
# 修改 config.yaml 中的 discovery_mode: "fast_classify"
bash run_discovery.sh

# 多模态增强示例
# 修改 config.yaml 中的 discovery_mode: "fast_classify_enhanced"
bash run_discovery.sh
```

### 快速切换数据集示例
```bash
# 切换到鸟类数据集
# 修改 config.yaml 中的 dataset.name: "bird"
bash run_full_pipeline.sh

# 切换到花类数据集
# 修改 config.yaml 中的 dataset.name: "flower"
bash run_build_knowledge_base.sh

## 配置说明

### config.yaml 文件结构

```yaml
# GPU配置
gpu:
  cuda_visible_devices: "4"    # GPU编号，多GPU用逗号分隔

# 数据集配置
dataset:
  name: "dog"                   # 数据集: dog, bird, flower, pet, car
  test_data_suffix: "10"        # 测试数据后缀: 1, 3, 5, 10

# 模型超参数
hyperparameters:
  kshot: 3                      # K-shot learning 的 K 值

# 运行模式
modes:
  discovery_mode: "build_knowledge_base"  # run_discovery.sh 的模式
  eval_mode: "fast_slow"                  # 评估模式

# 环境配置
environment:
  conda_env: "finer_dynamic"              # conda环境名
  conda_base: "/home/hdl/miniconda3"      # conda路径
  project_root: "/home/hdl/project/fgvr_test"  # 项目根目录

# 日志配置
logging:
  base_dir: "/home/hdl/project/fgvr_test/logs"  # 日志目录
```

### 主要配置参数

| 参数 | 位置 | 说明 | 默认值 | 支持值 |
|------|------|------|--------|--------|
| **数据集** | `dataset.name` | 数据集选择 | "dog" | dog, bird, flower, pet, car |
| **GPU** | `gpu.cuda_visible_devices` | GPU编号 | "4" | 任意可用GPU编号 |
| **K-shot** | `hyperparameters.kshot` | 每类样本数 | 3 | 正整数 |
| **测试后缀** | `dataset.test_data_suffix` | 每个类别的测试样本数 | "10" | 1,2,3,...,10,random |
| **发现模式** | `modes.discovery_mode` | run_discovery.sh模式 | "build_knowledge_base" | **19种模式（见下表）** |
| **评估模式** | `modes.eval_mode` | 评估模式 | "fast_slow" | fastonly, slowonly, fast_slow |

#### 🔥 **发现模式详细说明**

| 类别 | 模式名称 | 功能说明 | 使用场景 |
|------|----------|----------|----------|
| **传统VQA流程** | `identify` | 识别图像的超类 | 数据集探索 |
| | `howto` | 询问区分方法 | 属性分析 |
| | `describe` | 生成属性描述 | 知识发现 |
| | `guess` | 基于属性推理类别 | 类别推断 |
| | `postprocess` | 后处理推理结果 | 结果清理 |
| **快慢思考系统** | `build_knowledge_base` | 构建知识库 | 系统初始化 |
| | `classify` | 单张图像分类 | 实时分类 |
| | `evaluate` | 完整系统评估 | 性能测试 |
| | `fastonly` | 仅快思考评估 | 快速测试 |
| | `slowonly` | 仅慢思考评估 | 深度分析 |
| | `fast_slow` | 完整快慢思考 | 标准评估 |
| **分离式推理分类** | `fast_slow_infer` | 推理阶段 | 消融实验 |
| | `fast_slow_classify` | 分类阶段 | 决策分析 |
| **并行分类** | `fast_classify` | 并行快思考分类 | 快速处理 |
| | `slow_classify` | 并行慢思考分类 | 深度处理 |
| | `terminal_decision` | 最终决策融合 | 结果整合 |
| **多模态增强** | `fast_classify_enhanced` | 增强快思考分类 | 性能提升 |
| | `slow_classify_enhanced` | 增强慢思考分类 | 精度优化 |
| | `terminal_decision_enhanced` | 增强决策融合 | 最优结果 |

### 测试数据后缀说明

`test_data_suffix` 参数控制每个类别使用的测试样本数量：

| 后缀值 | 含义 | 适用场景 |
|---------|------|----------|
| "1" | 每个类别1个样本 | 快速测试 |
| "3" | 每个类别3个样本 | 小规模测试 |
| "5" | 每个类别5个样本 | 中等规模测试 |
| "10" | 每个类别10个样本 | 完整测试（推荐） |
| "random" | 随机数量的样本 | 随机测试 |

**不同数据集的建议配置：**
- **宠物数据集 (pet_37)**: 类别数较少，建议使用 `"5"` 或 `"10"`
- **狗类数据集 (dog120)**: 类别数中等，建议使用 `"3"` 或 `"5"`
- **鸟类数据集 (bird200)**: 类别数较多，建议使用 `"1"` 或 `"3"`

### 自动配置
以下参数会根据 `dataset.name` 自动配置：
- **类别数**: 数据集的类别数量
- **配置文件**: discovering.py 的配置文件
- **数据目录**: 数据集在 datasets/ 下的目录名
- **路径生成**: 知识库、测试数据、结果输出路径

## 🚀 **全面增强功能特点**

### 🔄 分离式快慢思考流程
新增的分离式脚本提供以下优势：

1. **推理与分类分离**: 
   - `fast_slow_infer`: 执行推理并保存所有中间结果
   - `fast_slow_classify`: 基于保存的结果执行分类逻辑

2. **消融实验友好**:
   - 推理阶段只需运行一次，分类阶段可重复运行
   - 便于测试不同的分类策略和融合方法
   - 大幅减少实验时间成本

3. **结果保存位置**:
   - 推理结果: `experiments/<dataset><num>/infer/`
   - 分类结果: `experiments/<dataset><num>/classify/`

4. **三种分类路径支持**:
   - 快思考直接分类 (`decision_path: "fast_only"`)
   - 慢思考一致分类 (`decision_path: "slow_consistent"`)
   - 快慢不一致裁决 (`decision_path: "final_arbitration"`)

### 🎯 **19种发现模式全支持**
run_discovery.sh现在支持discovering.py的全部19种模式：

#### 📊 **模式分类和用途**

##### 1. 传统VQA流程模式
```bash
# 数据集探索和属性发现
config.yaml: discovery_mode: "identify"     # 识别图像超类
config.yaml: discovery_mode: "describe"     # 生成属性描述  
config.yaml: discovery_mode: "guess"        # 推理类别名称
config.yaml: discovery_mode: "postprocess"  # 后处理结果
```

##### 2. 快慢思考系统模式
```bash
# 标准快慢思考流程
config.yaml: discovery_mode: "build_knowledge_base"  # 构建知识库
config.yaml: discovery_mode: "fastonly"             # 仅快思考评估
config.yaml: discovery_mode: "fast_slow"            # 完整快慢思考
```

##### 3. 分离式推理分类模式
```bash
# 推理与分类分离（消融实验友好）
config.yaml: discovery_mode: "fast_slow_infer"     # 推理阶段
config.yaml: discovery_mode: "fast_slow_classify"  # 分类阶段
```

##### 4. 并行分类模式
```bash
# 并行处理（可真正并行运行）
config.yaml: discovery_mode: "fast_classify"      # 并行快思考
config.yaml: discovery_mode: "slow_classify"      # 并行慢思考
config.yaml: discovery_mode: "terminal_decision"  # 最终决策融合
```

##### 5. 多模态增强模式
```bash
# MEC框架增强（性能提升）
config.yaml: discovery_mode: "fast_classify_enhanced"      # 增强快思考
config.yaml: discovery_mode: "slow_classify_enhanced"      # 增强慢思考  
config.yaml: discovery_mode: "terminal_decision_enhanced"  # 增强决策融合
```

### 📊 详细结果记录
分类结果包含完整的决策路径信息，便于深度分析：
```json
{
  "decision_path": "final_arbitration",
  "fast_slow_consistent": false,
  "fast_prediction": "Chihuahua",
  "slow_prediction": "Pomeranian",
  "final_prediction": "Pomeranian"
}
```

### 🔗 **模式执行流程示例**

#### 完整传统VQA发现流程
```bash
# 1. 数据集探索
echo 'discovery_mode: "identify"' >> config.yaml && bash run_discovery.sh
# 2. 属性描述生成  
echo 'discovery_mode: "describe"' >> config.yaml && bash run_discovery.sh
# 3. 类别推理
echo 'discovery_mode: "guess"' >> config.yaml && bash run_discovery.sh
# 4. 结果后处理
echo 'discovery_mode: "postprocess"' >> config.yaml && bash run_discovery.sh
```

#### 并行分类完整流程
```bash
# 1. 构建知识库
echo 'discovery_mode: "build_knowledge_base"' >> config.yaml && bash run_discovery.sh
# 2. 推理阶段
echo 'discovery_mode: "fast_slow_infer"' >> config.yaml && bash run_discovery.sh
# 3. 并行分类（可在不同终端/GPU同时运行）
echo 'discovery_mode: "fast_classify"' >> config.yaml && bash run_discovery.sh &
echo 'discovery_mode: "slow_classify"' >> config.yaml && bash run_discovery.sh &
wait
# 4. 最终决策融合
echo 'discovery_mode: "terminal_decision"' >> config.yaml && bash run_discovery.sh
```

#### 增强分类完整流程
```bash
# 1. 构建知识库
echo 'discovery_mode: "build_knowledge_base"' >> config.yaml && bash run_discovery.sh
# 2. 推理阶段  
echo 'discovery_mode: "fast_slow_infer"' >> config.yaml && bash run_discovery.sh
# 3. 增强并行分类
echo 'discovery_mode: "fast_classify_enhanced"' >> config.yaml && bash run_discovery.sh &
echo 'discovery_mode: "slow_classify_enhanced"' >> config.yaml && bash run_discovery.sh &
wait
# 4. 增强决策融合
echo 'discovery_mode: "terminal_decision_enhanced"' >> config.yaml && bash run_discovery.sh
```

#### 🎯 **推荐使用场景**

| 使用场景 | 推荐模式 | 优势 |
|---------|---------|------|
| **新数据集探索** | `identify` → `describe` → `guess` | 完整的属性发现流程 |
| **快速性能测试** | `build_knowledge_base` → `fastonly` | 最快得到基准结果 |
| **标准评估** | `build_knowledge_base` → `fast_slow` | 完整系统性能 |
| **消融实验** | `fast_slow_infer` → `fast_slow_classify` | 可重复分类实验 |
| **并行加速** | `fast_classify` + `slow_classify` → `terminal_decision` | 真正并行处理 |
| **性能优化** | `*_enhanced` 系列模式 | MEC框架增强 |
| **资源受限** | `fast_classify` 仅快思考 | 节省计算资源 |

### 📋 **模式依赖关系**

```
传统VQA流程: identify → describe → guess → postprocess

快慢思考系统: build_knowledge_base → [fastonly|slowonly|fast_slow|evaluate]

分离式流程: build_knowledge_base → fast_slow_infer → fast_slow_classify

并行流程: build_knowledge_base → fast_slow_infer → [fast_classify + slow_classify] → terminal_decision

增强流程: build_knowledge_base → fast_slow_infer → [fast_classify_enhanced + slow_classify_enhanced] → terminal_decision_enhanced
```

## 注意事项

1. **依赖关系**: 
   - 评估脚本需要先运行知识库构建
   - `fast_slow_classify` 需要先运行 `fast_slow_infer`
   - 并行模式需要先运行对应的前置模式
   - 增强模式需要MEC框架支持
2. **环境检查**: 确保conda环境 `finer_dynamic` 已正确配置
3. **GPU设置**: 根据可用GPU修改CUDA_VISIBLE_DEVICES
4. **数据路径**: 确保测试数据目录存在
5. **日志监控**: 使用 `tail -f` 查看实时日志
6. **存储空间**: 推理结果会占用一定存储空间，注意磁盘容量
7. **并行处理**: `fast_classify`和`slow_classify`可真正并行运行
8. **模式选择**: 根据实验需求选择合适的模式组合
9. **特殊数据集路径**: 
   - **鸟类数据集(CUB-200-2011)**: 具有特殊的双层目录结构 `CUB_200_2011/CUB_200_2011/`
   - 脚本已自动处理这种特殊结构，无需手动调整
