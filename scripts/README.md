# 脚本使用说明

本目录包含了快慢思考系统的各种运行脚本，支持五个数据集，**所有配置参数都在 `config.yaml` 文件中统一管理**。

## 支持的数据集

| 数据集 | DATASET | 类别数 | 配置文件 | 数据目录 |
|---------|---------|---------|-------------|-------------|
| 狗类 | dog | 120 | dog120_all.yml | dogs_120 |
| 鸟类 | bird | 200 | bird200_all.yml | CUB_200_2011 |
| 花类 | flower | 102 | flower102_all.yml | flowers_102 |
| 宠物 | pet | 37 | pet37_all.yml | pet_37 |
| 车类 | car | 196 | car196_all.yml | car_196 |

## 脚本列表

### 1. run_discovery.sh
- **功能**: 通用发现脚本，支持四种模式
- **支持模式**: 
  - `build_knowledge_base`: 构建知识库
  - `fastonly`: 仅快思考评估
  - `slowonly`: 仅慢思考评估  
  - `fast_slow`: 快慢思考联合评估
- **用法**: `bash run_discovery.sh`
- **说明**: 修改脚本中的MODE变量来选择运行模式

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

### 4. run_full_pipeline.sh
- **功能**: 完整流程脚本
- **流程**: 先构建知识库 → 然后进行快慢思考评估
- **用法**: `bash run_full_pipeline.sh`
- **说明**: 自动执行完整的知识库构建和评估流程

### 5. test_config.sh
- **功能**: 配置文件测试脚本
- **用法**: `bash test_config.sh`
- **说明**: 测试 config.yaml 配置文件的读取是否正常

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

# 修改运行模式
modes:
  discovery_mode: "build_knowledge_base"  # 仅run_discovery.sh使用
  eval_mode: "fast_slow"                  # 评估模式
```

### 第二步：运行脚本

#### 单独运行
```bash
# 1. 构建知识库
bash run_build_knowledge_base.sh

# 2. 进行快慢思考评估
bash run_fast_slow.sh
```

#### 完整流程
```bash
# 一键运行完整流程
bash run_full_pipeline.sh
```

#### 灵活使用
```bash
# 使用通用脚本
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

| 参数 | 位置 | 说明 | 默认值 |
|------|------|------|--------|
| **数据集** | `dataset.name` | 数据集选择 | "dog" |
| **GPU** | `gpu.cuda_visible_devices` | GPU编号 | "4" |
| **K-shot** | `hyperparameters.kshot` | 每类样本数 | 3 |
| **测试后缀** | `dataset.test_data_suffix` | 每个类别的测试样本数 | "10" |
| **发现模式** | `modes.discovery_mode` | run_discovery.sh模式 | "build_knowledge_base" |
| **评估模式** | `modes.eval_mode` | 评估模式 | "fast_slow" |

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

## 注意事项

1. **依赖关系**: 评估脚本需要先运行知识库构建
2. **环境检查**: 确保conda环境 `finer_dynamic` 已正确配置
3. **GPU设置**: 根据可用GPU修改CUDA_VISIBLE_DEVICES
4. **数据路径**: 确保测试数据目录存在
5. **日志监控**: 使用 `tail -f` 查看实时日志
