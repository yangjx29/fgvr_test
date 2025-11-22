# quick_start_serial.sh - 智能多GPU串行执行脚本完整文档

**版本**: 2.2  
**更新日期**: 2025-11-22  
**作者**: AI Assistant  
**脚本位置**: `scripts/quick_start_serial.sh`

---

## 📋 目录

- [1. 概述](#1-概述)
- [2. 核心功能](#2-核心功能)
- [3. 配置参数](#3-配置参数)
- [4. 工作原理](#4-工作原理)
- [5. GPU调度算法](#5-gpu调度算法)
- [6. 使用方法](#6-使用方法)
- [7. 命令格式](#7-命令格式)
- [8. 后台运行模式](#8-后台运行模式)
- [9. 故障排查](#9-故障排查)
- [10. 最佳实践](#10-最佳实践)

---

## 1. 概述

`quick_start_serial.sh` 是一个智能的多GPU任务调度脚本，能够自动管理GPU资源，串行执行多个命令，并提供完善的错误处理和重试机制。

### 1.1 设计目标

- **智能调度**: 自动分配空闲GPU，避免资源冲突
- **真正串行**: 确保上一个命令完全结束后才执行下一个
- **资源共享**: 支持GPU预留机制，适合多人共享服务器
- **容错性强**: 失败自动重试，支持无限重试模式
- **易于使用**: 配置简单，支持Bash和Python脚本

### 1.2 适用场景

- ✅ 批量训练多个模型
- ✅ 多数据集实验
- ✅ 夜间无人值守运行
- ✅ 共享服务器环境
- ✅ 资源受限场景

---

## 2. 核心功能

### 2.1 功能列表

| 功能 | 说明 |
|------|------|
| **多GPU动态分配** | 自动从GPU池中选择空闲GPU |
| **智能显存监控** | 检查GPU剩余显存是否满足要求 |
| **单卡/多卡支持** | 支持1-30张GPU的任务 |
| **多卡命令优先** | 自动优先调度多卡任务 |
| **GPU占用追踪** | 避免同一GPU被分配给多个命令 |
| **GPU预留机制** | 为其他用户预留GPU |
| **自动重试** | 失败自动重试，支持无限重试 |
| **真正串行** | 等待进程完全结束 |
| **后台运行** | 支持完全后台运行模式 |
| **类型识别** | 自动识别Bash/Python脚本 |
| **递增日志** | 自动避免日志覆盖 |

---

## 3. 配置参数

### 3.1 核心配置（脚本开头，便于修改）

```bash
# 第2行：GPU资源池
TARGET_GPUS="0 1 2 3 4 5 6 7 8 9"  # 或 "all" 表示所有GPU

# 第3行：日志目录
LOG_DIR="/home/hdl/project/fgvr_test/logs/quick_start"

# 第4行：预留GPU数量
RESERVED_GPUS=1

# 第5行：最大重试次数
MAX_RETRY_TIMES=-1

# 第6行：后台运行模式
RUN_IN_BACKGROUND=true

# 第7行：极限资源利用模式（🆕 v2.2）
EXTREME_MODE=false

# 第8行：GPU共享模式
ALLOW_SHARED_GPU=true

# 第9行：调试模式（🆕 v2.2）
DEBUG_MODE=false
```

### 3.2 参数详解

#### TARGET_GPUS
- **类型**: 字符串（空格分隔）或特殊值 `"all"`
- **示例**: 
  - `"0 1 2 3"` - 指定GPU 0, 1, 2, 3
  - `"0 2 4 6 8"` - 指定偶数编号GPU
  - `"all"` - 🆕 使用服务器上所有可用GPU
- **说明**: 定义可用的GPU资源池，脚本只从这些GPU中分配

#### LOG_DIR
- **类型**: 路径字符串
- **默认**: `./logs/quick_start`
- **说明**: 日志文件保存目录，自动创建

#### RESERVED_GPUS
- **类型**: 整数 (≥0)
- **默认**: `1`
- **说明**: 为其他用户预留的GPU数量
  - `0`: 不预留，使用所有空闲GPU
  - `k`: 至少预留k张GPU给其他用户

#### MAX_RETRY_TIMES
- **类型**: 整数 (≥-1)
- **默认**: `-1`
- **说明**: 命令失败时的重试策略
  - `-1`: 无限重试，直到成功
  - `0`: 不重试，失败直接放弃
  - `N>0`: 最多重试N次

#### RUN_IN_BACKGROUND
- **类型**: 布尔值 (`true`/`false`)
- **默认**: `true`
- **说明**: 是否后台运行
  - `true`: 完全后台运行，日志实时输出到文件
  - `false`: 前台运行，输出到终端和日志文件

#### EXTREME_MODE (🆕 v2.2)
- **类型**: 布尔值 (`true`/`false`)
- **默认**: `false`
- **说明**: 极限资源利用模式
  - `true`: 只要有足够显存就调度，允许多进程共享同一GPU
  - `false`: 正常模式，遵循ALLOW_SHARED_GPU设置
- **优先级**: 高于ALLOW_SHARED_GPU
- **示例**: A800(80GB)被占用16GB，进程需12GB
  - `EXTREME_MODE=true`: 可再放5个进程（5×12=60GB < 64GB剩余）
  - `EXTREME_MODE=false`: 不允许再放进程（已被占用）

#### DEBUG_MODE (🆕 v2.2)
- **类型**: 布尔值 (`true`/`false`)
- **默认**: `false`
- **说明**: 调试模式
  - `true`: 输出详细的调试信息、GPU状态、错误分析
  - `false`: 正常输出
- **用途**: 排查调度问题、显存分配、命令执行错误

#### CHECK_INTERVAL
- **类型**: 整数（秒）
- **默认**: `3`
- **位置**: 脚本内部（第102行）
- **说明**: GPU显存检查间隔

---

## 4. 工作原理

### 4.1 整体流程

```
开始
 ↓
初始化GPU占用追踪表
 ↓
解析并排序命令（多卡优先）
 ↓
┌─────────────────────┐
│ 对每个命令执行：    │
│  1. 扫描可用GPU     │
│  2. 分配GPU         │
│  3. 标记为占用      │
│  4. 执行命令        │
│  5. 等待完成        │
│  6. 释放GPU         │
│  7. 检查结果        │
│  8. 失败→重试       │
└─────────────────────┘
 ↓
输出统计信息
 ↓
结束
```

### 4.2 GPU占用追踪机制

脚本维护一个关联数组 `SCRIPT_GPU_OCCUPIED`，记录每张GPU的占用状态：

```bash
SCRIPT_GPU_OCCUPIED[0]=0  # 0=空闲
SCRIPT_GPU_OCCUPIED[1]=1  # 1=被本脚本占用
```

**关键点**：
- ✅ 只追踪本脚本启动的进程
- ✅ 其他用户的进程通过显存检查识别
- ✅ 分配GPU时标记为占用
- ✅ 进程结束后立即释放

---

## 5. GPU调度算法

### 5.1 可用GPU筛选流程

```
1. 获取TARGET_GPUS中的所有GPU
   ↓
2. 过滤：跳过被本脚本占用的GPU
   ↓
3. 过滤：跳过显存不足的GPU
   ↓
4. 应用预留策略：
   - 可用GPU数 < RESERVED_GPUS → 返回空列表（等待）
   - 可用GPU数 = RESERVED_GPUS → 返回1张GPU
   - 可用GPU数 > RESERVED_GPUS → 返回(可用数 - RESERVED_GPUS)张
   ↓
5. 返回可分配的GPU列表
```

### 5.2 GPU预留策略详解

**规则**：
- 空闲GPU数 < k → 不使用任何GPU，等待
- 空闲GPU数 = k → 使用1张（特殊处理）
- 空闲GPU数 > k → 使用到只剩k张

**示例** (`RESERVED_GPUS=2`):

| 满足显存的GPU | 本脚本占用 | 可用数 | 结果 |
|--------------|----------|--------|------|
| 8张 | 0张 | 8张 | 可分配6张（留2张） |
| 5张 | 1张 | 4张 | 可分配2张（留2张） |
| 4张 | 1张 | 3张 | 可分配1张（留2张） |
| 3张 | 1张 | 2张 | 可分配1张（特殊：空闲=预留） |
| 2张 | 1张 | 1张 | 等待（不足预留数） |

### 5.3 调度优先级

1. **按GPU数量排序**（降序）
   - 4卡命令 > 3卡命令 > 2卡命令 > 单卡命令

2. **避免GPU冲突**
   - 已被本脚本占用的GPU不会再次分配
   - 即使GPU显存充足，也不会分配给新命令

3. **显存检查**
   - 检查nvidia-smi报告的剩余显存
   - 包含本脚本进程和其他用户进程的占用

---

## 6. 使用方法

### 6.1 基本使用

**步骤1**: 编辑脚本
```bash
vim scripts/quick_start_serial.sh
```

**步骤2**: 配置GPU池和参数（第2-6行）
```bash
TARGET_GPUS="0 1 2 3"
RESERVED_GPUS=1
MAX_RETRY_TIMES=3
RUN_IN_BACKGROUND=true
```

**步骤3**: 添加命令到COMMANDS数组
```bash
COMMANDS=(
    "bash scripts/run_pipeline.sh aircraft --gpu \${available_gpu} | 35 | 1"
    "python train.py --gpu \${available_gpu} --epochs 100 | 20 | 1"
)
```

**步骤4**: 运行脚本
```bash
bash scripts/quick_start_serial.sh
```

**步骤5**: 查看日志
```bash
tail -f logs/quick_start/quick_start_serial.log
```

### 6.2 前台运行

```bash
# 修改配置
RUN_IN_BACKGROUND=false

# 运行
bash scripts/quick_start_serial.sh
```

### 6.3 后台运行

```bash
# 修改配置
RUN_IN_BACKGROUND=true

# 运行（自动后台）
bash scripts/quick_start_serial.sh

# 查看日志
tail -f logs/quick_start/quick_start_serial.log
```

---

## 7. 命令格式

### 7.1 基本格式

```
"命令内容 | 所需显存(GB) | GPU数量"
```

### 7.2 Bash脚本示例

#### 单卡命令
```bash
"bash scripts/run_pipeline.sh aircraft --gpu \${available_gpu} | 35 | 1"
"bash scripts/run_fast_slow.sh eurosat --gpu \${available_gpu} | 20 | 1"
```

#### 多卡命令
```bash
"bash scripts/run_ddp.sh model1 --gpus \${available_gpu_0},\${available_gpu_1} | 40 | 2"
```

### 7.3 Python脚本示例

#### 单卡命令 - 使用CUDA_VISIBLE_DEVICES
```bash
"CUDA_VISIBLE_DEVICES=\${available_gpu} python train.py --epochs 100 | 18 | 1"
```

#### 单卡命令 - 使用--gpu参数
```bash
"python train.py --gpu \${available_gpu} --lr 0.001 | 20 | 1"
```

#### 多卡命令 - PyTorch DDP
```bash
"CUDA_VISIBLE_DEVICES=\${available_gpu_0},\${available_gpu_1} python -m torch.distributed.launch train.py | 40 | 2"
```

#### 多卡命令 - 自定义参数
```bash
"python train_multi.py --gpus \${available_gpu_0} \${available_gpu_1} \${available_gpu_2} | 50 | 3"
```

### 7.4 CPU命令
```bash
"python preprocess.py --input data/ --output processed/ | 0 | 0"
"bash scripts/analyze.sh | 0 | 0"
```

### 7.5 GPU变量说明

| 变量 | 说明 | 示例 |
|------|------|------|
| `${available_gpu}` | 单卡命令，自动分配的GPU编号 | `0`, `3`, `7` |
| `${available_gpu_0}` | 多卡命令第1张GPU | `0` |
| `${available_gpu_1}` | 多卡命令第2张GPU | `1` |
| `${available_gpu_N}` | 多卡命令第N+1张GPU | `2`, `3`, ... |

**注意**: 命令中必须使用转义 `\${...}`，否则会被shell提前替换

---

## 8. 后台运行模式

### 8.1 工作原理

当 `RUN_IN_BACKGROUND=true` 时：

1. 脚本检查环境变量 `QUICK_START_BACKGROUND`
2. 如果未设置，使用 `nohup` 在后台重新启动脚本
3. 设置环境变量 `QUICK_START_BACKGROUND=1`
4. 后台实例执行实际任务

### 8.2 优点

- ✅ 完全后台运行，不占用终端
- ✅ 日志实时写入文件
- ✅ 终端关闭后继续运行
- ✅ 可以退出SSH会话

### 8.3 查看运行状态

```bash
# 查看进程
ps aux | grep quick_start_serial

# 查看日志
tail -f logs/quick_start/quick_start_serial.log

# 实时刷新GPU状态
watch -n 1 nvidia-smi
```

### 8.4 停止后台任务

```bash
# 找到进程ID
ps aux | grep quick_start_serial

# 终止进程
kill <PID>

# 强制终止
kill -9 <PID>
```

---

## 9. 故障排查

### 9.1 常见问题

#### 问题1: 两张空闲卡但没有分配

**可能原因**:
1. GPU已被本脚本的其他命令占用
2. 显存不足（检查阈值）
3. 预留策略限制

**排查方法**:
```bash
# 检查GPU状态
nvidia-smi

# 检查日志
grep "可用GPU" logs/quick_start/quick_start_serial.log

# 检查占用追踪
grep "标记\|释放" logs/quick_start/quick_start_serial.log
```

**解决方案**:
- 检查 `RESERVED_GPUS` 设置
- 降低显存需求
- 等待GPU释放

#### 问题2: 前台被占用

**原因**: `RUN_IN_BACKGROUND=false`

**解决**: 设置 `RUN_IN_BACKGROUND=true`

#### 问题3: 命令失败不重试

**原因**: `MAX_RETRY_TIMES=0`

**解决**: 设置 `MAX_RETRY_TIMES=-1` 或更大的值

#### 问题4: GPU显存不足

**排查**:
```bash
# 查看日志中的错误
grep "out of memory" logs/quick_start/quick_start_serial.log

# 检查实际显存需求
```

**解决**:
- 增加命令的显存需求值
- 减少batch size
- 使用更大显存的GPU

### 9.2 调试模式

在命令前添加 `set -x` 查看详细执行过程：
```bash
set -x
bash scripts/quick_start_serial.sh
```

---

## 10. 最佳实践

### 10.1 配置建议

#### 共享服务器
```bash
TARGET_GPUS="0 1 2 3 4 5 6 7"
RESERVED_GPUS=2              # 为其他人预留
MAX_RETRY_TIMES=3            # 有限重试
RUN_IN_BACKGROUND=true
```

#### 独占服务器
```bash
TARGET_GPUS="0 1 2 3"
RESERVED_GPUS=0              # 不预留
MAX_RETRY_TIMES=-1           # 无限重试
RUN_IN_BACKGROUND=true
```

#### 测试环境
```bash
TARGET_GPUS="0 1"
RESERVED_GPUS=0
MAX_RETRY_TIMES=0            # 不重试，快速失败
RUN_IN_BACKGROUND=false      # 前台运行，便于调试
```

### 10.2 命令编写建议

1. **显存设置留余量**
   ```bash
   # 实际需要20GB，设置为25GB
   "python train.py | 25 | 1"
   ```

2. **使用绝对路径**
   ```bash
   "bash /home/user/scripts/run.sh | 20 | 1"
   ```

3. **多卡命令放前面**
   - 脚本会自动排序，但建议手动放前面便于阅读

4. **测试单个命令**
   ```bash
   COMMANDS=(
       "echo TEST | 0 | 0"
   )
   ```

### 10.3 监控建议

#### 实时监控GPU
```bash
watch -n 1 nvidia-smi
```

#### 监控日志
```bash
tail -f logs/quick_start/quick_start_serial.log
```

#### 监控进程
```bash
ps aux | grep python | grep train
```

### 10.4 性能优化

1. **调整CHECK_INTERVAL**
   - 默认3秒，可适当调整
   - 更快: `CHECK_INTERVAL=1`
   - 更省资源: `CHECK_INTERVAL=5`

2. **合理设置RESERVED_GPUS**
   - 共享服务器: 1-2张
   - 独占服务器: 0张

3. **批量执行相似任务**
   - 相同显存需求的命令放一起
   - 便于GPU资源利用

---

## 11. 版本历史

### v2.2 (2025-11-22)
- ✅ 新增极限资源利用模式（`EXTREME_MODE=true`）
- ✅ 支持多进程共享同一GPU（极限模式）
- ✅ 支持 `TARGET_GPUS="all"` 自动使用所有GPU
- ✅ 新增调试模式（`DEBUG_MODE=true`）
- ✅ 增强错误类型识别和诊断
- ✅ 显示详细的错误分析和修复建议
- ✅ 显存追踪机制（极限模式下）
- ✅ 改进日志输出和调试信息

### v2.1 (2025-11-22)
- ✅ 新增显存优先选择策略（优先选择显存多的GPU）
- ✅ 新增独占模式（`ALLOW_SHARED_GPU=false`）
- ✅ 新增详细的调度日志输出
- ✅ 每次GPU扫描显示时间戳和状态
- ✅ 显示调度成功/失败原因
- ✅ 增加GPU统计信息（利用率、显存占用等）

### v2.0 (2025-11-22)
- ✅ 新增GPU占用追踪机制
- ✅ 新增后台运行模式
- ✅ 新增Python脚本支持
- ✅ 修复调度冲突问题
- ✅ 完善文档

### v1.0 (2025-11-22)
- ✅ 基础多GPU调度
- ✅ 显存监控
- ✅ 预留机制
- ✅ 重试机制

---

## 12. 附录

### 12.1 脚本结构

```
quick_start_serial.sh
├── 配置区域 (第2-6行)
├── 命令数组 (第8-54行)
├── 初始化 (第106-117行)
├── 工具函数 (第119-431行)
│   ├── 颜色输出
│   ├── 日志管理
│   ├── 命令解析
│   ├── GPU监控
│   ├── GPU分配
│   ├── 占用追踪
│   └── 进程管理
├── 主执行逻辑 (第433-662行)
└── 脚本入口 (第664-681行)
```

### 12.2 关键函数说明

| 函数 | 说明 |
|------|------|
| `get_available_gpus()` | 获取可用GPU列表（考虑占用、显存、预留） |
| `allocate_gpus()` | 分配GPU给命令 |
| `mark_gpus_occupied()` | 标记GPU为占用 |
| `mark_gpus_free()` | 释放GPU |
| `wait_for_process_completion()` | 等待进程完全结束 |
| `check_command_success()` | 检查命令执行结果 |
| `identify_command_type()` | 识别命令类型（bash/python） |

### 12.3 环境要求

- **操作系统**: Linux
- **Shell**: Bash 4.0+
- **GPU工具**: nvidia-smi
- **Python**: 2.7+ 或 3.6+ (可选)

### 12.4 联系方式

如有问题或建议，请联系开发团队。

---

## 13. v2.1 新增功能详解

### 13.1 显存优先选择策略

当有多张GPU满足条件时，脚本会自动选择剩余显存最多的GPU。

**工作原理**:
1. 扫描所有满足条件的GPU
2. 记录每张GPU的剩余显存
3. 按剩余显存降序排序
4. 优先分配显存多的GPU

**示例**:
```
GPU 0: 40GB 可用
GPU 1: 45GB 可用  ← 优先选择
GPU 2: 35GB 可用

结果：命令会被分配到GPU 1
```

**优势**:
- 更好的资源利用
- 减少显存碎片化
- 为后续任务预留更多选择

### 13.2 独占模式（ALLOW_SHARED_GPU）

**配置**: `ALLOW_SHARED_GPU=false`

**判断标准**:
- GPU利用率 = 0%
- 显存占用 < 3%（即空闲显存 > 97%）

**使用场景**:

#### 共享模式（默认）
```bash
ALLOW_SHARED_GPU=true
```
- ✅ 允许与其他用户共享GPU
- ✅ 只要显存充足即可使用
- ✅ 适合多人共享服务器

#### 独占模式
```bash
ALLOW_SHARED_GPU=false
```
- ✅ 只使用完全空闲的GPU
- ✅ 避免与他人任务冲突
- ✅ 获得最佳性能
- ✅ 适合关键任务

**示例对比**:

| GPU状态 | 利用率 | 显存占用 | 共享模式 | 独占模式 |
|---------|--------|----------|----------|----------|
| 完全空闲 | 0% | 1% | ✅ 可用 | ✅ 可用 |
| 轻度使用 | 5% | 10% | ✅ 可用 | ❌ 不可用 |
| 中度使用 | 30% | 50% | ✅ 可用 | ❌ 不可用 |
| 显存占用高 | 0% | 80% | ❌ 不可用 | ❌ 不可用 |

### 13.3 详细日志输出

#### 启动信息
```
[INFO] GPU共享模式: 独占模式
  └─ 独占模式已启用：仅使用利用率0%且显存占用<3%的GPU
[INFO] GPU选择策略: 优先选择剩余显存多的GPU
```

#### GPU扫描日志
```
[2025-11-22 23:45:12] 开始扫描可用GPU...
  └─ GPU 0: 可用 (空闲:45GB, 利用率:0%, 显存占用:1.2%)
  └─ GPU 1: 已被本脚本占用，跳过
  └─ GPU 2: 不满足独占条件 (利用率:15%, 显存占用:30%), 跳过
  └─ GPU 3: 显存不足 (空闲:10GB < 需求:35GB)
  └─ 找到 1 张可用GPU（已按剩余显存降序排列）: 0
  └─ 预留策略: 可用数=2 > 预留数=1，可使用1张
```

#### 分配流程日志
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
开始GPU分配流程
  需求: 1张GPU，每张需35GB显存
  独占模式: 启用
  预留GPU数: 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[2025-11-22 23:45:15] 第 1 次尝试分配GPU
  GPU统计信息:
    • 总GPU数: 4
    • 本脚本占用: 1 张
    • 显存充足: 2 张
    • 完全空闲: 1 张
    • 预留要求: 1 张
    • 最终可分配: 1 张

✓ 调度成功！分配 1 张GPU: 0
  • GPU 0: 45GB可用, 利用率0%
```

#### 调度失败日志
```
[2025-11-22 23:46:20] 第 5 次尝试分配GPU
  GPU统计信息:
    • 总GPU数: 4
    • 本脚本占用: 2 张
    • 显存充足: 2 张
    • 完全空闲: 0 张
    • 预留要求: 1 张
    • 最终可分配: 0 张

✗ 调度失败：需要 1 张，可分配 0 张
  失败原因: 本脚本占用2张; 独占模式要求不满足(仅0张完全空闲); 预留策略限制; 
  将在 3 秒后重试...
```

### 13.4 配置组合建议

#### 场景1：独占服务器 + 最大性能
```bash
TARGET_GPUS="0 1 2 3"
RESERVED_GPUS=0              # 不预留
ALLOW_SHARED_GPU=false       # 独占模式
MAX_RETRY_TIMES=-1           # 无限重试
```

#### 场景2：共享服务器 + 礼貌模式
```bash
TARGET_GPUS="0 1 2 3 4 5 6 7"
RESERVED_GPUS=2              # 预留2张
ALLOW_SHARED_GPU=true        # 允许共享
MAX_RETRY_TIMES=3            # 有限重试
```

#### 场景3：混合模式
```bash
TARGET_GPUS="0 1 2 3"
RESERVED_GPUS=1              # 留1张
ALLOW_SHARED_GPU=false       # 独占模式（严格）
MAX_RETRY_TIMES=-1           # 无限重试
```

#### 场景4：测试/调试
```bash
TARGET_GPUS="0 1"
RESERVED_GPUS=0
ALLOW_SHARED_GPU=true        # 允许共享
MAX_RETRY_TIMES=0            # 快速失败
RUN_IN_BACKGROUND=false      # 前台运行
```

### 13.5 日志解读指南

**时间戳格式**: `[YYYY-MM-DD HH:MM:SS]`

**关键标记**:
- `✓` : 成功
- `✗` : 失败
- `•` : 列表项
- `└─` : 层级缩进

**GPU状态指标**:
- **空闲显存**: 剩余可用显存（GB）
- **利用率**: GPU核心使用率（%）
- **显存占用**: 已使用显存比例（%）

**调度状态**:
- `调度成功`: GPU已分配
- `调度失败`: 需要等待
- `已被本脚本占用`: GPU正在运行本脚本的其他命令
- `不满足独占条件`: 独占模式下GPU不空闲
- `显存不足`: 剩余显存小于需求

---

## 14. v2.2 新增功能详解

### 14.1 极限资源利用模式（EXTREME_MODE）

**配置**: `EXTREME_MODE=true`

#### 工作原理

在极限模式下，脚本会追踪每张GPU上已分配的显存，允许多个进程共享同一GPU，只要总显存不超限。

**正常模式 vs 极限模式**:

| 场景 | 正常模式 | 极限模式 |
|------|----------|----------|
| GPU已被本脚本占用 | ❌ 不再分配 | ✅ 检查显存后可继续分配 |
| GPU被其他用户占用 | 看ALLOW_SHARED_GPU | ✅ 只要显存够就分配 |
| 多进程共享GPU | ❌ 不允许 | ✅ 允许 |

#### 实际案例

**场景1**: A800 (80GB显存)，进程需要12GB

```
正常模式:
  进程1 → GPU 0 (12GB)
  进程2 → GPU 1 (12GB)
  进程3 → GPU 2 (12GB)
  ...需要多张GPU

极限模式:
  进程1-5 → 全部GPU 0 (60GB)
  仅需1张GPU！
```

**场景2**: 混合工作负载

```
GPU 0 状态:
  - 总显存: 80GB
  - 其他用户使用: 16GB
  - 剩余: 64GB

进程队列 (每个需12GB):
  极限模式: 可放5个进程 (5×12=60GB < 64GB)
  正常模式: 不放任何进程 (GPU已被占用)
```

#### 显存追踪机制

```bash
# 脚本内部维护两个追踪表
SCRIPT_GPU_OCCUPIED[gpu_id]  # 0=空闲, 1=占用
SCRIPT_GPU_USED_MEM[gpu_id]  # 已分配的显存(MB)

# 极限模式流程
1. 分配GPU: SCRIPT_GPU_USED_MEM[0] += 12GB
2. 再次分配: SCRIPT_GPU_USED_MEM[0] += 12GB (累加)
3. 释放进程: SCRIPT_GPU_USED_MEM[0] -= 12GB
4. 当SCRIPT_GPU_USED_MEM[0]=0时，GPU完全释放
```

#### 日志示例

```
[INFO] 标记 GPU 0 [极限模式] (已分配显存: 12GB, 本次新增: 12GB)
[DEBUG]   GPU 0 显存追踪: 0 MB -> 12288 MB

[INFO] 标记 GPU 0 [极限模式] (已分配显存: 24GB, 本次新增: 12GB)
[DEBUG]   GPU 0 显存追踪: 12288 MB -> 24576 MB

[INFO] 释放 GPU 0 [极限模式] (剩余已分配: 12GB, 本次释放: 12GB)
[DEBUG]   GPU 0 显存追踪: 24576 MB -> 12288 MB
```

#### 优势与风险

**优势**:
- ✅ 显著提高GPU利用率
- ✅ 减少等待时间
- ✅ 适合大量小任务
- ✅ 适合显存富余的GPU

**风险**:
- ⚠️ 进程实际用量可能超过声明
- ⚠️ GPU负载高时可能影响性能
- ⚠️ 需要准确估算显存需求

**适用场景**:
- ✓ 推理任务（显存需求小且稳定）
- ✓ 评估脚本
- ✓ 数据预处理
- ✗ 大模型训练（不推荐）
- ✗ 显存需求不确定的任务

### 14.2 TARGET_GPUS="all" 支持

**配置**: `TARGET_GPUS="all"`

#### 功能说明

自动检测服务器上所有可用GPU，无需手动指定。

**使用前**:
```bash
# 需要手动列出所有GPU
TARGET_GPUS="0 1 2 3 4 5 6 7 8 9"
```

**使用后**:
```bash
# 自动检测
TARGET_GPUS="all"
```

#### 检测过程

```bash
# 脚本启动时
[DEBUG] TARGET_GPUS='all' 检测到 0 1 2 3 4 5 6 7

# 相当于
TARGET_GPUS="0 1 2 3 4 5 6 7"
```

#### 使用场景

- ✓ 不同服务器GPU数量不同
- ✓ 动态GPU配置
- ✓ 脚本通用性
- ✓ 减少配置错误

#### 注意事项

- 需要nvidia-smi可用
- 包括所有物理GPU
- 仍受RESERVED_GPUS限制

### 14.3 调试模式（DEBUG_MODE）

**配置**: `DEBUG_MODE=true`

#### 调试信息类型

**1. GPU扫描详情**
```
[DEBUG]   模式配置: EXTREME_MODE=true, ALLOW_SHARED_GPU=true
[DEBUG]   所需显存: 35GB (35840MB)
[DEBUG]   GPU 0 详细信息:
[DEBUG]     - 总显存: 80GB
[DEBUG]     - 空闲显存: 64GB
[DEBUG]     - 利用率: 20%
[DEBUG]     - 本脚本已用: 12GB
```

**2. 显存追踪**
```
[DEBUG]   GPU 0 显存追踪: 0 MB -> 12288 MB
[DEBUG]   GPU 0 显存追踪: 12288 MB -> 24576 MB
```

**3. 命令执行**
```
[DEBUG] 执行命令: CUDA_VISIBLE_DEVICES=0 python train.py
[DEBUG] 临时日志: /path/to/temp_cmd_1_0.log
[DEBUG] 进程状态: S (sleeping)
```

**4. 错误分析**
```
[DEBUG] 失败命令: python train.py --gpu 0
[DEBUG]   最后10行输出:
    Traceback (most recent call last):
    File "train.py", line 15, in <module>
    RuntimeError: CUDA out of memory
    ...
```

### 14.4 增强的错误诊断

#### 错误类型识别

脚本自动识别以下错误类型：

| 错误类型 | 关键词 | 退出码 | 建议 |
|---------|--------|--------|------|
| 显存不足 (OOM) | "out of memory" | - | 增加显存需求或减少batch size |
| CUDA显存错误 | "cuda.*memory" | - | 检查GPU可用性 |
| CUDA运行时错误 | "cuda error" | - | 检查CUDA环境 |
| 运行时错误 | "runtime error" | - | 检查代码逻辑 |
| 段错误 | "segmentation fault" | - | 检查内存访问 |
| 模块导入错误 | "ImportError", "ModuleNotFoundError" | - | 检查Python环境和依赖 |
| 命令未找到 | - | 127 | 检查命令路径和环境变量 |
| 权限不足 | - | 126 | 检查文件执行权限 |

#### 错误输出示例

```
[ERROR] 命令执行失败 (退出码: 1, 耗时: 2分35秒)
[DEBUG] 失败命令: python train.py --gpu 0 --epochs 100
[ERROR]   └─ 错误类型: 显存不足 (OOM)
[WARNING]  └─ 建议: 增加显存需求参数或减少batch size
[DEBUG]   最后10行输出:
    RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
    (GPU 0; 79.15 GiB total capacity; 65.50 GiB already allocated;
    1.50 GiB free; 70.00 GiB reserved in total by PyTorch)
[WARNING]  └─ 等待显存释放...
```

### 14.5 配置组合建议（更新）

#### 场景1：极限性能（小任务批量处理）
```bash
TARGET_GPUS="all"             # 使用所有GPU
RESERVED_GPUS=0               # 不预留
EXTREME_MODE=true             # 极限模式
MAX_RETRY_TIMES=-1            # 无限重试
RUN_IN_BACKGROUND=true
DEBUG_MODE=false
```
**适用**: 推理、评估、数据处理

#### 场景2：保守模式（大模型训练）
```bash
TARGET_GPUS="0 1 2 3"
RESERVED_GPUS=1
EXTREME_MODE=false            # 禁用极限模式
ALLOW_SHARED_GPU=false        # 独占GPU
MAX_RETRY_TIMES=3
RUN_IN_BACKGROUND=true
DEBUG_MODE=false
```
**适用**: 大模型训练、关键任务

#### 场景3：共享服务器（礼貌模式）
```bash
TARGET_GPUS="all"
RESERVED_GPUS=2               # 预留2张
EXTREME_MODE=false
ALLOW_SHARED_GPU=true         # 允许共享
MAX_RETRY_TIMES=3
RUN_IN_BACKGROUND=true
DEBUG_MODE=false
```
**适用**: 多人共享环境

#### 场景4：调试排错
```bash
TARGET_GPUS="0 1"
RESERVED_GPUS=0
EXTREME_MODE=false
ALLOW_SHARED_GPU=true
MAX_RETRY_TIMES=1             # 有限重试
RUN_IN_BACKGROUND=false       # 前台运行
DEBUG_MODE=true               # 开启调试
```
**适用**: 问题排查、脚本开发

### 14.6 极限模式实战案例

#### 案例1: 批量推理任务

**任务**: 100个模型评估，每个需要10GB显存，每个耗时5分钟

**配置**:
```bash
TARGET_GPUS="0 1 2 3"  # 4×A800 (80GB)
EXTREME_MODE=true
RESERVED_GPUS=0
```

**资源利用**:
```
正常模式:
  - 同时运行: 4个任务
  - 总耗时: 100/4 × 5 = 125分钟

极限模式 (每GPU放7个):
  - 同时运行: 28个任务 (7×4)
  - 总耗时: 100/28 × 5 ≈ 18分钟
  - 效率提升: 7倍！
```

#### 案例2: 多数据集实验

**任务**: 10个数据集，每个需要15GB，服务器有2×A100(80GB)

**配置**:
```bash
TARGET_GPUS="0 1"
EXTREME_MODE=true
```

**分配结果**:
```
GPU 0: 5个数据集 (5×15=75GB)
GPU 1: 5个数据集 (5×15=75GB)

正常模式需要10张GPU才能同时运行
极限模式仅需2张GPU！
```

### 14.7 调试技巧

#### 技巧1: 逐步排查

```bash
# 第1步：开启调试模式
DEBUG_MODE=true

# 第2步：前台运行单个命令
RUN_IN_BACKGROUND=false
COMMANDS=(
    "your_problematic_command | 20 | 1"
)

# 第3步：查看详细输出
bash scripts/quick_start_serial.sh
```

#### 技巧2: GPU状态监控

```bash
# 终端1: 运行脚本
bash scripts/quick_start_serial.sh

# 终端2: 实时监控
watch -n 1 'nvidia-smi && echo "---" && tail -20 logs/quick_start/quick_start_serial.log'
```

#### 技巧3: 日志分析

```bash
# 查看所有错误
grep -E "\[ERROR\]|\[WARNING\]" logs/quick_start/quick_start_serial.log

# 查看GPU分配
grep "已分配GPU\|标记\|释放" logs/quick_start/quick_start_serial.log

# 查看显存追踪
grep "显存追踪" logs/quick_start/quick_start_serial.log
```

### 14.8 常见问题（更新）

#### Q1: 极限模式下进程失败率高？
**A**: 可能是显存估算不准确
```bash
# 解决方案：增加安全余量
"python train.py | 15 | 1"  # 实际需12GB，设15GB
```

#### Q2: TARGET_GPUS="all"但只检测到部分GPU？
**A**: 检查CUDA_VISIBLE_DEVICES环境变量
```bash
# 确保未设置限制
unset CUDA_VISIBLE_DEVICES
bash scripts/quick_start_serial.sh
```

#### Q3: 调试模式输出太多？
**A**: 使用grep过滤
```bash
# 只看错误和警告
bash scripts/quick_start_serial.sh 2>&1 | grep -E "ERROR|WARNING|SUCCESS"
```

#### Q4: 极限模式下GPU利用率低？
**A**: 可能是CPU瓶颈或I/O瓶颈，不是GPU问题

---

**文档结束**

