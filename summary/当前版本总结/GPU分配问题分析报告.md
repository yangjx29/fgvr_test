# GPU 分配问题分析报告

**日期**: 2025-11-30  
**问题**: 所有任务都被分配到 GPU 8，而非调度器日志显示的 GPU 4/6/7

---

## 问题现象

1. `compete_gpus_retry.py` 日志显示任务分配到 GPU 4、6、7
2. 实际 pipeline 日志显示所有任务都在 GPU 8 上运行
3. `nvidia-smi` 确认进程都在 GPU 8 上

**日志对比**:
```
# compete_gpu 日志 (错误分配)
🚀 Starting task compete_789806 (Queue 5) on GPU 4
🚀 Starting task compete_576803 (Queue 4) on GPU 6
🚀 Starting task compete_321917 (Queue 3) on GPU 7

# pipeline 日志 (实际运行)
GPU: CUDA_VISIBLE_DEVICES=8  # 全部是 8
```

---

## 根本原因

### 执行流分析

```
compete_gpus_retry.py
    │
    ├── execute_task(task, gpu_id=4)
    │       │
    │       └── full_cmd = "CUDA_VISIBLE_DEVICES=4 bash run_pipeline.sh ..."
    │               │
    │               └── subprocess.run(full_cmd, shell=True)
    │                       │
    │                       └── bash run_pipeline.sh
    │                               │
    │                               ├── 第175行: CUDA_VISIBLE_DEVICES=$(get_yaml_value ...)
    │                               │   ↑↑↑ 问题在这里！从 config.yaml 读取并覆盖环境变量
    │                               │
    │                               └── config.yaml: cuda_visible_devices: "8"
    │                                   ↑↑↑ 硬编码为 8
```

### 问题代码

**`run_pipeline.sh` 第 175 行 (修复前)**:
```bash
CUDA_VISIBLE_DEVICES=$(get_yaml_value "cuda_visible_devices" "${CONFIG_FILE}")
```

这行代码**无条件**从 `config.yaml` 读取 GPU 配置，覆盖了环境变量。

**`config.yaml`**:
```yaml
cuda_visible_devices: "8"  # 硬编码
```

---

## 验证测试

### 测试 1: 直接使用环境变量前缀

```bash
CUDA_VISIBLE_DEVICES=4 python3 gpu_test_worker.py
```

**结果**: ✅ 正确分配到 GPU 4

### 测试 2: 通过 subprocess 传递

```python
cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python3 gpu_test_worker.py"
subprocess.run(cmd, shell=True)
```

**结果**: ✅ 正确分配到指定 GPU

### 测试 3: 调用 run_pipeline.sh

```python
cmd = f"CUDA_VISIBLE_DEVICES=4 bash run_pipeline.sh ..."
subprocess.run(cmd, shell=True)
```

**结果**: ❌ 分配到 GPU 8 (被 config.yaml 覆盖)

---

## 修复方案

### 修改 `run_pipeline.sh`

**修复后 (第 175-178 行)**:
```bash
# 优先使用环境变量中的 CUDA_VISIBLE_DEVICES，否则从 YAML 读取
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    CUDA_VISIBLE_DEVICES=$(get_yaml_value "cuda_visible_devices" "${CONFIG_FILE}")
fi
```

**优先级**: 环境变量 > 命令行参数 `--gpu` > YAML 配置

---

## 其他已实现功能

### 1. 顺序分配机制 ✅
- 每次只分配一个任务
- 等待进程确认启动 (检查 JSON 中的 PID)
- 30 秒延迟后分配下一个

### 2. 重试机制 ✅
- 检测 `abnormal_exit` 状态
- 自动重试，生成新的 `uni_id`
- 每 3 次重试后退避 10 分钟

### 3. GPU 检测 ✅
- 实时检测 GPU 可用显存
- 非极限模式下检测用户 Python 进程占用

---

## 总结

| 组件 | 状态 | 说明 |
|------|------|------|
| `compete_gpus_retry.py` GPU 分配 | ✅ | 正确设置 `CUDA_VISIBLE_DEVICES` |
| 环境变量传递 | ✅ | subprocess 正确传递 |
| `run_pipeline.sh` 接收 | ❌→✅ | **已修复**: 优先使用环境变量 |
| 顺序分配 | ✅ | 每次一个任务 + 30s 延迟 |
| 重试机制 | ✅ | 3 次重试后退避 10 分钟 |

**根本原因**: `run_pipeline.sh` 无条件从 `config.yaml` 读取 GPU 配置，覆盖了环境变量。

**修复**: 修改 `run_pipeline.sh`，优先使用环境变量中的 `CUDA_VISIBLE_DEVICES`。
