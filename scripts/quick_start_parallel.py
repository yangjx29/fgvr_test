#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import threading
import queue
import signal
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import argparse

# =============================================================================
# Configuration Section (修改这里的配置)
# =============================================================================

TARGET_GPUS = "all"  # GPU资源池 ("0 1 2 3" 或 "all")
LOG_DIR = "/home/hdl/project/fgvr_test/logs/quick_start"
RESERVED_GPUS = 1  # 预留GPU数量
MAX_RETRY_TIMES = -1  # -1=无限重试, 0=不重试, >0=最多重试N次
EXTREME_MODE = False  # 极限模式：允许多进程共享GPU
ALLOW_SHARED_GPU = True  # 允许与他人共享GPU (仅在EXTREME_MODE=False时生效)
DEBUG_MODE = True  # 调试模式
CHECK_INTERVAL = 3  # GPU检查间隔(秒)
GPU_PROCESS_CHECK_INTERVAL = 2  # GPU进程检查间隔(秒)
GPU_PROCESS_MAX_WAIT = 60  # 等待GPU进程完成的最大时间(秒)

# Commands to execute - Format: (command, memory_gb, gpu_count)
COMMANDS = [
    # 实际运行的命令
    ("bash /home/hdl/project/fgvr_test/scripts/run_fast_slow.sh eurosat --gpu ${available_gpu} --test_suffix 1", 35, 1),
    ("bash /home/hdl/project/fgvr_test/scripts/run_fast_slow.sh pet --gpu ${available_gpu} --test_suffix 1", 35, 1),
    ("bash /home/hdl/project/fgvr_test/scripts/run_fast_slow.sh bird --gpu ${available_gpu} --test_suffix 1", 35, 1),
    
    # 示例：Bash脚本
    # ("bash /path/to/script.sh --gpu ${available_gpu}", 20, 1),
    
    # 示例：Python脚本（单卡）
    # ("CUDA_VISIBLE_DEVICES=${available_gpu} python train.py --epochs 100", 18, 1),
    # ("python train.py --gpu ${available_gpu} --epochs 50", 20, 1),
    
    # 示例：Python脚本（多卡）
    # ("CUDA_VISIBLE_DEVICES=${available_gpu_0},${available_gpu_1} python train_ddp.py", 40, 2),
    # ("python train.py --gpus ${available_gpu_0},${available_gpu_1} --distributed", 40, 2),
    
    # 示例：CPU任务
    # ("python preprocess.py --input data/ --output processed/", 0, 0),
]

"""
Smart Multi-GPU Parallel Task Scheduler
智能多GPU并行任务调度器

Version: 3.0
Author: AI Assistant
Date: 2025-11-22

Features:
- Parallel execution (not serial) - tasks run simultaneously when GPUs available
- Extreme mode: multiple processes can share same GPU
- Dynamic GPU allocation with memory tracking
- Support for TARGET_GPUS="all"
- Enhanced error diagnostics
- Debug mode with detailed logging
"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Task:
    """任务信息"""
    id: int
    command: str
    memory_gb: int
    gpu_count: int
    retry_count: int = 0
    status: str = "pending"  # pending, running, completed, failed
    allocated_gpus: List[int] = None
    pid: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    exit_code: Optional[int] = None
    error_type: Optional[str] = None
    log_file: Optional[str] = None
    
    def __post_init__(self):
        if self.allocated_gpus is None:
            self.allocated_gpus = []
    
    @property
    def duration(self) -> float:
        """任务执行时长(秒)"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def command_type(self) -> str:
        """识别命令类型"""
        cmd = self.command.lower()
        if 'python' in cmd or cmd.endswith('.py'):
            return "python"
        elif 'bash' in cmd or '.sh' in cmd:
            return "bash"
        else:
            return "unknown"


@dataclass
class GPUInfo:
    """GPU信息"""
    id: int
    total_memory_mb: int
    free_memory_mb: int
    used_memory_mb: int
    utilization: int
    script_used_memory_mb: int = 0  # 本脚本已分配的显存
    occupied_by_script: bool = False  # 是否被本脚本占用(非极限模式)
    
    @property
    def memory_usage_percent(self) -> float:
        """显存占用率"""
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100
    
    @property
    def is_fully_idle(self) -> bool:
        """是否完全空闲(利用率0%且显存占用<3%)"""
        return self.utilization == 0 and self.memory_usage_percent < 3
    
    @property
    def actual_free_memory_mb(self) -> int:
        """实际可用显存(考虑本脚本已分配的)"""
        return self.free_memory_mb - self.script_used_memory_mb


# =============================================================================
# GPU Manager
# =============================================================================

class GPUManager:
    """GPU管理器"""
    
    def __init__(self, target_gpus: str, reserved_gpus: int, extreme_mode: bool, 
                 allow_shared_gpu: bool, debug_mode: bool):
        self.logger = logging.getLogger("GPUManager")
        
        self.target_gpus = self._parse_target_gpus(target_gpus)
        self.reserved_gpus = reserved_gpus
        self.extreme_mode = extreme_mode
        self.allow_shared_gpu = allow_shared_gpu
        self.debug_mode = debug_mode
        
        # GPU状态追踪
        self.gpu_lock = threading.Lock()
        self.gpu_info: Dict[int, GPUInfo] = {}
        self.task_gpu_mapping: Dict[int, List[int]] = {}  # task_id -> [gpu_ids]
    
    def _parse_target_gpus(self, target_gpus: str) -> List[int]:
        """解析目标GPU列表"""
        if target_gpus.lower() == "all":
            # 自动检测所有GPU
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    capture_output=True, text=True, check=True
                )
                gpu_ids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
                self.logger.info(f"TARGET_GPUS='all' 检测到 {len(gpu_ids)} 张GPU: {gpu_ids}")
                return gpu_ids
            except Exception as e:
                self.logger.error(f"检测GPU失败: {e}")
                sys.exit(1)
        else:
            # 手动指定GPU
            return [int(x.strip()) for x in target_gpus.split() if x.strip()]
    
    def get_gpu_info(self, gpu_id: int) -> Optional[GPUInfo]:
        """获取GPU信息"""
        try:
            # 获取显存信息
            cmd = [
                "nvidia-smi", 
                f"--id={gpu_id}",
                "--query-gpu=memory.total,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            total, free, used, util = [int(float(x.strip())) for x in result.stdout.strip().split(',')]
            
            # 获取脚本已分配的显存
            with self.gpu_lock:
                script_used = self.gpu_info.get(gpu_id, GPUInfo(gpu_id, 0, 0, 0, 0)).script_used_memory_mb
                occupied = self.gpu_info.get(gpu_id, GPUInfo(gpu_id, 0, 0, 0, 0)).occupied_by_script
            
            info = GPUInfo(
                id=gpu_id,
                total_memory_mb=total,
                free_memory_mb=free,
                used_memory_mb=used,
                utilization=util,
                script_used_memory_mb=script_used,
                occupied_by_script=occupied
            )
            
            return info
        except Exception as e:
            self.logger.error(f"获取GPU {gpu_id} 信息失败: {e}")
            return None
    
    def refresh_gpu_info(self):
        """刷新所有GPU信息"""
        with self.gpu_lock:
            for gpu_id in self.target_gpus:
                info = self.get_gpu_info(gpu_id)
                if info:
                    # 保留脚本追踪的信息
                    if gpu_id in self.gpu_info:
                        info.script_used_memory_mb = self.gpu_info[gpu_id].script_used_memory_mb
                        info.occupied_by_script = self.gpu_info[gpu_id].occupied_by_script
                    self.gpu_info[gpu_id] = info
    
    def get_available_gpus(self, required_memory_mb: int, required_count: int) -> List[int]:
        """获取可用GPU列表"""
        self.refresh_gpu_info()
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info(f"[{current_time}] 开始扫描可用GPU...")
        self.logger.debug(f"  模式配置: EXTREME_MODE={self.extreme_mode}, ALLOW_SHARED_GPU={self.allow_shared_gpu}")
        self.logger.debug(f"  所需显存: {required_memory_mb//1024}GB ({required_memory_mb}MB)")
        
        available = []
        gpu_mem_map = {}  # {gpu_id: free_memory_mb}
        
        with self.gpu_lock:
            for gpu_id in self.target_gpus:
                info = self.gpu_info.get(gpu_id)
                if not info:
                    continue
                
                self.logger.debug(f"  GPU {gpu_id} 详细信息:")
                self.logger.debug(f"    - 总显存: {info.total_memory_mb//1024}GB")
                self.logger.debug(f"    - 空闲显存: {info.free_memory_mb//1024}GB")
                self.logger.debug(f"    - 利用率: {info.utilization}%")
                self.logger.debug(f"    - 本脚本已用: {info.script_used_memory_mb//1024}GB")
                
                # 极限模式
                if self.extreme_mode:
                    actual_free = info.actual_free_memory_mb
                    if actual_free >= required_memory_mb:
                        available.append(gpu_id)
                        gpu_mem_map[gpu_id] = actual_free
                        self.logger.info(
                            f"  └─ GPU {gpu_id}: 可用 [极限模式] "
                            f"(实际空闲:{actual_free//1024}GB, 系统空闲:{info.free_memory_mb//1024}GB, "
                            f"本脚本占用:{info.script_used_memory_mb//1024}GB, 利用率:{info.utilization}%)"
                        )
                    else:
                        self.logger.info(
                            f"  └─ GPU {gpu_id}: 显存不足 [极限模式] "
                            f"(实际空闲:{actual_free//1024}GB < 需求:{required_memory_mb//1024}GB)"
                        )
                    continue
                
                # 非极限模式：检查是否被本脚本占用
                if info.occupied_by_script:
                    self.logger.info(f"  └─ GPU {gpu_id}: 已被本脚本占用，跳过")
                    self.logger.debug(f"    - 本脚本已在该GPU上分配了 {info.script_used_memory_mb//1024}GB 显存")
                    continue
                
                # 独占模式检查
                if not self.allow_shared_gpu:
                    if not info.is_fully_idle:
                        self.logger.info(
                            f"  └─ GPU {gpu_id}: 不满足独占条件 "
                            f"(利用率:{info.utilization}%, 显存占用:{info.memory_usage_percent:.1f}%), 跳过"
                        )
                        continue
                
                # 检查显存
                if info.free_memory_mb >= required_memory_mb:
                    available.append(gpu_id)
                    gpu_mem_map[gpu_id] = info.free_memory_mb
                    self.logger.info(
                        f"  └─ GPU {gpu_id}: 可用 "
                        f"(空闲:{info.free_memory_mb//1024}GB, 利用率:{info.utilization}%, "
                        f"显存占用:{info.memory_usage_percent:.1f}%)"
                    )
                else:
                    self.logger.info(
                        f"  └─ GPU {gpu_id}: 显存不足 "
                        f"(空闲:{info.free_memory_mb//1024}GB < 需求:{required_memory_mb//1024}GB)"
                    )
        
        # 按剩余显存降序排序
        if available:
            available.sort(key=lambda x: gpu_mem_map.get(x, 0), reverse=True)
            self.logger.info(f"  └─ 找到 {len(available)} 张可用GPU（已按剩余显存降序排列）: {available}")
        else:
            self.logger.warning("  └─ 未找到满足条件的可用GPU")
        
        # 应用预留策略
        available_count = len(available)
        if available_count == 0:
            self.logger.info("  └─ 预留策略: 无可用GPU")
            return []
        elif available_count == self.reserved_gpus:
            self.logger.info(f"  └─ 预留策略: 可用数={available_count} = 预留数={self.reserved_gpus}，使用1张")
            return available[:1] if required_count <= 1 else []
        elif available_count > self.reserved_gpus:
            usable_count = available_count - self.reserved_gpus
            self.logger.info(f"  └─ 预留策略: 可用数={available_count} > 预留数={self.reserved_gpus}，可使用{usable_count}张")
            return available[:min(usable_count, required_count)]
        else:
            self.logger.warning(f"  └─ 预留策略: 可用数={available_count} < 预留数={self.reserved_gpus}，不分配")
            return []
    
    def allocate_gpus(self, task: Task) -> Optional[List[int]]:
        """为任务分配GPU"""
        if task.gpu_count == 0:
            return []  # CPU任务
        
        required_memory_mb = task.memory_gb * 1024
        
        self.logger.info("━" * 70)
        self.logger.info("开始GPU分配流程")
        self.logger.info(f"  任务ID: {task.id}")
        self.logger.info(f"  需求: {task.gpu_count}张GPU，每张需{task.memory_gb}GB显存")
        self.logger.info(f"  极限模式: {'✓ 启用（允许多进程共享GPU）' if self.extreme_mode else '✗ 禁用'}")
        if not self.extreme_mode:
            self.logger.info(f"  独占模式: {'✓ 启用（仅用完全空闲GPU）' if not self.allow_shared_gpu else '✗ 禁用'}")
        self.logger.info(f"  预留GPU数: {self.reserved_gpus}")
        self.logger.info("━" * 70)
        
        available = self.get_available_gpus(required_memory_mb, task.gpu_count)
        
        if len(available) >= task.gpu_count:
            allocated = available[:task.gpu_count]
            self.logger.info("")
            self.logger.info(f"✓ 调度成功！分配 {task.gpu_count} 张GPU: {allocated}")
            
            # 显示每张GPU详情
            for gpu_id in allocated:
                info = self.gpu_info.get(gpu_id)
                if info:
                    self.logger.info(f"  • GPU {gpu_id}: {info.free_memory_mb//1024}GB可用, 利用率{info.utilization}%")
            
            return allocated
        else:
            self.logger.warning(f"✗ 调度失败：需要 {task.gpu_count} 张，可分配 {len(available)} 张")
            return None
    
    def mark_gpus_occupied(self, task_id: int, gpu_ids: List[int], memory_mb: int):
        """标记GPU为被本脚本占用"""
        with self.gpu_lock:
            self.task_gpu_mapping[task_id] = gpu_ids
            
            for gpu_id in gpu_ids:
                if gpu_id not in self.gpu_info:
                    self.gpu_info[gpu_id] = GPUInfo(gpu_id, 0, 0, 0, 0)
                
                if self.extreme_mode:
                    # 极限模式：累加显存
                    current_used = self.gpu_info[gpu_id].script_used_memory_mb
                    self.gpu_info[gpu_id].script_used_memory_mb = current_used + memory_mb
                    new_used_gb = self.gpu_info[gpu_id].script_used_memory_mb // 1024
                    self.logger.info(
                        f"标记 GPU {gpu_id} [极限模式] "
                        f"(已分配显存: {new_used_gb}GB, 本次新增: {memory_mb//1024}GB)"
                    )
                    self.logger.debug(f"  GPU {gpu_id} 显存追踪: {current_used} MB -> {self.gpu_info[gpu_id].script_used_memory_mb} MB")
                else:
                    # 正常模式：标记为占用
                    self.gpu_info[gpu_id].occupied_by_script = True
                    self.gpu_info[gpu_id].script_used_memory_mb = memory_mb
                    self.logger.info(f"标记 GPU {gpu_id} 为占用 (分配显存: {memory_mb//1024}GB)")
    
    def release_gpus(self, task_id: int, memory_mb: int):
        """释放GPU"""
        with self.gpu_lock:
            gpu_ids = self.task_gpu_mapping.get(task_id, [])
            
            for gpu_id in gpu_ids:
                if gpu_id not in self.gpu_info:
                    continue
                
                if self.extreme_mode:
                    # 极限模式：减少显存
                    current_used = self.gpu_info[gpu_id].script_used_memory_mb
                    self.gpu_info[gpu_id].script_used_memory_mb = max(0, current_used - memory_mb)
                    remaining_gb = self.gpu_info[gpu_id].script_used_memory_mb // 1024
                    self.logger.info(
                        f"释放 GPU {gpu_id} [极限模式] "
                        f"(剩余已分配: {remaining_gb}GB, 本次释放: {memory_mb//1024}GB)"
                    )
                    self.logger.debug(f"  GPU {gpu_id} 显存追踪: {current_used} MB -> {self.gpu_info[gpu_id].script_used_memory_mb} MB")
                else:
                    # 正常模式：标记为空闲
                    self.gpu_info[gpu_id].occupied_by_script = False
                    self.gpu_info[gpu_id].script_used_memory_mb = 0
                    self.logger.info(f"释放 GPU {gpu_id}")
            
            if task_id in self.task_gpu_mapping:
                del self.task_gpu_mapping[task_id]
    
    def wait_for_gpu_processes(self, gpu_ids: List[int], timeout: int = 60) -> bool:
        """等待GPU上的所有进程完成"""
        if not gpu_ids:
            return True
        
        start_time = time.time()
        for gpu_id in gpu_ids:
            self.logger.info(f"等待 GPU {gpu_id} 上的进程完成...")
            
            while time.time() - start_time < timeout:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", f"-i{gpu_id}"],
                        capture_output=True, text=True, check=True, timeout=5
                    )
                    pids = [line.strip() for line in result.stdout.strip().split('\n') if line.strip() and line.strip() != 'No running']
                    
                    if not pids:
                        self.logger.info(f"  GPU {gpu_id} 上的所有进程已完成")
                        break
                    
                    time.sleep(GPU_PROCESS_CHECK_INTERVAL)
                except Exception as e:
                    self.logger.warning(f"  检查GPU {gpu_id} 进程时出错: {e}")
                    break
            
            if time.time() - start_time >= timeout:
                self.logger.warning(f"  等待GPU {gpu_id} 进程超时，继续执行")
                return False
        
        return True


# =============================================================================
# Task Executor
# =============================================================================

class TaskExecutor:
    """任务执行器"""
    
    def __init__(self, gpu_manager: GPUManager, log_dir: str, max_retry: int):
        self.gpu_manager = gpu_manager
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_retry = max_retry
        
        self.logger = logging.getLogger("TaskExecutor")
        
        # 错误类型映射
        self.error_patterns = {
            "显存不足 (OOM)": r"out of memory",
            "CUDA显存错误": r"cuda.*memory",
            "CUDA运行时错误": r"cuda error",
            "运行时错误": r"runtime error",
            "段错误": r"segmentation fault",
            "模块导入错误": r"importerror|modulenotfounderror",
        }
        
        self.error_suggestions = {
            "显存不足 (OOM)": "增加显存需求参数或减少batch size",
            "CUDA显存错误": "检查GPU可用性",
            "CUDA运行时错误": "检查CUDA环境",
            "运行时错误": "检查代码逻辑",
            "段错误": "检查内存访问",
            "模块导入错误": "检查Python环境和依赖",
        }
    
    def replace_gpu_variables(self, command: str, gpu_ids: List[int]) -> str:
        """替换命令中的GPU变量"""
        if not gpu_ids:
            return command
        
        # 替换 ${available_gpu}
        cmd = command.replace("${available_gpu}", str(gpu_ids[0]))
        
        # 替换 ${available_gpu_0}, ${available_gpu_1}, ...
        for i, gpu_id in enumerate(gpu_ids):
            cmd = cmd.replace(f"${{available_gpu_{i}}}", str(gpu_id))
        
        return cmd
    
    def identify_error_type(self, log_file: Path, exit_code: int) -> Tuple[str, str]:
        """识别错误类型"""
        if not log_file.exists():
            return "未知错误", ""
        
        try:
            log_content = log_file.read_text().lower()
            
            # 检查错误模式
            for error_type, pattern in self.error_patterns.items():
                if re.search(pattern, log_content, re.IGNORECASE):
                    suggestion = self.error_suggestions.get(error_type, "")
                    return error_type, suggestion
            
            # 根据退出码判断
            if exit_code == 127:
                return "命令未找到", "检查命令路径和环境变量"
            elif exit_code == 126:
                return "权限不足", "检查文件执行权限"
            
            return "未知错误", ""
        except Exception as e:
            self.logger.error(f"识别错误类型时出错: {e}")
            return "未知错误", ""
    
    def execute_task(self, task: Task) -> bool:
        """执行任务"""
        # 替换GPU变量
        actual_command = self.replace_gpu_variables(task.command, task.allocated_gpus)
        
        # 创建日志文件(带时间戳避免覆盖)
        timestamp = int(time.time())
        log_file = self.log_dir / f"task_{task.id}_retry_{task.retry_count}_{timestamp}.log"
        task.log_file = str(log_file)
        
        self.logger.info(f"实际执行命令: {actual_command}")
        self.logger.debug(f"日志文件: {log_file}")
        
        # 执行命令
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    actual_command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid  # 创建新进程组
                )
                
                task.pid = process.pid
                task.start_time = time.time()
                task.status = "running"
                
                self.logger.info(f"任务 {task.id} 已启动 (PID: {task.pid})")
                
                # 等待进程完成
                exit_code = process.wait()
                
                task.end_time = time.time()
                task.exit_code = exit_code
                
                # 检查是否成功
                if exit_code == 0 and not self._has_error_in_log(log_file):
                    task.status = "completed"
                    duration_min = int(task.duration // 60)
                    duration_sec = int(task.duration % 60)
                    self.logger.info(f"✓ 任务 {task.id} 执行成功 (耗时: {duration_min}分{duration_sec}秒)")
                    return True
                else:
                    task.status = "failed"
                    error_type, suggestion = self.identify_error_type(log_file, exit_code)
                    task.error_type = error_type
                    
                    duration_min = int(task.duration // 60)
                    duration_sec = int(task.duration % 60)
                    self.logger.error(f"✗ 任务 {task.id} 执行失败 (退出码: {exit_code}, 耗时: {duration_min}分{duration_sec}秒)")
                    self.logger.debug(f"失败命令: {actual_command}")
                    self.logger.error(f"  └─ 错误类型: {error_type}")
                    if suggestion:
                        self.logger.warning(f"  └─ 建议: {suggestion}")
                    
                    # 显示最后10行日志
                    if DEBUG_MODE and log_file.exists():
                        try:
                            lines = log_file.read_text().strip().split('\n')
                            last_lines = lines[-10:] if len(lines) > 10 else lines
                            if last_lines:
                                self.logger.debug("  最后10行输出:")
                                for line in last_lines:
                                    self.logger.debug(f"    {line}")
                        except:
                            pass
                    
                    return False
                    
        except Exception as e:
            task.status = "failed"
            task.end_time = time.time()
            self.logger.error(f"✗ 任务 {task.id} 执行异常: {e}")
            return False
    
    def _has_error_in_log(self, log_file: Path) -> bool:
        """检查日志中是否有错误"""
        if not log_file.exists():
            return False
        
        try:
            content = log_file.read_text().lower()
            error_keywords = ["out of memory", "cuda error", "runtime error", "segmentation fault"]
            return any(keyword in content for keyword in error_keywords)
        except:
            return False


# =============================================================================
# Task Scheduler
# =============================================================================

class TaskScheduler:
    """任务调度器 - 并行执行"""
    
    def __init__(self, tasks: List[Task], gpu_manager: GPUManager, 
                 executor: TaskExecutor, max_retry: int, check_interval: int):
        self.tasks = tasks
        self.gpu_manager = gpu_manager
        self.executor = executor
        self.max_retry = max_retry
        self.check_interval = check_interval
        
        self.running_tasks: Dict[int, Task] = {}  # task_id -> Task
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        self.lock = threading.Lock()
        self.running = True
        
        self.logger = logging.getLogger("TaskScheduler")
    
    def schedule(self):
        """并行调度执行所有任务"""
        self.logger.info("=" * 70)
        self.logger.info(f"共有 {len(self.tasks)} 个任务待执行（并行模式）")
        self.logger.info("=" * 70)
        
        # 按GPU数量排序（多卡优先）
        self.tasks.sort(key=lambda t: t.gpu_count, reverse=True)
        
        # 显示任务列表
        for i, task in enumerate(self.tasks, 1):
            self.logger.info(f"  [{i}] {task.command}")
            if task.gpu_count == 0:
                self.logger.info(f"      └─ CPU任务（无需GPU）[类型: {task.command_type}]")
            else:
                self.logger.info(f"      └─ 所需显存: {task.memory_gb}GB × {task.gpu_count}卡 [类型: {task.command_type}]")
        
        self.logger.info("")
        start_time = time.time()
        self.logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 70)
        self.logger.info("")
        
        # 启动任务监控线程
        monitor_thread = threading.Thread(target=self._monitor_tasks, daemon=True)
        monitor_thread.start()
        
        # 主调度循环
        pending_tasks = list(self.tasks)
        
        while pending_tasks or self.running_tasks:
            # 尝试启动新任务
            tasks_to_start = []
            
            for task in pending_tasks[:]:
                if self._try_start_task(task):
                    tasks_to_start.append(task)
                    pending_tasks.remove(task)
            
            # 如果有任务启动，稍等一下让它们运行
            if tasks_to_start:
                time.sleep(1)
            
            # 等待一段时间再检查
            time.sleep(self.check_interval)
            
            # 检查是否需要停止
            if not self.running:
                break
        
        # 等待所有任务完成
        while self.running_tasks:
            time.sleep(1)
        
        end_time = time.time()
        
        # 输出统计信息
        self._print_summary(start_time, end_time)
    
    def _try_start_task(self, task: Task) -> bool:
        """尝试启动任务"""
        # 检查重试次数
        if self.max_retry != -1 and task.retry_count > self.max_retry:
            self.logger.error(f"任务 {task.id} 已达到最大重试次数 ({self.max_retry})，放弃")
            self.failed_tasks.append(task)
            return False
        
        # 分配GPU
        if task.gpu_count == 0:
            # CPU任务，直接执行
            allocated_gpus = []
        else:
            allocated_gpus = self.gpu_manager.allocate_gpus(task)
            if allocated_gpus is None:
                return False
        
        task.allocated_gpus = allocated_gpus
        
        # 标记GPU占用
        if allocated_gpus:
            memory_mb = task.memory_gb * 1024
            self.gpu_manager.mark_gpus_occupied(task.id, allocated_gpus, memory_mb)
        
        # 在新线程中执行任务
        with self.lock:
            self.running_tasks[task.id] = task
        
        thread = threading.Thread(target=self._run_task, args=(task,), daemon=True)
        thread.start()
        
        return True
    
    def _run_task(self, task: Task):
        """运行任务（在独立线程中）"""
        try:
            success = self.executor.execute_task(task)
            
            # 等待GPU进程完成
            if task.allocated_gpus:
                self.gpu_manager.wait_for_gpu_processes(task.allocated_gpus, GPU_PROCESS_MAX_WAIT)
            
            # 释放GPU
            if task.allocated_gpus:
                memory_mb = task.memory_gb * 1024
                self.gpu_manager.release_gpus(task.id, memory_mb)
            
            # 更新任务状态
            with self.lock:
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
                
                if success:
                    self.completed_tasks.append(task)
                else:
                    # 失败：重新加入队列重试
                    if self.max_retry == -1 or task.retry_count < self.max_retry:
                        task.retry_count += 1
                        task.status = "pending"
                        task.allocated_gpus = []
                        self.tasks.append(task)
                        
                        if self.max_retry == -1:
                            self.logger.warning(f"任务 {task.id} 将重试（第 {task.retry_count} 次，无限重试模式）")
                        else:
                            self.logger.warning(f"任务 {task.id} 将重试（第 {task.retry_count} 次，最多 {self.max_retry} 次）")
                    else:
                        self.failed_tasks.append(task)
        
        except Exception as e:
            self.logger.error(f"运行任务 {task.id} 时出错: {e}")
            with self.lock:
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
                self.failed_tasks.append(task)
    
    def _monitor_tasks(self):
        """监控任务执行"""
        while self.running:
            time.sleep(10)
            
            with self.lock:
                if self.running_tasks:
                    self.logger.info(f"当前运行中的任务: {len(self.running_tasks)}")
                    for task in self.running_tasks.values():
                        elapsed = time.time() - task.start_time if task.start_time else 0
                        self.logger.info(f"  - 任务 {task.id}: PID={task.pid}, GPU={task.allocated_gpus}, 已运行{int(elapsed)}秒")
    
    def _print_summary(self, start_time: float, end_time: float):
        """打印执行总结"""
        total_duration = end_time - start_time
        total_duration_min = int(total_duration // 60)
        total_duration_sec = int(total_duration % 60)
        
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        success_count = len(self.completed_tasks)
        failed_count = len(self.failed_tasks)
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("所有任务执行完成")
        self.logger.info("=" * 70)
        self.logger.info("")
        self.logger.info("执行统计：")
        self.logger.info(f"  • GPU资源池: {self.gpu_manager.target_gpus}")
        self.logger.info(f"  • 总任务数: {total_tasks}")
        self.logger.info(f"  • 成功: {success_count}")
        self.logger.info(f"  • 失败: {failed_count}")
        if total_tasks > 0:
            self.logger.info(f"  • 成功率: {(success_count/total_tasks)*100:.1f}%")
        self.logger.info(f"  • 总耗时: {total_duration_min}分{total_duration_sec}秒")
        if total_tasks > 0:
            self.logger.info(f"  • 平均每任务: {total_duration/total_tasks:.1f}秒")
        self.logger.info(f"  • 开始时间: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"  • 结束时间: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")
        
        # 显示失败任务
        if self.failed_tasks:
            self.logger.error("失败的任务:")
            for task in self.failed_tasks:
                self.logger.error(f"  - 任务 {task.id}: {task.command}")
                self.logger.error(f"    错误: {task.error_type or '未知'}, 重试次数: {task.retry_count}")
        
        self.logger.info("=" * 70)


# =============================================================================
# Main Function
# =============================================================================

def setup_logging(log_dir: str, debug_mode: bool):
    """设置日志"""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    base_name = "quick_start_parallel"
    log_file = log_dir_path / f"{base_name}.log"
    
    counter = 1
    while log_file.exists():
        log_file = log_dir_path / f"{base_name}({counter}).log"
        counter += 1
    
    # 配置日志
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file


def main():
    """主函数"""
    # 设置日志
    log_file = setup_logging(LOG_DIR, DEBUG_MODE)
    logger = logging.getLogger("Main")
    
    logger.info("=" * 70)
    logger.info("智能多GPU并行任务调度器 - Smart Multi-GPU Parallel Task Scheduler")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"GPU资源池: {TARGET_GPUS}")
    logger.info(f"预留GPU数: {RESERVED_GPUS}")
    logger.info(f"显存检查间隔: {CHECK_INTERVAL}秒")
    logger.info(f"调试模式: {'✓ 启用（详细输出）' if DEBUG_MODE else '✗ 禁用'}")
    logger.info("")
    logger.info("═══ GPU调度策略 ═══")
    logger.info(f"极限资源利用模式: {'✓ 启用' if EXTREME_MODE else '✗ 禁用'}")
    if EXTREME_MODE:
        logger.warning("  └─ 极限模式：允许多进程共享GPU，只要有足够显存就调度")
    else:
        logger.info(f"GPU共享模式: {'允许共享' if ALLOW_SHARED_GPU else '独占模式'}")
        if not ALLOW_SHARED_GPU:
            logger.warning("  └─ 独占模式：仅使用利用率0%且显存占用<3%的GPU")
    
    if MAX_RETRY_TIMES == -1:
        logger.info("重试策略: 无限重试")
    else:
        logger.info(f"最大重试次数: {MAX_RETRY_TIMES}")
    logger.info("GPU选择策略: 优先选择剩余显存多的GPU")
    logger.info("执行模式: 并行执行（非串行）")
    logger.info("")
    
    # 初始化组件
    gpu_manager = GPUManager(TARGET_GPUS, RESERVED_GPUS, EXTREME_MODE, ALLOW_SHARED_GPU, DEBUG_MODE)
    executor = TaskExecutor(gpu_manager, LOG_DIR, MAX_RETRY_TIMES)
    
    # 创建任务
    tasks = []
    for i, (cmd, mem_gb, gpu_count) in enumerate(COMMANDS, 1):
        task = Task(
            id=i,
            command=cmd,
            memory_gb=mem_gb,
            gpu_count=gpu_count
        )
        tasks.append(task)
    
    # 调度执行
    scheduler = TaskScheduler(tasks, gpu_manager, executor, MAX_RETRY_TIMES, CHECK_INTERVAL)
    
    try:
        scheduler.schedule()
    except KeyboardInterrupt:
        logger.warning("\n收到中断信号，正在停止...")
        scheduler.running = False
    except Exception as e:
        logger.error(f"执行过程中出错: {e}", exc_info=True)
    
    # 返回退出码
    if scheduler.failed_tasks:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

