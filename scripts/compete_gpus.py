#!/usr/bin/env python3
"""
GPU Competition Script
Manages GPU resource allocation based on available memory
"""

import os
import sys
import time
import subprocess
import threading
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
import json
import psutil
import uuid
import random

# 运行 
# nohup python compete_gpus.py > /dev/null 2>&1 & 
# 即可

# Configuration
work_dir = '/home/hdl/project/fgvr_test_new'
log_dir = f'{work_dir}/logs/compete_gpu'
process_json = f'{work_dir}/scripts/script_process/uni_id.json'
check_time = 5
maximize_resource_utilization = False # 极限利用资源模式，如果开启允许当前用户的多进程放同一GPU，否则当前用户只能放一个进程到同一GPU
# 支持单命令和多命令串行执行
# compete_gpus 始终是显式 GPU 列表，例如 [0,1,2,3]；当 use_all_gpus=True 时，该列表会被自动忽略
compete_gpus = [0,1,2,3,4,5,6,7,8,9] # 当前要竞争的GPU列表，只有在 use_all_gpus=False 时生效
use_all_gpus = True # 是否自动使用所有可见GPU（优先 CUDA_VISIBLE_DEVICES，否则使用 nvidia-smi 探测）
# 注意：GPU 分配通过 CUDA_VISIBLE_DEVICES 环境变量在执行时动态设置，无需在命令中指定

# 重试配置
max_retry_before_backoff = 3  # 每 3 次重试后进入退避
backoff_duration = 600        # 退避时间（秒），10 分钟

# 三元组（待执行命令列表，队列ID，估计显存）
command_tasks=[
            # 第一个队列，汽车数据集
            (
                [
                    "rm -rf {work_dir}/experiments/car196/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 8",
                    "bash {work_dir}/scripts/run_pipeline.sh car --uni_id {uni_id}"
                ],
                1,
                20
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/car196/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 5",
                    "bash {work_dir}/scripts/run_pipeline.sh car --uni_id {uni_id}"
                ],
                1,
                20
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/car196/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 3",
                    "bash {work_dir}/scripts/run_pipeline.sh car --uni_id {uni_id}"
                ],
                1,
                20
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/car196/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 20",
                    "bash {work_dir}/scripts/run_pipeline.sh car --uni_id {uni_id}"
                ],
                1,
                20
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/car196/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 50",
                    "bash {work_dir}/scripts/run_pipeline.sh car --uni_id {uni_id}"
                ],
                1,
                20
            ),
            # 第二个队列，花数据集
            (
                [
                    "rm -rf {work_dir}/experiments/flower102/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 15",
                    "bash {work_dir}/scripts/run_pipeline.sh flower --uni_id {uni_id}"
                ],
                2,
                20
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/flower102/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 5",
                    "bash {work_dir}/scripts/run_pipeline.sh flower --uni_id {uni_id}"
                ],
                2,
                20
            ),
            # 第三个队列，鸟数据集
            (
                [
                    "rm -rf {work_dir}/experiments/bird200/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 50",
                    "bash {work_dir}/scripts/run_pipeline.sh bird --uni_id {uni_id}"
                ],
                3,
                20
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/bird200/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 50",
                    "bash {work_dir}/scripts/run_pipeline.sh bird --uni_id {uni_id}"
                ],
                3,
                20
            ),
            # 第四个队列,imagenet_v2
            (
                [
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 8",
                    "bash {work_dir}/scripts/run_pipeline.sh imagenet_v2 --uni_id {uni_id}"
                ],
                4,
                20
            ),
            #第五个队列,狗数据集
            (
                [
                    "rm -rf {work_dir}/experiments/dog120/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 5",
                    "bash {work_dir}/scripts/run_pipeline.sh dog --uni_id {uni_id}"
                ],
                5,
                20
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/dog120/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 15",
                    "bash {work_dir}/scripts/run_pipeline.sh dog --uni_id {uni_id}"
                ],
                5,
                20
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/dog120/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 3",
                    "bash {work_dir}/scripts/run_pipeline.sh dog --uni_id {uni_id}"
                ],
                5,
                20
            ),
            #第六个队列，sun
            (
                [
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 8",
                    "bash {work_dir}/scripts/run_pipeline.sh sun397 --uni_id {uni_id}"
                ],
                6,
                20
            ),
        ]


# =============================================================================
# 核心类定义
# =============================================================================

@dataclass
class Task:
    """任务数据结构"""
    commands: List[str]          # 命令列表（串行执行）
    queue_id: int                # 队列 ID
    estimated_memory_gb: int     # 预估显存 (GB)
    uni_id: str = ""             # 唯一标识符
    status: str = "pending"      # pending / running / completed / failed
    assigned_gpu: int = -1       # 分配的 GPU ID
    pid: int = 0                 # Python 进程 PID
    retry_count: int = 0         # 重试次数
    backoff_until: float = 0     # 退避结束时间戳（0 表示无退避）
    error_type: str = ""         # 错误类型


class GPUMonitor:
    """GPU 状态监控"""
    
    @staticmethod
    def get_available_memory(gpu_id: int) -> float:
        """获取指定 GPU 的可用显存 (GB)"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', f'--id={gpu_id}'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return float(result.stdout.strip()) / 1024  # MB -> GB
        except Exception as e:
            logging.debug(f"Failed to get memory for GPU {gpu_id}: {e}")
        return 0.0
    
    @staticmethod
    def get_user_processes_on_gpu(gpu_id: int) -> List[int]:
        """获取当前用户在指定 GPU 上的 Python 进程 PID 列表"""
        pids = []
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits', f'--id={gpu_id}'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                current_user = os.getenv('USER', '')
                for line in result.stdout.strip().split('\n'):
                    try:
                        pid = int(line.strip())
                        proc = psutil.Process(pid)
                        if proc.username() == current_user and 'python' in proc.name().lower():
                            pids.append(pid)
                    except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
        except Exception as e:
            logging.debug(f"Failed to get processes for GPU {gpu_id}: {e}")
        return pids
    
    @staticmethod
    def detect_gpus() -> List[int]:
        """检测可用的 GPU 列表"""
        # 优先使用 CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            try:
                return [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
            except ValueError:
                pass
        # 否则使用 nvidia-smi 探测
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
        except Exception:
            pass
        return []


class ProcessJSON:
    """管理 uni_id.json 文件"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self._ensure_exists()
    
    def _ensure_exists(self):
        """确保 JSON 文件存在"""
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        if not os.path.exists(self.json_path):
            with open(self.json_path, 'w') as f:
                json.dump({}, f)
    
    def load(self) -> dict:
        """安全加载 JSON"""
        try:
            with open(self.json_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {}
                data = json.loads(content)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            logging.warning(f"Failed to load JSON: {e}")
            return {}
    
    def save(self, data: dict):
        """安全保存 JSON"""
        try:
            with open(self.json_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save JSON: {e}")
    
    def get_record(self, uni_id: str) -> Optional[dict]:
        """获取指定 uni_id 的记录"""
        data = self.load()
        return data.get(uni_id)
    
    def update_record(self, uni_id: str, pid: int, state: str, error_type: str = None):
        """更新记录"""
        data = self.load()
        if uni_id not in data:
            data[uni_id] = {'retry_count': 0}
        data[uni_id]['pid'] = pid
        data[uni_id]['state'] = state
        if error_type:
            data[uni_id]['error_type'] = error_type
        elif state == 'normal_exit' and 'error_type' in data[uni_id]:
            del data[uni_id]['error_type']
        self.save(data)
    
    def increment_retry(self, uni_id: str) -> int:
        """增加重试次数并返回新值"""
        data = self.load()
        if uni_id in data:
            data[uni_id]['retry_count'] = data[uni_id].get('retry_count', 0) + 1
            self.save(data)
            return data[uni_id]['retry_count']
        return 0
    
    def get_running_processes(self) -> Dict[str, dict]:
        """获取所有 running 状态的进程"""
        data = self.load()
        return {k: v for k, v in data.items() 
                if isinstance(v, dict) and v.get('state') == 'running'}
    
    def is_process_running(self, uni_id: str) -> bool:
        """检查进程是否真正在运行"""
        record = self.get_record(uni_id)
        if not record or record.get('state') != 'running':
            return False
        pid = record.get('pid', 0)
        if pid <= 0:
            return False
        return psutil.pid_exists(pid)


class GPUCompetitor:
    """GPU 竞争调度器 - 核心类
    
    核心逻辑：队列内串行，队列间并行
    - 同一队列的任务严格按顺序执行
    - 不同队列的任务可以并行执行（在不同 GPU 上）
    - 每次只分配一个任务，等待确认后再分配下一个
    """
    
    def __init__(self):
        # 初始化日志
        self._setup_logging()
        
        # 初始化 GPU 列表
        if use_all_gpus:
            self.gpus = GPUMonitor.detect_gpus()
        else:
            self.gpus = compete_gpus
        logging.info(f"🖥️ Available GPUs: {self.gpus}")
        
        # 初始化 JSON 管理器
        self.process_json = ProcessJSON(process_json)
        
        # 初始化任务队列
        self.tasks: List[Task] = []
        self.queues: Dict[int, List[Task]] = {}  # queue_id -> [tasks]
        self._setup_tasks()
        
        # 运行状态
        self.running = True
        
        # 配置
        self.task_start_delay = 30  # 每个任务启动后等待秒数
    
    def _setup_logging(self):
        """配置日志"""
        os.makedirs(log_dir, exist_ok=True)
        log_file = self._get_next_log_file()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"📝 Log file: {log_file}")
    
    def _get_next_log_file(self) -> str:
        """获取下一个日志文件名"""
        base = os.path.join(log_dir, 'compete_gpu')
        if not os.path.exists(f"{base}.log"):
            return f"{base}.log"
        i = 1
        while os.path.exists(f"{base}({i}).log"):
            i += 1
        return f"{base}({i}).log"
    
    def _generate_uni_id(self) -> str:
        """生成唯一标识符"""
        return f"compete_{random.randint(100000, 999999)}"
    
    def _setup_tasks(self):
        """初始化任务列表"""
        for commands, queue_id, memory in command_tasks:
            task = Task(
                commands=commands,
                queue_id=queue_id,
                estimated_memory_gb=memory,
                uni_id=self._generate_uni_id()
            )
            self.tasks.append(task)
            
            if queue_id not in self.queues:
                self.queues[queue_id] = []
            self.queues[queue_id].append(task)
        
        logging.info(f"📋 Total tasks: {len(self.tasks)}, Queues: {list(self.queues.keys())}")
        for qid, tasks in self.queues.items():
            logging.info(f"   Queue {qid}: {len(tasks)} tasks")
    
    def find_available_gpu(self, required_memory: int, exclude_gpus: set = None) -> Optional[int]:
        """查找可用的 GPU
        
        条件：
        1. 有足够的显存
        2. 非极限模式下，当前用户没有其他 Python 进程在该 GPU 上
        """
        exclude_gpus = exclude_gpus or set()
        
        for gpu_id in self.gpus:
            if gpu_id in exclude_gpus:
                continue
            
            # 检查显存
            available = GPUMonitor.get_available_memory(gpu_id)
            if available < required_memory:
                logging.debug(f"GPU {gpu_id}: insufficient memory ({available:.1f}GB < {required_memory}GB)")
                continue
            
            # 非极限模式：检查用户进程
            if not maximize_resource_utilization:
                user_procs = GPUMonitor.get_user_processes_on_gpu(gpu_id)
                if user_procs:
                    logging.debug(f"GPU {gpu_id}: user processes exist {user_procs}")
                    continue
            
            logging.info(f"✅ GPU {gpu_id} available: {available:.1f}GB free")
            return gpu_id
        
        return None
    
    def get_busy_queues(self) -> set:
        """获取当前正在运行任务的队列 ID 集合
        
        通过检查 JSON 中 state=running 且进程确实存在的记录
        """
        busy = set()
        running = self.process_json.get_running_processes()
        
        for uni_id, record in running.items():
            pid = record.get('pid', 0)
            if pid > 0 and psutil.pid_exists(pid):
                # 找到对应的任务获取队列 ID
                for task in self.tasks:
                    if task.uni_id == uni_id:
                        busy.add(task.queue_id)
                        break
        
        return busy
    
    def get_occupied_gpus(self) -> set:
        """获取当前被占用的 GPU 集合（非极限模式下）"""
        if maximize_resource_utilization:
            return set()
        
        occupied = set()
        for gpu_id in self.gpus:
            user_procs = GPUMonitor.get_user_processes_on_gpu(gpu_id)
            if user_procs:
                occupied.add(gpu_id)
        return occupied
    
    def get_queue_head_task(self, queue_id: int) -> Optional[Task]:
        """获取队列的第一个 pending 任务"""
        for task in self.queues.get(queue_id, []):
            if task.status == "pending":
                return task
        return None
    
    def execute_task(self, task: Task, gpu_id: int) -> bool:
        """执行任务（同步执行所有命令）
        
        Returns:
            True 如果任务成功启动（bash 脚本已启动后台进程）
        """
        task.assigned_gpu = gpu_id
        task.status = "running"
        
        logging.info(f"🚀 Starting task {task.uni_id} (Queue {task.queue_id}) on GPU {gpu_id}")
        
        for i, cmd_template in enumerate(task.commands):
            # 替换变量
            cmd = cmd_template.format(
                work_dir=work_dir,
                uni_id=task.uni_id
            )
            
            # 在命令前添加 CUDA_VISIBLE_DEVICES 环境变量（确保子进程继承）
            full_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
            
            logging.info(f"   [{i+1}/{len(task.commands)}] [GPU {gpu_id}] {cmd[:80]}...")
            
            try:
                result = subprocess.run(
                    full_cmd, shell=True, capture_output=True, text=True, timeout=300
                )
                
                if result.returncode != 0:
                    logging.error(f"   Command failed: {result.stderr[:200]}")
                    task.status = "failed"
                    return False
                
                # 打印输出（简化）
                if result.stdout.strip():
                    for line in result.stdout.strip().split('\n')[:5]:
                        logging.info(f"   > {line[:100]}")
                        
            except subprocess.TimeoutExpired:
                logging.error(f"   Command timeout")
                task.status = "failed"
                return False
            except Exception as e:
                logging.error(f"   Command error: {e}")
                task.status = "failed"
                return False
        
        task.status = "completed"
        logging.info(f"✅ Task {task.uni_id} commands completed, waiting for background process...")
        return True
    
    def wait_for_process_start(self, task: Task, timeout: int = 60) -> bool:
        """等待任务的 Python 进程真正启动
        
        通过检查 JSON 中的 PID 是否有效来确认
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            record = self.process_json.get_record(task.uni_id)
            if record:
                pid = record.get('pid', 0)
                state = record.get('state', '')
                
                if state == 'running' and pid > 0 and psutil.pid_exists(pid):
                    task.pid = pid
                    logging.info(f"✅ Process confirmed: {task.uni_id} PID={pid}")
                    return True
                elif state in ('normal_exit', 'abnormal_exit'):
                    logging.warning(f"⚠️ Process already exited: {task.uni_id} state={state}")
                    return True  # 进程已结束，也算确认
            
            time.sleep(2)
        
        logging.warning(f"⏰ Timeout waiting for process: {task.uni_id}")
        return False
    
    def print_status(self):
        """打印当前状态"""
        pending = sum(1 for t in self.tasks if t.status == "pending")
        running = sum(1 for t in self.tasks if t.status == "running")
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        
        logging.info("=" * 60)
        logging.info(f"📊 Tasks: Pending={pending}, Running={running}, Completed={completed}, Failed={failed}")
        
        busy_queues = self.get_busy_queues()
        for qid in sorted(self.queues.keys()):
            status = "🔴 BUSY" if qid in busy_queues else "🟢 IDLE"
            q_pending = sum(1 for t in self.queues[qid] if t.status == "pending")
            q_completed = sum(1 for t in self.queues[qid] if t.status == "completed")
            logging.info(f"   Queue {qid}: {status}, Pending={q_pending}, Completed={q_completed}")
        
        logging.info("=" * 60)
    
    def check_and_handle_finished_tasks(self):
        """检查已完成/异常的任务并处理重试
        
        通过 JSON 文件检测进程状态变化
        """
        for task in self.tasks:
            if task.status != "running" and task.status != "completed":
                continue
            
            record = self.process_json.get_record(task.uni_id)
            if not record:
                continue
            
            state = record.get('state', '')
            
            if state == 'normal_exit':
                # 正常退出
                if task.status != "completed":
                    task.status = "completed"
                    logging.info(f"✅ Task {task.uni_id} (Queue {task.queue_id}) completed successfully")
            
            elif state == 'abnormal_exit':
                # 异常退出，需要重试
                if task.status == "running":
                    task.status = "pending"  # 重置为 pending 以便重试
                    task.retry_count += 1
                    task.error_type = record.get('error_type', 'unknown')
                    
                    # 检查是否需要退避
                    if task.retry_count % max_retry_before_backoff == 0:
                        task.backoff_until = time.time() + backoff_duration
                        logging.warning(f"🔄 Task {task.uni_id} failed (retry #{task.retry_count}, error={task.error_type}), "
                                       f"entering backoff for {backoff_duration//60} minutes")
                    else:
                        logging.warning(f"🔄 Task {task.uni_id} failed (retry #{task.retry_count}, error={task.error_type}), "
                                       f"will retry soon")
                    
                    # 生成新的 uni_id 用于重试
                    task.uni_id = self._generate_uni_id()
    
    def is_task_ready(self, task: Task) -> bool:
        """检查任务是否可以被调度（考虑退避）"""
        if task.status != "pending":
            return False
        if task.backoff_until > 0 and time.time() < task.backoff_until:
            return False  # 还在退避期
        return True
    
    def run(self):
        """主调度循环
        
        核心逻辑：
        1. 检查已完成/异常的任务，处理重试
        2. 获取所有空闲队列的头部任务
        3. 随机打乱顺序（公平调度）
        4. 逐个分配任务到可用 GPU
        5. 每分配一个任务后等待确认，再分配下一个
        """
        logging.info("🏁 Starting GPU competition scheduler")
        self.print_status()
        
        try:
            while self.running:
                # 检查已完成/异常的任务
                self.check_and_handle_finished_tasks()
                
                # 检查是否所有任务都完成
                incomplete_tasks = [t for t in self.tasks if t.status != "completed"]
                if not incomplete_tasks:
                    logging.info("🎉 All tasks completed!")
                    break
                
                # 获取当前忙碌的队列
                busy_queues = self.get_busy_queues()
                
                # 获取空闲队列的头部任务（考虑退避）
                candidate_tasks = []
                for qid in self.queues.keys():
                    if qid not in busy_queues:
                        head_task = self.get_queue_head_task(qid)
                        if head_task and self.is_task_ready(head_task):
                            candidate_tasks.append(head_task)
                
                if not candidate_tasks:
                    # 没有可调度的任务，等待
                    time.sleep(check_time)
                    continue
                
                # 随机打乱顺序（公平调度）
                random.shuffle(candidate_tasks)
                
                # 获取当前被占用的 GPU
                occupied_gpus = self.get_occupied_gpus()
                
                # 逐个分配任务
                tasks_started = 0
                for task in candidate_tasks:
                    # 重新获取占用的 GPU（实时检测）
                    occupied_gpus = self.get_occupied_gpus()
                    
                    # 查找可用 GPU
                    gpu_id = self.find_available_gpu(task.estimated_memory_gb, occupied_gpus)
                    
                    if gpu_id is None:
                        logging.info(f"⏳ No GPU available for task {task.uni_id} (need {task.estimated_memory_gb}GB)")
                        continue
                    
                    # 执行任务
                    success = self.execute_task(task, gpu_id)
                    
                    if success:
                        tasks_started += 1
                        
                        # 等待进程确认启动
                        self.wait_for_process_start(task, timeout=60)
                        
                        # 等待一段时间再分配下一个任务
                        logging.info(f"⏳ Waiting {self.task_start_delay}s before next task...")
                        time.sleep(self.task_start_delay)
                
                if tasks_started > 0:
                    logging.info(f"📈 Started {tasks_started} task(s) this round")
                
                # 打印状态
                self.print_status()
                
                # 等待下一轮调度
                time.sleep(check_time)
                
        except KeyboardInterrupt:
            logging.info("🛑 Interrupted by user")
            self.running = False
        
        logging.info("🏁 Scheduler stopped")


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    competitor = GPUCompetitor()
    competitor.run()
