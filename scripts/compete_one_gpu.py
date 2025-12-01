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

# 运行 
# nohup python compete_gpu.py > /dev/null 2>&1 & 
# 即可

# Configuration
work_dir = f'/home/hdl/project/fgvr_test_new'
log_dir = f'{work_dir}/logs/compete_gpu'
check_time = 5
maximize_resource_utilization = False # 极限利用资源模式，如果开启允许当前用户的多进程放同一GPU，否则当前用户只能放一个进程到同一GPU
# 三元组（待执行命令列表，GPU ID，估计显存）
# 支持单命令和多命令串行执行
command_tasks=[
            (
                [
                    "rm -rf {work_dir}/experiments/dog120/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 5",
                    "bash {work_dir}/scripts/run_pipeline.sh dog --gpu 4"
                ],
                4, 
                25
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/dog120/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 3",
                    "bash {work_dir}/scripts/run_pipeline.sh dog --gpu 4"
                ],
                4, 
                25
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/dog120/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 10",
                    "bash {work_dir}/scripts/run_pipeline.sh dog --gpu 4"
                ],
                4, 
                25
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/dog120/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 16",
                    "bash {work_dir}/scripts/run_pipeline.sh dog --gpu 4"
                ],
                4, 
                25
            ),
            (
                [
                    "rm -rf {work_dir}/experiments/dog120/knowledge_base",
                    "bash {work_dir}/scripts/set_hyperparameters.sh --experience_number 32",
                    "bash {work_dir}/scripts/run_pipeline.sh dog --gpu 4"
                ],
                4, 
                25
            ),
        ]
@dataclass
class CommandTask:
    """Represents a command task with GPU requirements"""
    commands: List[str]  # 支持多个命令串行执行
    gpu_id: int
    estimated_memory_gb: float
    timestamp: datetime
    process: Optional[subprocess.Popen] = None
    current_command_index: int = 0  # 当前执行的命令索引
    status: str = "pending"  # pending, running, completed, failed


class GPUMonitor:
    """Monitor GPU memory usage and user processes"""
    
    @staticmethod
    def get_gpu_memory_info(gpu_id: int) -> Tuple[float, float]:
        """
        Get GPU memory information for specified GPU
        Returns: (used_memory_gb, total_memory_gb)
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                 '--format=csv,noheader,nounits', f'--id={gpu_id}'],
                capture_output=True, text=True, check=True
            )
            used_mb, total_mb = map(int, result.stdout.strip().split(','))
            return used_mb / 1024, total_mb / 1024
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            logging.error(f"Failed to get GPU {gpu_id} memory info")
            return 0.0, 0.0
    
    @staticmethod
    def get_available_memory(gpu_id: int) -> float:
        """Get available GPU memory in GB"""
        used, total = GPUMonitor.get_gpu_memory_info(gpu_id)
        return total - used
    
    @staticmethod
    def get_user_gpu_processes(gpu_id: int) -> List[Dict]:
        """Get current user's processes using the specified GPU"""
        try:
            current_user = os.getenv('USER')
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                 '--format=csv,noheader,nounits', f'--id={gpu_id}'],
                capture_output=True, text=True, check=True
            )
            
            user_processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    pid, process_name, memory_mb = line.split(', ')
                    # Check if this process belongs to current user
                    try:
                        pid_int = int(pid)
                        if psutil.pid_exists(pid_int):
                            process = psutil.Process(pid_int)
                            if process.username() == current_user or os.getlogin() in process.username():
                                user_processes.append({
                                    'pid': pid_int,
                                    'name': process_name,
                                    'memory_gb': float(memory_mb) / 1024
                                })
                    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                        continue
            
            return user_processes
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            logging.error(f"Failed to get GPU {gpu_id} process info")
            return []
    
    @staticmethod
    def is_user_using_gpu(gpu_id: int) -> bool:
        """Check if current user has processes running on the specified GPU"""
        return len(GPUMonitor.get_user_gpu_processes(gpu_id)) > 0


class CommandQueue:
    """Manages command queue for a specific GPU with FIFO logic"""
    
    def __init__(self, gpu_id: int, maximize_resource_utilization: bool = False):
        self.gpu_id = gpu_id
        self.queue: List[CommandTask] = []
        self.current_task: Optional[CommandTask] = None
        self.lock = threading.Lock()
        self.maximize_resource_utilization = maximize_resource_utilization
        self.last_task_start_time: Optional[datetime] = None
        self.task_wait_interval = 60  # 60秒等待间隔
    
    def add_task(self, task: CommandTask):
        """Add a task to the queue (FIFO)"""
        with self.lock:
            self.queue.append(task)
            logging.info(f"GPU {self.gpu_id}: Task added to queue. Queue length: {len(self.queue)}")
    
    def get_next_task(self) -> Optional[CommandTask]:
        """Get the next pending task (FIFO)"""
        with self.lock:
            for task in self.queue:
                if task.status == "pending":
                    return task
            return None
    
    def remove_completed_tasks(self):
        """Remove completed tasks from queue"""
        with self.lock:
            self.queue = [task for task in self.queue 
                         if task.status not in ["completed", "failed"]]
    
    def can_start_new_task(self, task: CommandTask) -> bool:
        """Check if a new task can start based on resource utilization settings"""
        # Check memory availability first
        available_memory = GPUMonitor.get_available_memory(self.gpu_id)
        if available_memory < task.estimated_memory_gb:
            return False
        
        # If maximize_resource_utilization is True, allow multiple tasks per user
        if self.maximize_resource_utilization:
            return True
        
        # If maximize_resource_utilization is False, only allow one task per user per GPU
        if not GPUMonitor.is_user_using_gpu(self.gpu_id):
            return True
        
        return False
    
    def should_wait_for_next_attempt(self) -> bool:
        """Check if should wait for next attempt based on last task start time"""
        if self.last_task_start_time is None:
            return False
        
        time_since_last_task = (datetime.now() - self.last_task_start_time).total_seconds()
        return time_since_last_task < self.task_wait_interval
    
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        with self.lock:
            return {
                'gpu_id': self.gpu_id,
                'queue_length': len(self.queue),
                'pending': len([t for t in self.queue if t.status == "pending"]),
                'running': len([t for t in self.queue if t.status == "running"]),
                'completed': len([t for t in self.queue if t.status == "completed"]),
                'failed': len([t for t in self.queue if t.status == "failed"]),
                'user_processes': len(GPUMonitor.get_user_gpu_processes(self.gpu_id)),
                'maximize_resource_utilization': self.maximize_resource_utilization
            }


class Logger:
    """Centralized logging system"""
    
    def __init__(self, log_dir: str = "/home/hdl/project/fgvr_test_new/logs/compete_gpu"):
        self.log_dir = log_dir
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging directories and handlers"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Main log file
        log_file = os.path.join(self.log_dir, "compete_gpu.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),  # 覆盖模式
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Error log file
        error_log_file = os.path.join(self.log_dir, "error.log")
        self.error_handler = logging.FileHandler(error_log_file, mode='w')  # 覆盖模式
        self.error_handler.setLevel(logging.ERROR)
        
        logger = logging.getLogger()
        logger.addHandler(self.error_handler)
        
        # Log script PID at startup
        logging.info(f"🚀 GPU Competition Script Started - PID: {os.getpid()}")
        logging.info(f"📁 Log directory: {self.log_dir}")
        logging.info(f"⚙️  Maximize Resource Utilization: {maximize_resource_utilization}")
        logging.info(f"⏱️  Check interval: {check_time} seconds")
        logging.info(f"⏰ Task wait interval: 60 seconds")
    
    def log_gpu_status(self, gpu_id: int, available_gb: float, task_memory_gb: float):
        """Log GPU memory status"""
        status = "SUFFICIENT" if available_gb > task_memory_gb else "INSUFFICIENT"
        logging.info(f"GPU {gpu_id}: Available={available_gb:.2f}GB, "
                    f"Required={task_memory_gb:.2f}GB -> {status}")
    
    def log_task_start(self, task: CommandTask):
        """Log task start"""
        commands_str = " -> ".join(task.commands) if len(task.commands) > 1 else task.commands[0]
        logging.info(f"Starting task on GPU {task.gpu_id}: {commands_str}")
    
    def log_task_complete(self, task: CommandTask):
        """Log task completion"""
        commands_str = " -> ".join(task.commands) if len(task.commands) > 1 else task.commands[0]
        logging.info(f"Task completed on GPU {task.gpu_id}: {commands_str}")
    
    def log_task_error(self, task: CommandTask, error: str):
        """Log task error"""
        commands_str = " -> ".join(task.commands) if len(task.commands) > 1 else task.commands[0]
        logging.error(f"Task failed on GPU {task.gpu_id}: {commands_str} - Error: {error}")


class GPUCompetitor:
    """Main GPU competition manager"""
    
    def __init__(self, log_dir: str = "/home/hdl/project/fgvr_test_new/logs/compete_gpu", 
                 check_interval: int = 5, maximize_resource_utilization: bool = False):
        self.log_dir = log_dir
        self.check_interval = check_interval
        self.maximize_resource_utilization = maximize_resource_utilization
        self.logger = Logger(log_dir)
        self.gpu_queues: Dict[int, CommandQueue] = {}
        self.running = True
        self.monitor_threads: List[threading.Thread] = []
        
        # Define command tasks (command, gpu_id, estimated_memory_gb)
        self.command_tasks = command_tasks
        
        self.setup_queues()
    
    def setup_queues(self):
        """Setup command queues for each GPU"""
        for commands, gpu_id, memory_gb in self.command_tasks:
            if gpu_id not in self.gpu_queues:
                self.gpu_queues[gpu_id] = CommandQueue(gpu_id, self.maximize_resource_utilization)
            
            # 支持单命令（字符串）和多命令（列表）
            if isinstance(commands, str):
                commands = [commands]
            
            task = CommandTask(
                commands=commands,
                gpu_id=gpu_id,
                estimated_memory_gb=memory_gb,
                timestamp=datetime.now()
            )
            self.gpu_queues[gpu_id].add_task(task)
    
    def execute_task(self, task: CommandTask) -> bool:
        """Execute a command task with serial command execution"""
        try:
            self.logger.log_task_start(task)
            
            # Create log file for this task
            task_log_file = os.path.join(
                self.log_dir, 
                f"gpu_{task.gpu_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            
            # 串行执行所有命令
            for i, command in enumerate(task.commands):
                task.current_command_index = i
                logging.info(f"Executing command {i+1}/{len(task.commands)} on GPU {task.gpu_id}: {command}")
                
                # 第一个命令使用覆盖模式，后续命令使用追加模式
                file_mode = 'w' if i == 0 else 'a'
                with open(task_log_file, file_mode) as log_f:
                    if i == 0:
                        # 第一个命令，写入文件头
                        log_f.write(f"=== Task Log - GPU {task.gpu_id} ===\n")
                        log_f.write(f"Start Time: {datetime.now()}\n")
                        log_f.write(f"Total Commands: {len(task.commands)}\n")
                        log_f.write("=" * 50 + "\n")
                    
                    log_f.write(f"\n=== Executing Command {i+1}/{len(task.commands)} ===\n")
                    log_f.write(f"Command: {command}\n")
                    log_f.write(f"Timestamp: {datetime.now()}\n")
                    log_f.write("Output:\n")
                    log_f.flush()
                    
                    task.process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                
                task.status = "running"
                
                # Wait for current command to complete
                return_code = task.process.wait()
                
                if return_code != 0:
                    error_msg = f"Command {i+1} failed with code {return_code}: {command}"
                    self.logger.log_task_error(task, error_msg)
                    task.status = "failed"
                    return False
                
                logging.info(f"Command {i+1}/{len(task.commands)} completed successfully")
            
            # 所有命令都成功完成
            task.status = "completed"
            self.logger.log_task_complete(task)
            return True
                
        except Exception as e:
            task.status = "failed"
            self.logger.log_task_error(task, str(e))
            return False
    
    def monitor_gpu_queue(self, gpu_id: int):
        """Monitor and manage tasks for a specific GPU with FIFO and user process control"""
        queue = self.gpu_queues[gpu_id]
        
        while self.running:
            try:
                # Check if current task is running
                if queue.current_task is None or queue.current_task.status in ["completed", "failed"]:
                    queue.current_task = None
                    queue.remove_completed_tasks()
                    
                    # Get next task (FIFO)
                    next_task = queue.get_next_task()
                    if next_task:
                        # Check if should wait for next attempt
                        if queue.should_wait_for_next_attempt():
                            wait_remaining = queue.task_wait_interval - (datetime.now() - queue.last_task_start_time).total_seconds()
                            logging.info(f"GPU {gpu_id}: Waiting {wait_remaining:.1f}s before next attempt")
                            time.sleep(min(self.check_interval, wait_remaining))
                            continue
                        
                        # Check if task can start based on resource utilization and user processes
                        if queue.can_start_new_task(next_task):
                            available_memory = GPUMonitor.get_available_memory(gpu_id)
                            self.logger.log_gpu_status(gpu_id, available_memory, next_task.estimated_memory_gb)
                            
                            if available_memory > next_task.estimated_memory_gb:
                                queue.current_task = next_task
                                queue.last_task_start_time = datetime.now()
                                
                                # Log task start with user process info
                                user_processes = GPUMonitor.get_user_gpu_processes(gpu_id)
                                logging.info(f"GPU {gpu_id}: Starting task. User processes on GPU: {len(user_processes)}")
                                
                                # Execute task in separate thread
                                task_thread = threading.Thread(
                                    target=self.execute_task,
                                    args=(next_task,)
                                )
                                task_thread.start()
                            else:
                                logging.info(f"GPU {gpu_id}: Insufficient memory for task. Available: {available_memory:.2f}GB, Required: {next_task.estimated_memory_gb:.2f}GB")
                        else:
                            user_processes = GPUMonitor.get_user_gpu_processes(gpu_id)
                            if not self.maximize_resource_utilization and user_processes:
                                logging.info(f"GPU {gpu_id}: User has {len(user_processes)} processes running. Waiting for user processes to complete (maximize_resource_utilization=False)")
                            else:
                                available_memory = GPUMonitor.get_available_memory(gpu_id)
                                logging.info(f"GPU {gpu_id}: Cannot start task. Available memory: {available_memory:.2f}GB, Required: {next_task.estimated_memory_gb:.2f}GB")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logging.error(f"Error monitoring GPU {gpu_id}: {e}")
                time.sleep(self.check_interval)
    
    def start_competition(self):
        """Start the GPU competition process"""
        logging.info("Starting GPU competition process")
        logging.info(f"Check interval: {self.check_interval} seconds")
        logging.info(f"Log directory: {self.log_dir}")
        
        # Start monitoring thread for each GPU
        for gpu_id in self.gpu_queues.keys():
            monitor_thread = threading.Thread(
                target=self.monitor_gpu_queue,
                args=(gpu_id,),
                daemon=True
            )
            monitor_thread.start()
            self.monitor_threads.append(monitor_thread)
        
        try:
            # Main monitoring loop
            while self.running:
                self.print_status()
                time.sleep(self.check_interval * 2)  # Print status less frequently
                
        except KeyboardInterrupt:
            logging.info("Shutting down GPU competition...")
            self.running = False
            
            # Wait for all threads to complete
            for thread in self.monitor_threads:
                thread.join(timeout=10)
    
    def print_status(self):
        """Print current status of all queues"""
        logging.info("=== Queue Status ===")
        for gpu_id, queue in self.gpu_queues.items():
            status = queue.get_queue_status()
            logging.info(f"GPU {status['gpu_id']}: "
                        f"Pending={status['pending']}, "
                        f"Running={status['running']}, "
                        f"Completed={status['completed']}, "
                        f"Failed={status['failed']}")


def main():
    """Main function"""
    # Create and start GPU competitor
    competitor = GPUCompetitor(
        log_dir=log_dir, 
        check_interval=check_time,
        maximize_resource_utilization=maximize_resource_utilization
    )
    competitor.start_competition()


if __name__ == "__main__":
    main()