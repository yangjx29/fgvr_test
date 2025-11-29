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

# Configuration
log_dir = '/home/hdl/project/fgvr_test_new/logs/compete_gpu'
check_time = 5
# 三元组（待执行命令列表，GPU ID，估计显存）
# 支持单命令和多命令串行执行
command_tasks=[
            (
                [
                    "rm -rf /home/hdl/project/fgvr_test_new_TestExperience/experiments/dog120/knowledge_base",
                    "bash /home/hdl/project/fgvr_test_new_TestExperience/scripts/run_fast_slow.sh bird --gpu 9"
                ],
                9, 
                40
            ),
            (
                [
                    "rm -rf /home/hdl/project/fgvr_test_new_TestExperience/experiments/dog120/knowledge_base",
                    "bash /home/hdl/project/fgvr_test_new_TestExperience/scripts/run_fast_slow.sh sun397 --gpu 9"
                ],
                9, 
                40
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
    """Monitor GPU memory usage"""
    
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


class CommandQueue:
    """Manages command queue for a specific GPU"""
    
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.queue: List[CommandTask] = []
        self.current_task: Optional[CommandTask] = None
        self.lock = threading.Lock()
    
    def add_task(self, task: CommandTask):
        """Add a task to the queue"""
        with self.lock:
            self.queue.append(task)
    
    def get_next_task(self) -> Optional[CommandTask]:
        """Get the next pending task"""
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
    
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        with self.lock:
            return {
                'gpu_id': self.gpu_id,
                'queue_length': len(self.queue),
                'pending': len([t for t in self.queue if t.status == "pending"]),
                'running': len([t for t in self.queue if t.status == "running"]),
                'completed': len([t for t in self.queue if t.status == "completed"]),
                'failed': len([t for t in self.queue if t.status == "failed"])
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
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Error log file
        error_log_file = os.path.join(self.log_dir, "error.log")
        self.error_handler = logging.FileHandler(error_log_file)
        self.error_handler.setLevel(logging.ERROR)
        
        logger = logging.getLogger()
        logger.addHandler(self.error_handler)
    
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
                 check_interval: int = 5):
        self.log_dir = log_dir
        self.check_interval = check_interval
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
                self.gpu_queues[gpu_id] = CommandQueue(gpu_id)
            
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
                
                with open(task_log_file, 'a') as log_f:
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
        """Monitor and manage tasks for a specific GPU"""
        queue = self.gpu_queues[gpu_id]
        
        while self.running:
            try:
                # Check if current task is running
                if queue.current_task is None or queue.current_task.status in ["completed", "failed"]:
                    queue.current_task = None
                    queue.remove_completed_tasks()
                    
                    # Get next task
                    next_task = queue.get_next_task()
                    if next_task:
                        # Check GPU memory availability
                        available_memory = GPUMonitor.get_available_memory(gpu_id)
                        self.logger.log_gpu_status(gpu_id, available_memory, next_task.estimated_memory_gb)
                        
                        if available_memory > next_task.estimated_memory_gb:
                            queue.current_task = next_task
                            # Execute task in separate thread
                            task_thread = threading.Thread(
                                target=self.execute_task,
                                args=(next_task,)
                            )
                            task_thread.start()
                
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
    competitor = GPUCompetitor(log_dir=log_dir, check_interval=check_time)
    competitor.start_competition()


if __name__ == "__main__":
    main()