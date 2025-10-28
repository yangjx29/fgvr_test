"""
MLLM单例管理器
防止Qwen模型重复加载导致显存爆满
"""

import torch
from typing import Optional

class MLLMSingleton:
    """MLLM单例管理器，确保整个系统只加载一次MLLM模型"""
    
    _instance: Optional['MLLMSingleton'] = None
    _mllm_bot = None
    _model_tag = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLLMSingleton, cls).__new__(cls)
        return cls._instance
    
    def get_mllm_bot(self, model_tag: str = "Qwen2.5-VL-7B", device: str = "cuda"):
        """
        获取MLLM模型实例，如果已存在则复用，否则创建新实例
        
        Args:
            model_tag: 模型标签
            device: 设备类型
            
        Returns:
            MLLMBot实例
        """
        # 检查是否需要重新加载（模型或设备发生变化）
        if (self._mllm_bot is None or 
            self._model_tag != model_tag or 
            self._device != device):
            
            # 清理旧模型（如果存在）
            if self._mllm_bot is not None:
                print(f"🗑️ 清理旧的MLLM模型: {self._model_tag}")
                del self._mllm_bot
                if device == "cuda":
                    torch.cuda.empty_cache()
            
            # 创建新模型
            print(f"🚀 初始化MLLM模型: {model_tag} on {device}")
            print(f"📊 当前显存: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
            
            from agents.mllm_bot import MLLMBot
            self._mllm_bot = MLLMBot(
                model_tag=model_tag,
                model_name=model_tag,
                device=device
            )
            self._model_tag = model_tag
            self._device = device
            
            print(f"✅ MLLM模型初始化完成")
            print(f"📊 加载后显存: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        else:
            print(f"♻️ 复用已加载的MLLM模型: {model_tag}")
        
        return self._mllm_bot
    
    def clear_cache(self):
        """清理MLLM缓存，释放显存"""
        if self._mllm_bot is not None:
            print(f"🗑️ 清理MLLM模型缓存")
            del self._mllm_bot
            self._mllm_bot = None
            self._model_tag = None
            self._device = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> dict:
        """获取显存使用情况"""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "usage_percent": torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
            }
        else:
            return {"error": "CUDA not available"}


# 全局单例实例
_mllm_manager = MLLMSingleton()

def get_mllm_bot(model_tag: str = "Qwen2.5-VL-7B", device: str = "cuda"):
    """
    获取MLLM模型实例的便捷函数
    
    Args:
        model_tag: 模型标签
        device: 设备类型
        
    Returns:
        MLLMBot实例
    """
    return _mllm_manager.get_mllm_bot(model_tag, device)

def clear_mllm_cache():
    """清理MLLM缓存的便捷函数"""
    _mllm_manager.clear_cache()

def get_memory_usage():
    """获取显存使用情况的便捷函数"""
    return _mllm_manager.get_memory_usage()
