# File: gpu_health.py
# GPU health monitoring and circuit breaker for CUDA stability

import torch
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import threading

logger = logging.getLogger(__name__)


@dataclass
class GPUHealth:
    """GPU health status snapshot"""
    is_available: bool
    device_count: int
    current_device: int
    memory_allocated_gb: float
    memory_reserved_gb: float
    memory_free_gb: float
    memory_total_gb: float
    last_error: Optional[str]
    error_count: int
    circuit_open: bool


class GPUHealthMonitor:
    """
    Monitors GPU health and implements circuit breaker pattern.
    
    Circuit breaker prevents continuous failures by "opening" after
    too many errors, giving the system time to recover.
    """
    
    def __init__(
        self,
        error_threshold: int = 5,
        reset_timeout_minutes: int = 5,
        min_free_memory_gb: float = 1.0
    ):
        """
        Initialize GPU health monitor.
        
        Args:
            error_threshold: Number of errors before opening circuit
            reset_timeout_minutes: Minutes to wait before attempting reset
            min_free_memory_gb: Minimum free GPU memory required
        """
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.last_error_time: Optional[datetime] = None
        self.circuit_open = False
        self.circuit_open_time: Optional[datetime] = None
        
        self.error_threshold = error_threshold
        self.reset_timeout = timedelta(minutes=reset_timeout_minutes)
        self.min_free_memory_gb = min_free_memory_gb
        
        logger.info(
            f"GPU Health Monitor initialized: "
            f"error_threshold={error_threshold}, "
            f"reset_timeout={reset_timeout_minutes}m, "
            f"min_memory={min_free_memory_gb}GB"
        )
    
    def check_health(self) -> GPUHealth:
        """
        Check current GPU health status.
        
        Returns:
            GPUHealth object with current status
        """
        # Check if circuit should auto-reset
        if self.circuit_open and self.circuit_open_time:
            if datetime.now() - self.circuit_open_time > self.reset_timeout:
                logger.info("Circuit breaker timeout reached, attempting reset")
                self.reset_circuit_breaker()
        
        if not torch.cuda.is_available():
            return GPUHealth(
                is_available=False,
                device_count=0,
                current_device=-1,
                memory_allocated_gb=0,
                memory_reserved_gb=0,
                memory_free_gb=0,
                memory_total_gb=0,
                last_error="CUDA not available",
                error_count=self.error_count,
                circuit_open=self.circuit_open
            )
        
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            mem_allocated = torch.cuda.memory_allocated(device) / 1e9
            mem_reserved = torch.cuda.memory_reserved(device) / 1e9
            mem_total = props.total_memory / 1e9
            mem_free = mem_total - mem_reserved
            
            return GPUHealth(
                is_available=True,
                device_count=torch.cuda.device_count(),
                current_device=device,
                memory_allocated_gb=mem_allocated,
                memory_reserved_gb=mem_reserved,
                memory_free_gb=mem_free,
                memory_total_gb=mem_total,
                last_error=self.last_error,
                error_count=self.error_count,
                circuit_open=self.circuit_open
            )
        except Exception as e:
            self._record_error(str(e))
            return GPUHealth(
                is_available=False,
                device_count=0,
                current_device=-1,
                memory_allocated_gb=0,
                memory_reserved_gb=0,
                memory_free_gb=0,
                memory_total_gb=0,
                last_error=str(e),
                error_count=self.error_count,
                circuit_open=self.circuit_open
            )
    
    def is_healthy(self) -> bool:
        """
        Check if GPU is currently healthy and ready for use.
        
        Returns:
            True if GPU is available and healthy
        """
        if self.circuit_open:
            logger.warning("GPU circuit breaker is OPEN - rejecting request")
            return False
        
        health = self.check_health()
        
        if not health.is_available:
            return False
        
        if health.memory_free_gb < self.min_free_memory_gb:
            logger.warning(
                f"GPU memory low: {health.memory_free_gb:.2f}GB free "
                f"(minimum {self.min_free_memory_gb}GB required)"
            )
            return False
        
        return True
    
    def _record_error(self, error_msg: str):
        """Record a GPU error and check circuit breaker threshold"""
        self.error_count += 1
        self.last_error = error_msg
        self.last_error_time = datetime.now()
        
        logger.error(f"GPU error #{self.error_count}: {error_msg}")
        
        if self.error_count >= self.error_threshold:
            self._open_circuit()
    
    def _open_circuit(self):
        """Open the circuit breaker to prevent cascading failures"""
        if not self.circuit_open:
            self.circuit_open = True
            self.circuit_open_time = datetime.now()
            logger.critical(
                f"GPU CIRCUIT BREAKER OPENED after {self.error_count} errors. "
                f"Will attempt auto-reset in {self.reset_timeout.total_seconds()/60:.1f} minutes."
            )
    
    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker"""
        was_open = self.circuit_open
        self.circuit_open = False
        self.circuit_open_time = None
        self.error_count = 0
        self.last_error = None
        
        if was_open:
            logger.info("GPU circuit breaker RESET - accepting requests again")
    
    def record_success(self):
        """Record a successful GPU operation"""
        # Gradually decrease error count on success
        if self.error_count > 0:
            self.error_count = max(0, self.error_count - 1)
            if self.error_count == 0:
                self.last_error = None
                logger.info("GPU error count cleared after successful operation")
    
    def get_memory_info(self) -> dict:
        """Get detailed memory information"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            mem_allocated = torch.cuda.memory_allocated(device)
            mem_reserved = torch.cuda.memory_reserved(device)
            mem_total = props.total_memory
            
            return {
                "allocated_gb": mem_allocated / 1e9,
                "reserved_gb": mem_reserved / 1e9,
                "free_gb": (mem_total - mem_reserved) / 1e9,
                "total_gb": mem_total / 1e9,
                "utilization_percent": (mem_reserved / mem_total) * 100
            }
        except Exception as e:
            return {"error": str(e)}


# Global instance
_gpu_monitor: Optional[GPUHealthMonitor] = None
_gpu_monitor_lock = threading.Lock()


def get_gpu_monitor() -> GPUHealthMonitor:
    """Get or create the global GPU health monitor instance (thread-safe)"""
    global _gpu_monitor
    if _gpu_monitor is None:
        with _gpu_monitor_lock:
            # Double-check locking pattern to prevent race conditions
            if _gpu_monitor is None:
                _gpu_monitor = GPUHealthMonitor(
                    error_threshold=5,
                    reset_timeout_minutes=5,
                    min_free_memory_gb=1.0
                )
    return _gpu_monitor
