import os
import sys
from typing import Optional

import torch
import torch.distributed as dist


def ddp_available(cfg_runtime: dict) -> bool:
    ddp_cfg = cfg_runtime.get("ddp", {}) if cfg_runtime else {}
    return bool(ddp_cfg.get("enabled", False)) and (int(os.environ.get("WORLD_SIZE", "1")) > 1)


def choose_backend() -> str:
    # NCCL не поддерживается в Windows; используем gloo как запасной вариант
    if sys.platform == "win32":
        return "gloo"
    return "nccl" if torch.cuda.is_available() else "gloo"


def init_distributed(backend: Optional[str] = None):
    if dist.is_available() and not dist.is_initialized():
        backend = backend or choose_backend()
        dist.init_process_group(backend=backend, timeout=torch.distributed.constants.default_pg_timeout)
        if backend == "nccl":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def get_device(cfg_runtime: dict, local_rank: int) -> torch.device:
    """
    Выбирает подходящее устройство torch.device с учётом доступности CUDA и среды DDP.
    """
    if torch.cuda.is_available():
        # При использовании DDP привязываемся к устройству local_rank
        if dist.is_available() and dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return torch.device(f"cuda:{local_rank}")
        # Иначе используем устройство CUDA по умолчанию
        return torch.device("cuda")
    # Запасной вариант — CPU
    return torch.device("cpu")

