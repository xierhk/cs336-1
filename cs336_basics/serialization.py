import os
from typing import IO, BinaryIO
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    
    # 2. 一次性将整个字典序列化到硬盘
    torch.save(checkpoint_dict, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    
    # 1. 读取 Checkpoint 字典
    # 架构师避坑：强烈建议加上 map_location='cpu'
    # 这样即使你的 Checkpoint 是在 8 卡 GPU 上训练保存的，现在拿到只有 1 张卡或没卡的机器上加载，也不会报显存错误
    checkpoint_dict = torch.load(src, map_location='cpu')
    
    # 2. 恢复模型权重 (就地操作，无需返回值)
    model.load_state_dict(checkpoint_dict["model"])
    
    # 3. 恢复优化器状态 (就地操作，无需返回值)
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    
    # 4. 提取并返回中断时的训练步数
    iteration = checkpoint_dict["iteration"]
    
    return iteration