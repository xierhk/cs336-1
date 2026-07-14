import torch
import numpy.typing as npt
import numpy as np

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str=None
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. 随机生成 batch_size 个起始索引
    # 上限是 len(dataset) - context_length
    # torch.randint 的区间是 [0, high)，所以最大能取到的索引是 len(dataset) - context_length - 1
    ix = torch.randint(len(dataset) - context_length, (batch_size,))
    
    # 2. 遍历这些随机索引，从原数组中切出 X (输入) 和 Y (标签)
    # X 的范围是 [i, i + context_length]
    # Y 的范围是 [i + 1, i + 1 + context_length] （整体错开1位）
    # 先将 numpy 切片转换为 int64 类型（对应 torch.LongTensor），再转换为 torch.Tensor
    # stack默认在dim=0的位置增加一个维度
    # form_numpy是零拷贝，用指针指向一片区域，不存在数据移动以及拷贝
    x = torch.stack([torch.from_numpy((dataset[i : i + context_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64)) for i in ix])
    
    # 3. 将组装好的批次张量推送到指定的物理设备 (CPU / GPU)
    x = x.to(device)
    y = y.to(device)
    
    return x, y
