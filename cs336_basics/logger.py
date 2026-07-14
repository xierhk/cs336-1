import time
import wandb

class ExperimentLogger:
    """
    实验追踪基础设施
    用于记录损失曲线，支持按梯度步数 (steps) 和 挂钟时间 (wall-clock time) 追踪。
    """
    def __init__(self, project_name: str, config: dict, run_name: str = None):
        # 初始化 wandb，并记录超参数 config
        # name可用L4-H16-D512-BS32-Ctx256-LR6e-4
        self.run = wandb.init(project=project_name, config=config, name=run_name)
        
        # 记录整个实验开始的绝对时间戳
        self.start_time = time.time()
        
    def log_metrics(self, step: int, train_loss: float = None, val_loss: float = None, lr: float = None,grad_norm:float=None,weight_norm:float=None):
        """
        在每一步记录指标
        """
        # 计算当前的挂钟时间 (Wall-clock time)，单位：秒
        elapsed_time = time.time() - self.start_time
        
        # 构建要上传的数据字典
        metrics = {
            "step": step,
            "wall_clock_time": elapsed_time  # 满足作业的核心要求
        }
        
        if train_loss is not None:
            metrics["train/loss"] = train_loss
        if val_loss is not None:
            metrics["val/loss"] = val_loss
        if lr is not None:
            metrics["train/lr"] = lr
        if grad_norm is not None:
            metrics["grid_norm"] = grad_norm
        if weight_norm is not None:
            metrics["weight_norm"] = weight_norm
        # 一键同步到云端
        wandb.log(metrics)

    def finish(self):
        """结束实验记录"""
        wandb.finish()