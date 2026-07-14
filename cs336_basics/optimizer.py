from collections.abc import Iterable
import math
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        # 1. 基础的超参数合法性检查
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 2. 将超参数打包成 defaults 字典，交给基类管理 (也就是你之前学过的逻辑)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad() # 极其重要：优化器的权重更新过程绝对不能被记录到计算图中！
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # 遍历参数组 (应对不同网络层使用不同学习率的情况)
        # group为一个字典{"params":[一组参数张量],"lr":0.001,"betas":(0.9,0.99),"eps":,"weight_decay":}
        for group in self.param_groups:
            # 提取该组的超参数
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            # 遍历该组中的每一个具体的张量参数 θ
            # p是一个参数张量
            # self.state[p] = {"step":步数，"exp_avg":一阶矩，"exp_avg_sq":二阶矩}
            for p in group['params']:
                # 如果没有计算出梯度，跳过
                if p.grad is None:
                    continue
                
                # [伪代码第 6 行] g <- ∇_θ L
                grad = p.grad
                
                # 从基类提供的 state 字典中获取该参数的状态
                state = self.state[p]

                # [伪代码第 2, 3 行] 初始化 m=0, v=0 和 步数 t=0
                if len(state) == 0:
                    state['step'] = 0
                    # 一阶矩 m：保持与参数相同的形状和设备
                    # zeros_like保存形状，数据类型和device。format：保存原来数据在内存中的排布
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 二阶矩 v：保持与参数相同的形状和设备
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # t <- t + 1
                state['step'] += 1
                t = state['step']

                # [伪代码第 8 行] θ <- θ - α * λ * θ (应用纯粹的权重衰减，解耦！)
                if weight_decay != 0:
                    # 使用 sub_ 直接在显存原地做减法：p = p - (lr * weight_decay) * p
                    # A-C*B，从左往右为ABC
                    p.sub_(p, alpha=lr * weight_decay)

                # [伪代码第 9 行] m <- β1 * m + (1 - β1) * g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # [伪代码第 10 行] v <- β2 * v + (1 - β2) * g^2
                # addcmul_ 的意思是：加 (add) 上一个常数 (c) 乘 (mul) 两个张量
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # [伪代码第 7 行] 计算修正后的学习率 α_t <- α * sqrt(1 - β2^t) / (1 - β1^t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # [伪代码第 11 行] θ <- θ - α_t * m / (sqrt(v) + ε)
                # 为了防止除法出错，我们先算出分母: sqrt(v) + ε
                denom = exp_avg_sq.sqrt().add_(eps)
                # addcdiv_ 的意思是：加 (add) 上一个常数 (c) 乘以两个张量相除 (div)
                p.addcdiv_(exp_avg, denom, value=-alpha_t)

        return loss
    

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it<warmup_iters:
        return max_learning_rate*it/warmup_iters
    elif warmup_iters<=it<=cosine_cycle_iters:
        a=math.pi*(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)
        return min_learning_rate+0.5*(1+math.cos(a))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate
    
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads=[p.grad for p in parameters if p.grad is not None]
    if len(grads)==0:
        return 

    # torch.linalg.vector_norm(p)**2 第二范数的平方，有一个数的张量
    g_norm=[torch.linalg.vector_norm(g)**2 for g in grads]
    
    # 把张量堆叠(改变维度)之后求和
    l2_norm=torch.sqrt(torch.stack(g_norm).sum())
    

    eps=1e-6
    if l2_norm>max_l2_norm:
        clip_coef=max_l2_norm/(l2_norm+eps)

        for g in grads:
            
            g.mul_(clip_coef)
    return None
