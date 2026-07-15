import argparse
import shutil
from cs336_basics.logger import ExperimentLogger
import wandb
import torch
import numpy as np
import os
from einops import rearrange

# 🌟 DDP修改 1: 导入分布式核心模块
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from cs336_basics.data import get_batch
from cs336_basics.serialization import  save_checkpoint, load_checkpoint
from cs336_basics.optimizer import  gradient_clipping, AdamW, lr_cosine_schedule
from cs336_basics.transformer import transformer_lm
from cs336_basics.layers import cross_entropy

def main():
    parser = argparse.ArgumentParser(description="LLM 训练主脚本")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--rope_theta", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/model_latest.pt")
    parser.add_argument("--b_ckpt_path", type=str, default="checkpoints/model_best.pt")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--witers", type=int, default=500)
    parser.add_argument("--citers", type=int, default=5000)
    parser.add_argument("--overfit_single_batch", action="store_true", help="开启单批次过拟合测试")
    parser.add_argument("--exp_note", type=str, default="baseline")
    parser.add_argument("--train_data", type=str, default="data/train.bin")
    parser.add_argument("--val_data", type=str, default="data/valid.bin")
    parser.add_argument("--vocab_size", type=int, default=10000)
    args = parser.parse_args()

    # ==========================================
    # 🌟 DDP修改 2: 初始化进程组与环境探测
    # ==========================================
    ddp = int(os.environ.get('RANK', -1)) != -1 # 探测是否是由 torchrun 启动的
    if ddp:
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # 只有0号卡是主进程
        seed_offset = ddp_rank # 用于错开两张卡抽到的数据
    else:
        master_process = True
        seed_offset = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 🌟 DDP修改 3: 极其关键！强制错开随机种子，防止两张卡训练一模一样的数据
    torch.manual_seed(1337 + seed_offset)

    train_data_path = os.path.abspath(args.train_data)
    val_data_path = os.path.abspath(args.val_data)
    ckpt_path_abs = os.path.abspath(args.ckpt_path)
    b_ckpt_path_abs = os.path.abspath(args.b_ckpt_path)
    
    run_name=f"[{args.exp_note}]-D{args.d_model}-L{args.num_layers}-LR{args.max_lr}-BS{args.batch_size}-H{args.num_heads}-Ctx{args.context_length}"
    if args.overfit_single_batch:
        run_name+="[OverFit_Test]-"
        
    # 🌟 DDP修改 4: 只有主进程才允许开启 WandB 遥测，否则界面会重影报错！
    logger = None
    if master_process:
        logger = ExperimentLogger(
            project_name="cs336-assignment1", 
            config=vars(args),
            run_name=run_name
        )

    if master_process:
        print("正在挂载数据集...")
        
    train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')

    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=int(round((args.d_model * 8 / 3) / 64)) * 64,
        rope_theta=args.rope_theta
    ).to(device)

    # 🌟 DDP修改 5: 给模型穿上分布式战甲
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # 🌟 DDP修改 6: 获取剥离了 DDP 外壳的“原始模型”（用于保存权重和查参数）
    raw_model = model.module if ddp else model
    optimizer = AdamW(raw_model.parameters())
    
    start_iter = 0
    best_val_loss = float('inf')
    if os.path.exists(args.ckpt_path):
        start_iter = load_checkpoint(args.ckpt_path, raw_model, optimizer)
        if master_process:
            print(f"成功从第 {start_iter} 步恢复训练！")

    if args.overfit_single_batch:
        if master_process:
            print("🚨 警告: 开启单批次过拟合测试！")
        X_fixed, Y_fixed = get_batch(train_data, args.batch_size, args.context_length, device)  

    if master_process:
        print("🚀 开始训练...")
        
    for iter_num in range(start_iter, args.max_iters):
        lr = lr_cosine_schedule(iter_num, args.max_lr, args.min_lr, args.witers, args.citers)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if args.overfit_single_batch:
            X, Y = X_fixed, Y_fixed
        else:
            X, Y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        logits = model(X)
        logits = rearrange(logits, 'b s v -> (b s) v')
        Y = rearrange(Y, 'b s -> (b s)')
        loss = cross_entropy(logits, Y) 

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if iter_num % 100 == 0 and master_process:
            total_grad_norm = 0.0
            total_weight_norm = 0.0
            # 注意这里要用 raw_model 去拿参数
            for p in raw_model.parameters():
                if p.requires_grad:
                    total_weight_norm += p.data.norm(2).item() ** 2
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
            
            print(f"Step {iter_num:05d} | Train Loss: {loss.item():.4f} | LR: {lr:.2e}")
            logger.log_metrics(
                step=iter_num, 
                train_loss=loss.item(), 
                lr=lr,  
                grad_norm=total_grad_norm ** 0.5,
                weight_norm=total_weight_norm ** 0.5
            )
            
        # 注意这里传给裁剪的也是 raw_model.parameters()
        gradient_clipping(raw_model.parameters(), max_l2_norm=1.0)
        optimizer.step()

        # 🌟 DDP修改 7: 极其致命的死锁坑防御！
        # 验证过程必须两张卡同时进入（即使只有0号卡打印日志），否则1号卡提前跑去算下一个 backward，
        # DDP 会因为等不到0号卡的梯度而彻底卡死（Deadlock）！
        if iter_num > start_iter and iter_num % args.eval_interval == 0 and not args.overfit_single_batch:
            model.eval()
            with torch.inference_mode(): 
                eval_iters=10
                total_val_loss=0.0
                for _ in range(eval_iters):
                    X_val, Y_val = get_batch(val_data, args.batch_size, args.context_length, device)
                    batch_loss = cross_entropy(rearrange(model(X_val),'b s v->(b s) v'), rearrange(Y_val,'b s -> (b s)'))
                    total_val_loss += batch_loss.item()
                val_loss = total_val_loss / eval_iters
                
            model.train() # 所有卡一起切回训练模式
            
            # 但只有主进程负责报告结果和保存文件
            if master_process:
                print(f"--- 评估 --- Step {iter_num} | Val Loss: {val_loss:.4f}")
                logger.log_metrics(step=iter_num, val_loss=val_loss)
                
                os.makedirs(os.path.dirname(ckpt_path_abs), exist_ok=True)
                # 🌟 DDP修改 8: 保存时必须剥离 DDP 外壳，否则以后单卡无法读取！
                save_checkpoint(raw_model, optimizer, iter_num, ckpt_path_abs)
                print(f"✅ Checkpoint 已保存至 {ckpt_path_abs}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs(os.path.dirname(b_ckpt_path_abs), exist_ok=True)
                    shutil.copyfile(ckpt_path_abs, b_ckpt_path_abs)
                    print(f"✅ best_checkpoint 已保存至 {b_ckpt_path_abs}")

    if master_process and logger is not None:
        logger.finish()
        
    # 🌟 DDP修改 9: 优雅退出，销毁进程组
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()