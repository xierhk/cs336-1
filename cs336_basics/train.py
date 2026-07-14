import argparse
import shutil
from cs336_basics.logger import ExperimentLogger
import wandb
import torch
import numpy as np
import os
from einops import rearrange
# 从之前写的模块中导入打磨好的函数
from cs336_basics.data import get_batch
from cs336_basics.serialization import  save_checkpoint,load_checkpoint
from cs336_basics.optimizer import  gradient_clipping,AdamW,lr_cosine_schedule
from cs336_basics.transformer import transformer_lm
from cs336_basics.layers import cross_entropy

def main():
    # ==========================================
    # 需求 1: 配置超参数 (使用 argparse 解析命令行)
    # ==========================================
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
    parser.add_argument("--exp_note", type=str, default="baseline", help="消融实验备注 (例如: no_dropout, gelu_ffn, deep_narrow)")
    parser.add_argument("--train_data", type=str, default="data/train.bin", help="训练集文件路径")
    parser.add_argument("--val_data", type=str, default="data/valid.bin", help="验证集文件路径")
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    # ... 可以根据需要添加更多（如 dropout等）
    #   WANDB_MODE=disabled python train.py 可以关掉wandb   

    args = parser.parse_args()
    train_data_path = os.path.abspath(args.train_data)
    val_data_path = os.path.abspath(args.val_data)
    ckpt_path_abs = os.path.abspath(args.ckpt_path)
    b_ckpt_path_abs = os.path.abspath(args.b_ckpt_path)
    run_name=f"[{args.exp_note}]-D{args.d_model}-L{args.num_layers}-LR{args.max_lr}-BS{args.batch_size}-H{args.num_heads}-Ctx{args.context_length}"
    if args.overfit_single_batch:
        run_name+="[OverFit_Test]-"
    logger = ExperimentLogger(
        project_name="cs336-assignment1", 
        config=vars(args),
        run_name=run_name
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ==========================================
    # 需求 2: 内存高效加载 (np.memmap 魔法)
    # ==========================================
    print("正在挂载数据集...")
    # 假设你已经把 token 处理成了 bin 文件
    train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    # ==========================================
    # 初始化模型与优化器
    # ==========================================
    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=int(round((args.d_model * 8 / 3) / 64)) * 64,
        rope_theta=args.rope_theta
        ).to(device)
    optimizer = AdamW(model.parameters())
    
    # 【可选】恢复历史训练
    start_iter = 0
    best_val_loss = float('inf')
    if os.path.exists(args.ckpt_path):
        start_iter = load_checkpoint(args.ckpt_path, model, optimizer)
        print(f"成功从第 {start_iter} 步恢复训练！")

    # 开启单批次过拟合测试
    if args.overfit_single_batch:
        print("🚨 警告: 开启单批次过拟合测试！模型将反复记忆同一个 Batch。")
        X_fixed, Y_fixed = get_batch(train_data, args.batch_size, args.context_length, device)  

    # ==========================================
    # 核心训练大循环
    # ==========================================
    print("🚀 开始训练...")
    for iter_num in range(start_iter, args.max_iters):
        # 0.余弦退火算学习率并赋值
        lr = lr_cosine_schedule(iter_num, args.max_lr, args.min_lr, args.witers,args.citers)
        for param_group in optimizer.param_groups:
            param_group['lr']=lr

        # 1. 抓取一个 Batch (使用你写的零拷贝 get_batch)
        # 单批次过拟合测试
        if args.overfit_single_batch:
            X, Y = X_fixed, Y_fixed
        else:
            X, Y= get_batch(train_data, args.batch_size, args.context_length, device)
        
        # 2. 前向传播
        logits= model(X)
        logits=rearrange(logits,'b s v->(b s) v')
        Y=rearrange(Y,'b s -> (b s)')
        loss=cross_entropy(logits,Y) # 算损失

        # 3. 反向传播
        optimizer.zero_grad(set_to_none=True) # 让指针不再指向原来的梯度，直接指向空值
        loss.backward()
        
        # 3.1反向传播后更新Norm
        if iter_num % 100 == 0:
            total_grad_norm = 0.0
            total_weight_norm = 0.0
            for p in model.parameters():
                if p.requires_grad:
                    total_weight_norm += p.data.norm(2).item() ** 2
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
            
            # 3.2记录学习率和损失值
            print(f"Step {iter_num:05d} | Train Loss: {loss.item():.4f} | LR: {lr:.2e}")
            logger.log_metrics(
                step=iter_num, 
                train_loss=loss.item(), 
                lr=lr,  # 余弦退火的当前学习率
                grad_norm=total_grad_norm ** 0.5,
                weight_norm=total_weight_norm ** 0.5
            )
        # 4. 梯度裁剪 (使用你写的防雷版 clip_grad)
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        
        # 5. 更新权重
        optimizer.step()

                
        # ==========================================
        # 需求 3 & 4: 验证性能与保存 Checkpoint
        # ==========================================
        if iter_num > start_iter and iter_num % args.eval_interval == 0 and not args.overfit_single_batch:
            # 切换到验证模式 (关闭 dropout 等)
            model.eval()
            
            with torch.inference_mode(): # 关闭求导，极致省显存！
                eval_iters=10
                total_val_loss=0.0


                for _ in range(eval_iters):
                    X_val, Y_val = get_batch(val_data, args.batch_size, args.context_length, device)
                    batch_loss = cross_entropy(rearrange(model(X_val),'b s v->(b s) v'),rearrange(Y_val,'b s -> (b s)'))
                    # 把batch_loss移到cpu中
                    total_val_loss+=batch_loss.item()

                val_loss=total_val_loss/eval_iters
                print(f"--- 评估 --- Step {iter_num} | Val Loss: {val_loss:.4f}")
            logger.log_metrics(
                step=iter_num, 
                val_loss=val_loss
            )
            # 切回训练模式
            model.train()
            
            # 执行序列化保存 (把模型、优化器、步数打包存盘)
            os.makedirs(os.path.dirname(ckpt_path_abs), exist_ok=True)
            save_checkpoint(model, optimizer, iter_num, ckpt_path_abs)
            print(f"✅ Checkpoint 已保存至 {ckpt_path_abs}")

            # 保存最优  
            if val_loss<best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(b_ckpt_path_abs),exist_ok=True)
                shutil.copyfile(ckpt_path_abs, b_ckpt_path_abs)
                print(f"✅ best_checkpoint 已保存至 {b_ckpt_path_abs}")
    logger.finish()
if __name__ == "__main__":
    main()