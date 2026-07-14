import torch
import torch.nn.functional as F

@torch.no_grad() # 推理时必须关闭梯度计算，极大节省显存！
def generate(model, input_ids, max_new_tokens, temperature=1.0, top_p=1.0, eos_token_id=None):
    """
    大模型标准解码函数 (支持 Temperature 和 Top-p 采样)
    
    参数:
        model: 你的语言模型 (YourLanguageModel)
        input_ids: 初始提示词 (Prompt)，形状为 [batch_size, seq_len]
        max_new_tokens: 允许生成的最大 token 数量
        temperature: 温度值，控制生成的随机性
        top_p: 核采样阈值
        eos_token_id: <|endoftext|> 的 token ID
    """
    # 确保模型处于评估模式 (关闭 Dropout 等)
    model.eval()

    # 循环生成，直到达到最大长度 (要求 2)
    for _ in range(max_new_tokens):
        # 1. 前向传播获取当前序列的 logits
        # 注意：这里我们传入了完整的历史上下文 (如果在实际工程中，这里会结合 KV Cache 优化)
        logits = model(input_ids)
        
        # 2. 我们只需要序列最后一个位置的输出，用来预测下一个词
        # 形状变为: [batch_size, vocab_size]
        next_token_logits = logits[:, -1, :]
        
        # ==========================================
        # 要求 3: 应用温度缩放 (Temperature Scaling)
        # ==========================================
        if temperature != 1.0:
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
            else:
                # 架构师细节：如果温度设为 0，直接退化为贪婪解码 (Greedy Search)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                continue

        # ==========================================
        # 要求 4: Top-p 采样 (Nucleus Sampling)
        # ==========================================
        if top_p < 1.0:
            # 步骤 A: 将 logits 降序排列
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            
            # 步骤 B: 计算累积概率分布
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 步骤 C: 找到累积概率超过 top_p 的索引，生成掩码 (Mask)
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # 架构师细节：向右平移一位，确保无论如何至少保留概率最大的那 1 个词，防止把词表全清空了
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0 
            
            # 步骤 D: 把掩码还原回原始的词表顺序
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            
            # 步骤 E: 将被砍掉的词的 logits 设为负无穷大 (-inf)
            next_token_logits[indices_to_remove] = float('-inf')

        # 3. 将最终的 logits 转化为概率，并进行多项式采样 (抽奖)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # ==========================================
        # 要求 1: 拼接与终止条件
        # ==========================================
        # 拼接生成的新词
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # 检查是否命中了终止符 <|endoftext|>
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return input_ids