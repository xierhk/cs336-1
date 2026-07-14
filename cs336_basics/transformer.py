from torch import nn,Tensor
from cs336_basics.layers import(
    RMSNorm,
    SwiGLU,
    Embedding,
    Linear
)
from cs336_basics.attention import multihead_self_attention_with_rope
import torch

class transformer_block(nn.Module):
    def __init__(
    self,
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    device,
):
        super().__init__()
        
        self.ffn=SwiGLU(d_model,d_ff,device)
        self.mha=multihead_self_attention_with_rope(d_model,num_heads,theta,max_seq_len,device)
        self.norm1=RMSNorm(d_model,0.00001,device)
        self.norm2=RMSNorm(d_model,0.00001,device)


    def forward(self, X: Tensor, token_positions: Tensor = None):
        
        # 1. 兜底逻辑：如果外部没传（比如跑单一 Block 测试），我才自己造
        if token_positions is None:
            batch_size, seq_len, _ = X.shape
            positions = torch.arange(seq_len, device=X.device)
            token_positions = positions.unsqueeze(0).expand(batch_size, -1)
            
        # 2. 如果外部传了（比如总装 LM），直接拿来用！
        block1 = X + self.mha(self.norm1(X), token_positions)
        block2 = block1 + self.ffn(self.norm2(block1))
        
        return block2
    

class transformer_lm(nn.Module):
    def __init__(self,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        # 1.词嵌入
        self.token_embeddings=Embedding(num_embeddings=vocab_size,embedding_dim=d_model,device=device)
        
        # 2.多个trans块
        self.blocks = nn.ModuleList([
            transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device
            ) for _ in range(num_layers)
        ])
        
        
        # 3. 输出收尾阶段 (Output Head)
        # ==========================================
        # 最后的全局归一化
        self.ln_final = RMSNorm(d_model, eps=1e-5,device=device)
        
        # 语言模型的分类头 (把特征映射回庞大的词表空间)
        # 很多大模型（如 GPT-2, LLaMA）这里默认是没有 bias 的
        self.lm_head = Linear(d_model, vocab_size,device=device)

    def forward(self, in_indices: Tensor) -> Tensor:
        """
        in_indices 形状: (batch_size, sequence_length)
        """
        batch_size, seq_len = in_indices.shape
        device = in_indices.device
        
        # 防御性检查：确保输入的序列长度没有超过模型支持的最大上下文
        assert seq_len <= self.context_length, f"Input sequence length {seq_len} exceeds max context length {self.context_length}"

        # ==========================================
        # 🌟 第一步：生成全局绝对位置信息 (核心枢纽！)
        # 这是我们上一关重点讨论的，顶层负责造轮子！
        # ==========================================
        positions = torch.arange(seq_len, device=device)
        token_positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # ==========================================
        # 第二步：输入嵌入
        # 从 (batch_size, seq_len) 变成 (batch_size, seq_len, d_model)
        # ==========================================
        x = self.token_embeddings(in_indices)
        
        # ==========================================
        # 第三步：闯塔！穿越所有的 Transformer 层
        # 每一层都要把位置信息透传进去，给 RoPE 做校准
        # ==========================================
        for block in self.blocks:
            x = block(x, token_positions)
            
        # ==========================================
        # 第四步：收尾与输出
        # ==========================================
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        # 返回未归一化的概率分布，形状: (batch, seq_len, vocab_size)
        return logits