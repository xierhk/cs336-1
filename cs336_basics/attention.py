from cs336_basics.layers import (
    safe_softmax,
    Linear as MyLinear,
    RotaryPositionalEmbedding,
    )
from torch import Tensor,nn
from jaxtyping import Bool, Float,Int
import torch,math
from einops import einsum,rearrange

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
    ):
    dot=einsum(Q,K,'... queries d_k,... keys d_k->... queries keys')
    d_k=Q.size(-1)
    dot/=math.sqrt(d_k)
    if mask is not None:
        dot.masked_fill_(mask==False,-torch.inf)
    
    return safe_softmax(dot,-1)@V


class multihead_self_attention(nn.Module):
    def __init__(self,d_model: int,num_heads: int,device:None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.qkv_proj=MyLinear(d_model,d_model*3,device)
        self.o_proj=MyLinear(d_model,d_model,device)

    def forward(self,X):
        seq_len=X.size(-2)
        qkv = self.qkv_proj(X)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        
        # 4. 生成下三角因果掩码 (Causal Mask)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=X.device))
        
        # 5. 点积、缩放、打掩码、Softmax
        out=scaled_dot_product_attention(q,k,v,mask)
        
        # 6. 特征提取与多头拼接
        out = rearrange(out, 'b h s d -> b s (h d)')
        
        # 7. 线性映射输出
        return self.o_proj(out)
    
class multihead_self_attention_with_rope(nn.Module):
    def __init__(self,d_model: int,num_heads: int,theta: float,max_seq_len: int,device):
        super().__init__()
        self.qkv_proj=MyLinear(d_model,d_model*3,device)
        self.o_proj=MyLinear(d_model,d_model,device)
        self.num_heads=num_heads
        self.d_model=d_model
        self.theta=theta
        self.max_seq_len=max_seq_len
        d_k = d_model // num_heads

        self.rope = RotaryPositionalEmbedding(self.theta, d_k, self.max_seq_len,device)

    def forward(self,X,token_positions: Int[Tensor, " ... sequence_length"]):
        seq_len=X.size(-2)
        qkv=self.qkv_proj(X)
        q,k,v=torch.chunk(qkv,chunks=3,dim=-1)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=X.device))
        q=rearrange(q,'b ... (h d) -> b h ... d',h=self.num_heads)
        k=rearrange(k,'b ... (h d) -> b h ... d',h=self.num_heads)
        v=rearrange(v,'b ... (h d) -> b h ... d',h=self.num_heads)
        q_rope=self.rope(q,token_positions,1)
        k_rope=self.rope(k,token_positions,1)
        out=scaled_dot_product_attention(q_rope,k_rope,v,mask)
        # 拼接多头
        out = rearrange(out, 'b h s d -> b s (h d)')
        
        # 线性映射输出
        return self.o_proj(out)
