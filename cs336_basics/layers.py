from einops import einsum,reduce,rearrange,repeat
from jaxtyping import Bool, Float, Int
import torch.nn as nn
import torch
from torch import Tensor
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        # 必须有nn.Parameter后面调用才能调到weight
        self.weight=nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))
        std_val=math.sqrt(2.0/(in_features+out_features))
        nn.init.trunc_normal_(
            self.weight,
            std=std_val,
            mean=0.0,
            a=-3*std_val,
            b=3*std_val
            )
    def forward(self,X):
        # 内部输入的x肯定是行向量，但是数学推导(Wx)用列向量更方便
        return torch.einsum("... i,o i -> ... o",X,self.weight)


# pdf-第20页
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        nn.init.trunc_normal_(
            tensor=self.weight,
            mean=0.0,
            std=1.0,
            a=-3,
            b=3
            )
        
    def forward(self, token_ids: torch.Tensor):  
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):#run_rmsnorm
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps=eps
        self.d_model=d_model
        self.weight=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))

    def forward(self,x:torch.Tensor):
        #转换类型，防止溢出
        in_dtype=x.dtype
        x=x.to(torch.float32)
        #后面加了1保证维度不变，
        variance=reduce(x.pow(2),"... d_model -> ... 1","mean")
        x=x*torch.rsqrt(variance+self.eps) #逐元素运算
        # 用*广播，一维张量默认是最后一个维度
        # (32, 128, 4096)*(4096,)=(32, 128, 4096)*(1,1,4096)再广播
        return x.to(in_dtype)* self.weight


class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,device=None,dtype=None):
        super().__init__()
        self.w1=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2=Linear(d_ff,d_model,device=device,dtype=dtype)
        self.w3=Linear(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,X):
        w1_out=self.w1(X)
        return self.w2(w1_out*torch.sigmoid(w1_out)*self.w3(X))
    

def rotate_half(x:Tensor):
    # 此时 x1 和 x2 的形状都是 (..., d)
    x1,x2=rearrange(x,"... (d c) -> c ... d",c=2)
    # 魔法 2：合并展平。将 [-x2, x1] 堆叠后，把最后的 d 和 c(2) 融合回原维度
    # stack会让他多一个维度
    return rearrange(torch.stack((-x2,x1),dim=-1),'... d c -> ... (d c)')

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # 构造角度矩阵
        inv_freq = torch.exp(
                    -torch.arange(0, d_k, 2, device=device, dtype=torch.float32) * (math.log(theta) / d_k)
                )
        t=torch.arange(0,max_seq_len,device=device,dtype=torch.float32)
        
        #得到角度
        freqs=torch.outer(t,inv_freq)

        # 由于每两个位置共用一个角度，需要复制一份
        freqs_paired=repeat(freqs,'... d -> ... (d c)',c=2)

        self.register_buffer('cos_cached',freqs_paired.cos(),persistent=False)
        self.register_buffer('sin_cached',freqs_paired.sin(),persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor,unsqueeze_dim:int=0):

        cos=self.cos_cached[token_positions]
        sin=self.sin_cached[token_positions]
        if unsqueeze_dim==1:
            cos=cos.unsqueeze(1)
            sin=sin.unsqueeze(1)
        elif unsqueeze_dim==2:
            cos=cos.unsqueeze(2)
            sin=sin.unsqueeze(2)
        return x*cos+rotate_half(x)*sin


def safe_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 1. 找到该维度上的最大值。
    # 必须使用 keepdim=True，保证形状依然是 (..., 1, ...) 而不是 (..., ...)
    # .max() 会返回 (values, indices) 的元组，我们只需要 values，所以取 [0]
    max_val=x.max(dim=dim,keepdim=True)[0]
    
    # 2. Max-Shift 黑魔法：所有元素减去最大值
    # 此时 x_safe 里的最大值变成了 0，其余全是负数
    x_safe = x - max_val
    
    # 3. 安全计算指数：因为最大是 exp(0)=1，绝对不会再发生 Inf 溢出！
    exp_x = torch.exp(x_safe)
    
    # 4. 计算分母，同样需要 keepdim=True 保证能正确相除
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    # 5. 归一化，得到概率分布
    return exp_x / sum_exp


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
):
    # 这里的batch_size由batch_size*seq_len所得
    batchsize=inputs.size(0)

    # 每一行计算作为softmax分母

    #分母每一行的最大值
    m = reduce(inputs, "... hidden -> ... 1", "max")
    
    x=inputs-m #[b,v]

    # 分母,[b]
    logx_rowsum=torch.log(reduce(torch.exp(x),'b v -> b','sum'))


    # 分子 高级索引，x[行索引集合, 列索引集合]
    x=inputs[torch.arange(batchsize),targets]
    
    return (logx_rowsum-x+m.squeeze(-1)).mean()
