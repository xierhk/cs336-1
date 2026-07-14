from __future__ import annotations
from cs336_basics.bpe.get_tokenizer import Tokenizer
from cs336_basics.bpe.run_train_bpe import train_bpe
from cs336_basics.layers import (
    Linear as MyLinear,
    Embedding as MyEmbedding,
    RMSNorm as MyRMSNorm,
    SwiGLU as MySwiGLU,
    RotaryPositionalEmbedding as RoPE,
    safe_softmax,
    cross_entropy,
    )
from cs336_basics.attention import (
    scaled_dot_product_attention,
    multihead_self_attention,
    multihead_self_attention_with_rope,
    )

from cs336_basics.transformer import transformer_block,transformer_lm

from cs336_basics.optimizer import AdamW,lr_cosine_schedule,gradient_clipping

from cs336_basics.data import get_batch

from cs336_basics.serialization import save_checkpoint,load_checkpoint

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn

# layers.py
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    in_features=in_features.cuda()
    linear=MyLinear(in_features=d_in,out_features=d_out,device='cuda')
    state_dict_to_load={"weight":weights}
    linear.load_state_dict(state_dict_to_load)
    return linear(in_features)

# layers.py
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    embedding=MyEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
        device='cuda',
        )
    state_dict_to_load={"weight":weights}
    embedding.load_state_dict(state_dict_to_load)
    return embedding(token_ids)



def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    d_ff = (int(2 * d_model * 4 / 3)+63)//64*64
    swiglu=MySwiGLU(d_model,d_ff,device='cuda')
    in_features=in_features.cuda()
    swiglu.w1.weight.data = w1_weight.cuda()
    swiglu.w2.weight.data = w2_weight.cuda()
    swiglu.w3.weight.data = w3_weight.cuda()
    return swiglu(in_features)
    


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    
    return scaled_dot_product_attention(Q,K,V,mask)
    


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    mha=multihead_self_attention(d_model,num_heads,'cuda')
    mha.qkv_proj.weight=nn.Parameter(torch.cat([q_proj_weight.to('cuda'),k_proj_weight.to('cuda'),v_proj_weight.to('cuda')],dim=0))
    mha.o_proj.weight.data=o_proj_weight.to('cuda')
    return mha(in_features.to('cuda'))



def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    mha=multihead_self_attention_with_rope(
        d_model,
        num_heads,
        theta,
        max_seq_len,
        'cuda'
        )
    mha.qkv_proj.weight=nn.Parameter(torch.cat([q_proj_weight.to('cuda'),k_proj_weight.to('cuda'),v_proj_weight.to('cuda')],dim=0))
    mha.o_proj.weight.data=o_proj_weight.to('cuda')
    return mha(in_features.to('cuda'),token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    rope=RoPE(theta,d_k,max_seq_len,'cuda')
    in_query_or_key=in_query_or_key.cuda()
    token_positions=token_positions.cuda()
    return rope(in_query_or_key,token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    in_features=in_features.to('cuda')
    weights = {k: v.to('cuda') for k, v in weights.items()}
    t_block=transformer_block(d_model,num_heads,d_ff,max_seq_len,theta,'cuda')
    t_block.ffn.w1.weight.data=weights['ffn.w1.weight']
    t_block.ffn.w2.weight.data=weights['ffn.w2.weight']
    t_block.ffn.w3.weight.data=weights['ffn.w3.weight']
    w_q = weights['attn.q_proj.weight']
    w_k = weights['attn.k_proj.weight']
    w_v = weights['attn.v_proj.weight']
    t_block.mha.qkv_proj.weight.data=torch.cat([w_q,w_k,w_v],dim=0)
    t_block.mha.o_proj.weight.data=weights['attn.output_proj.weight']
    t_block.norm1.weight.data=weights['ln1.weight']
    t_block.norm2.weight.data=weights['ln2.weight']

    return t_block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:

    # 1. 实例化这台巨型机器
    model = transformer_lm(
        vocab_size, context_length, d_model, num_layers, 
        num_heads, d_ff, rope_theta
    )
    
    # 2. 对齐设备
    device = in_indices.device
    model.to(device)
    weights = {k: v.to(device) for k, v in weights.items()}
    
    # ====================================================
    # 3. 终极装配！(The Checkpoint Converter)
    # 只有做了 QKV 融合，才需要写这段转换。如果你完全按字典拆分，就不用写。
    # ====================================================
    new_state_dict = {}
    
    # A. 顶层和底层的三个孤独的参数
    new_state_dict['token_embeddings.weight'] = weights['token_embeddings.weight']
    new_state_dict['ln_final.weight'] = weights['ln_final.weight']
    new_state_dict['lm_head.weight'] = weights['lm_head.weight']
    
    # B. 循环处理所有的中间层 (num_layers)
    for i in range(num_layers):
        layer_prefix = f'layers.{i}.'
        my_prefix = f'blocks.{i}.' # 你代码里叫 blocks
        
        # 名字一样的，直接改个前缀
        new_state_dict[f'{my_prefix}norm1.weight'] = weights[f'{layer_prefix}ln1.weight']
        new_state_dict[f'{my_prefix}norm2.weight'] = weights[f'{layer_prefix}ln2.weight']
        new_state_dict[f'{my_prefix}mha.o_proj.weight'] = weights[f'{layer_prefix}attn.output_proj.weight']
        new_state_dict[f'{my_prefix}ffn.w1.weight'] = weights[f'{layer_prefix}ffn.w1.weight']
        new_state_dict[f'{my_prefix}ffn.w2.weight'] = weights[f'{layer_prefix}ffn.w2.weight']
        new_state_dict[f'{my_prefix}ffn.w3.weight'] = weights[f'{layer_prefix}ffn.w3.weight']
        
        # QKV 融合！(注意 dim=0)
        new_state_dict[f'{my_prefix}mha.qkv_proj.weight'] = torch.cat([
            weights[f'{layer_prefix}attn.q_proj.weight'],
            weights[f'{layer_prefix}attn.k_proj.weight'],
            weights[f'{layer_prefix}attn.v_proj.weight']
        ], dim=0)

    # 4. 一键注入灵魂！
    model.load_state_dict(new_state_dict, strict=True)
    
    # 5. 点火起飞！
    with torch.no_grad():
        logits = model(in_indices)
        
    return logits


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    in_features=in_features.cuda()
    rmsnorm=MyRMSNorm(d_model,eps,device='cuda')
    state_dict_to_load={"weight":weights}
    rmsnorm.load_state_dict(state_dict_to_load)
    return rmsnorm(in_features)

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features*torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(dataset,batch_size,context_length,device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    
    return safe_softmax(in_features,dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    return cross_entropy(inputs,targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    return gradient_clipping(parameters,max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    
    return AdamW
    

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return lr_cosine_schedule(it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    save_checkpoint(model,optimizer,iteration,out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    return load_checkpoint(src,model,optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return train_bpe(input_path,vocab_size,special_tokens)

