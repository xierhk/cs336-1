import os
import pickle
import regex as re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from cs336_basics.pretokenization_example import find_chunk_boundaries
from pathlib import Path
import multiprocessing as mp


# 1-4.预分词，分块读取构建wf
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
#1.1建立子进程
def work_process_chunk(task:tuple):
    input_path,start,end,special_tokens=task#str,int,int,List[str]
    local_wf=Counter()

    escaped_tokens=[re.escape(i) for i in special_tokens]
    if escaped_tokens:
        split_pattern=re.compile(f"({'|'.join(escaped_tokens)})")
    else:
        split_pattern=None
    
    with open(input_path,"rb") as f:
        f.seek(start)
        raw_bytes=f.read(end-start)

        text_chunk=raw_bytes.decode("utf-8",errors="ignore")

        if split_pattern:
            text_segments=split_pattern.split(text_chunk)
        else:
            text_segments=[text_chunk]
        
        #segment是被分隔符分隔的最小单元
        for segment in text_segments:
            if not segment:
                continue

            if segment in special_tokens:
                continue

            for match in GPT2_PATTERN.finditer(segment):
                word_str=match.group()
                local_wf[tuple(word_str.encode("utf-8"))]+=1
            
    return local_wf

#1.2 分块，并调用子进程返回wf
def generate_wf_multiprocess(input_path:str,special_tokens:list[str],
                             num_workers:int=4):
    if not os.path.exists(input_path):
        raise FileNotFoundError
    
    with open(input_path,'rb') as f:
        #special_tokens[0] 是用来划分最大块的
        split_token_bytes=special_tokens[0].encode("utf_8") if special_tokens else b"\n"
        boundaries=find_chunk_boundaries(f,num_workers*2,split_token_bytes)

    task=[]
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        task.append((input_path,start,end,special_tokens))
    
    with mp.Pool(processes=num_workers) as pool:
        results=pool.map(work_process_chunk,task)
    
    global_wf=Counter()
    for local_wf in results:
        global_wf.update(local_wf)

    return global_wf

# 用新单词替换旧单词
def list_replace_pair(old_w:tuple, top_pair:tuple, new_token:int):
    #('l','o','w')换为(b'lo',w)
    l=[]
    i=0
    while(i<len(old_w)):
        if i<len(old_w)-1 and (old_w[i],old_w[i+1])==top_pair:
            l.append(new_token)
            i+=2
        else:
            l.append(old_w[i])
            i+=1
    return tuple(l)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
给定输入语料库的路径，训练一个 BPE 分词器，并输出其词汇表和合并规则。
参数:
    input_path (str | os.PathLike): BPE 分词器训练数据的路径。
    vocab_size (int): 分词器词汇表中的总项数（包含特殊词元）。
    special_tokens (list[str]): 要添加到分词器词汇表中的特殊词元（字符串列表）。
        这些字符串永远不会被拆分为多个词元，而会始终作为一个单一的词元保留。
        如果这些特殊词元出现在 `input_path` 的文本中，它们将与其他普通字符串一样被同等对待。

返回值:
    tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab (词汇表):
            训练好的分词器词汇表，这是一个从 int（词汇表中的词元 ID）到 bytes（词元的字节数据）的映射。
        merges (合并规则):
            BPE 合并规则列表。列表中的每一项都是一个字节元组 (<词元1>, <词元2>)，
            表示 <词元1> 与 <词元2> 进行了合并。
            合并规则按照创建的先后顺序进行排列。
    """
    #0.初始化词表,加入special_tokens
    vocab=defaultdict(bytes,{i:bytes([i]) for i in range (256)})
    vocab.update({256+i:token.encode("utf-8") for i,token in enumerate (special_tokens)})


    #1.1分块读取大文件
    #1.2对每个块进行split()，multiprocessing，先escape再split
    #1.3对每个split之后的文件进行finditer，multiprocessing
    #1.4对每个finditer之后的文件构建元组字典记为wf,元素形如('s','l', 'o', 'w'): 5,
    
    wf=generate_wf_multiprocess(input_path,special_tokens,os.cpu_count()-1)
     
    #2.根据wf构建一个字典tf，元素形如('l','o'):5,('o','w'):5，若为空则新增，不空则增加values的值
    #3.在构建tf的同时建立反向索引tw('l','o'):('l', 'o', 'w'),代表wf中'l''o'隶属的key
    tf=Counter()
    tw = defaultdict(set)
    for w,f in wf.items():
        for i in range(len(w)-1):
            tf[(w[i],w[i+1])]+=f
            tw[((w[i],w[i+1]))].add(w)

    #4.找出tf中最大values的key，假如为('l','o')，则将其保存为a，删除('l','o')的对应项后将tokenid:b'lo'加入vocab中，merge中也要增加元素 
    #5.找出tw中a对应的values记为b，将tw中这一项删除，修改wf中key为b的键值对，此时要往tf和tw中都增加元素。
        #5.1 看('l','o')对应的单词，假设有一个是('s','l', 'o', 'w')，记为oldword，其频次为fre
        #5.2 遍历oldword，产生三个"对子":('s','l'), ('l', 'o'), ('o', 'w')
        #5.3 对于tf，将"对子"的频次减去fre
        #5.4 对于tw，从"对子"的集合values中减去oldword
        #5.5在wf中删除oldword，增加newword:fre
        #5.6出现"新对子":('s',tokenid),(tokenid,'w')。newword:('s',256, 'w')
        #5.7 对于tf，增加('s',tokenid)出现的频率fre，增加(tokenid,'w')出现的频率fre
        #5.8对于tw,增加字典元素 "新对子":newword。
    #6.继续步骤4-5直到满足vocab_size
    merges=[]
    #4-6循环
    for i in range(len(vocab),vocab_size):
        if not tf:
            break
        
        # 1. 揪出当前全世界出现次数最多的对子
        top_pair = max(tf, key=lambda k: (tf[k], vocab[k[0]],vocab[k[1]]))
            # 调试：i=91 时打印 top 5 频次
        if 90 <= i <= 93:
            top5 = tf.most_common(5)
            print(f'step {i}: top_pair={top_pair}, freq={tf[top_pair]}')
            for p, f in top5:
                print(f'  {p}: {f} (bytes: {vocab[p[0]]!r}, {vocab[p[1]]!r})')
            print()

        # 新生的合并词元组合（比如把 (b'l',b'o') 融合成 b'lo'）
        token_a_bytes = vocab[top_pair[0]]#bytes
        token_b_bytes = vocab[top_pair[1]]
        #bytes相加类似str
        new_token = token_a_bytes + token_b_bytes
        
        # 2. 【核武器登场】我们不需要遍历百万个单词！
        # 只要顺着你写的 tw 地图，精准找到包含 top_pair 的单词集合即可
        words_to_update = list(tw[top_pair])
        #返回tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        new_tokenid=i
        merges.append((token_a_bytes,token_b_bytes))
        vocab[i]=new_token
        for old_w in words_to_update:
            # 先把旧单词在 wf 里的频次拿出来
            f = wf[old_w]
            
            # 3. 执行合并算法（在旧元组里找到 top_pair 并替换成 new_token）
            # 比如把 ('l', 'o', 'w') 变成 ('lo', 'w')
            new_w = list_replace_pair(old_w, top_pair, new_tokenid)
            
            #遍历old_w
            #一个单词可能出现两个'l','o'
            for j in range(len(old_w)-1):
                old_pair=(old_w[j],old_w[j+1])
                tf[old_pair]-=f
                if tf[old_pair]==0:
                    del tf[old_pair]  
                if old_w in tw[old_pair]:
                    tw[old_pair].remove(old_w)

            del wf[old_w]
            wf[new_w] = f

            #遍历new_w
            for j in range(len(new_w)-1):
                new_pair=(new_w[j],new_w[j+1])
                tf[new_pair]+=f
                tw[new_pair].add(new_w)
    #返回tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return (vocab,merges)

if __name__ == "__main__":
    import argparse
    import json  # 🌟 新增：导入 json 模块
    import os
    
    # ==========================================
    # 1. 建立命令行参数解析器
    # ==========================================
    parser = argparse.ArgumentParser(description="BPE 分词器训练脚本")
    
    # 核心路径参数 (添加默认值方便本地测试，云端可通过命令行覆盖)
    parser.add_argument("--input", type=str, required=True, help="原始纯文本语料的绝对路径")
    parser.add_argument("--output", type=str, default="data/test/tokenizer", help="词表保存的目录路径")
    
    # 超参数
    parser.add_argument("--vocab_size", type=int, default=10000, help="目标词表大小")
    
    args = parser.parse_args()

    # ==========================================
    # 2. 路径自适应装甲 (防爆处理)
    # ==========================================
    input_file_abs = os.path.abspath(args.input)
    output_dir_abs = os.path.abspath(args.output)

    if not os.path.exists(input_file_abs):
        print(f"❌ 致命错误: 找不到文件 {input_file_abs}")
        print("💡 提示: 如果在 Kaggle 上，请检查右侧 Input 面板里的路径是否拼写正确。")
        exit(1)

    print(f"🚀 开始训练 BPE 词表...")
    print(f"📂 读取语料: {input_file_abs}")
    print(f"🎯 目标词表大小: {args.vocab_size}")
    
    # ==========================================
    # 3. 启动核心算法
    # ==========================================
    vocab, merges = train_bpe(
        input_path=input_file_abs,
        vocab_size=args.vocab_size, 
        special_tokens=["<|endoftext|>"] 
    )
    
    print(f"✅ 训练完成！获得了 {len(vocab)} 个 Vocab 和 {len(merges)} 条 Merges 规则。")
    
    # ==========================================
    # 4. JSON 安全落盘保存 (防数据丢失版)
    # ==========================================
    os.makedirs(output_dir_abs, exist_ok=True)
    
    # 🌟 核心转换逻辑：
    # 1. Vocab: int 的 key 转成 str，bytes 的 value 用 latin-1 无损转成 str
    vocab_json = {str(k): v.decode("latin-1") for k, v in vocab.items()}
    
    # 2. Merges: tuple 转成 list，内部的 bytes 用 latin-1 转成 str
    merges_json = [[a.decode("latin-1"), b.decode("latin-1")] for a, b in merges]
    
    # 🌟 扩展名已更改为 .json
    vocab_path = os.path.join(output_dir_abs, "vocab.json")
    merges_path = os.path.join(output_dir_abs, "merges.json")
    
    # 写入 JSON，indent=2 保证人类可读性
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=2, ensure_ascii=False)
        
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_json, f, indent=2, ensure_ascii=False)
        
    print(f"💾 词表已成功转换为 JSON 格式，安全保存至: {output_dir_abs}")