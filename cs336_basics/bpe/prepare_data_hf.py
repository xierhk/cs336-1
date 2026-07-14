import os
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

def main():
    data_dir = "/home/cs336/projects/cs336-assignment1/data"
    input_txt = os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt")
    output_bin = os.path.join(data_dir, "train.bin")
    tokenizer_json = os.path.join(data_dir, "tokenizer.json")
    
    print("🚀 [1/3] 加载分词器...")
    tokenizer = Tokenizer.from_file(tokenizer_json)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    
    print(f"🎯 <|endoftext|> 的 ID 为: {eot_id}")
    
    # 块大小：每次塞给 Rust 引擎 10000 行文本进行多线程并发计算
    chunk_size = 10000
    batch_lines = []
    
    print(f"⏳ [2/3] 开始多线程极速 Encode 并流式写入硬盘: {output_bin}")
    
    # 使用 ab (append binary) 模式，流式写入，内存占用趋近于 0
    with open(output_bin, "wb") as f_out: # 注意：第一次打开用 wb 清空旧文件，之后可以用 ab
        with open(input_txt, "r", encoding="utf-8") as f_in:
            for line in tqdm(f_in, desc="读取进度"):
                # 如果是空行，我们不处理，直接加一个结束符
                if not line.strip():
                    np.array([eot_id], dtype=np.uint16).tofile(f_out)
                    continue
                    
                batch_lines.append(line)
                
                # 当攒够一个 Chunk 时，发射给 Rust 引擎！
                if len(batch_lines) >= chunk_size:
                    # encode_batch 会自动启动 CPU 所有核心
                    encoded_batch = tokenizer.encode_batch(batch_lines)
                    
                    # 取出结果，在句尾加上 <|endoftext|>，并写入硬盘
                    for enc in encoded_batch:
                        ids = enc.ids + [eot_id]
                        np.array(ids, dtype=np.uint16).tofile(f_out)
                        
                    batch_lines = [] # 清空列表，释放内存
            
            # 处理循环结束时剩下的小尾巴
            if batch_lines:
                encoded_batch = tokenizer.encode_batch(batch_lines)
                for enc in encoded_batch:
                    ids = enc.ids + [eot_id]
                    np.array(ids, dtype=np.uint16).tofile(f_out)

    print(f"✅ [3/3] 大功告成！全量训练数据已序列化为 {output_bin}")
    
    # 最后验证一下生成的文件大小
    file_size_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print(f"📊 train.bin 文件大小: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    main()