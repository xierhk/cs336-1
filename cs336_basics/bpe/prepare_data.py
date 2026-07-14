import os
import argparse
import numpy as np
from get_tokenizer import Tokenizer # 确保这里导入你的分词器类名是对的

def main():
    parser = argparse.ArgumentParser(description="语料库二进制压缩引擎 (双通道版)")
    
    # 🌟 核心改动：把一个 input 拆成了两个明确的独立输入
    parser.add_argument("--train_input", type=str, required=True, help="原始训练集 txt 路径")
    parser.add_argument("--val_input", type=str, required=True, help="原始验证集 txt 路径")
    
    parser.add_argument("--output_dir", type=str, default="data/test/token", help="输出 bin 文件的目录")
    parser.add_argument("--vocab_file", type=str, default="data/test/tokenizer/vocab.json", help="vocab.json 路径")
    parser.add_argument("--merges_file", type=str, default="data/test/tokenizer/merges.json", help="merges.json 路径")
    
    args = parser.parse_args()

    # 路径防爆处理
    train_input_abs = os.path.abspath(args.train_input)
    val_input_abs = os.path.abspath(args.val_input)
    output_dir_abs = os.path.abspath(args.output_dir)
    vocab_file_abs = os.path.abspath(args.vocab_file)
    merges_file_abs = os.path.abspath(args.merges_file)

    os.makedirs(output_dir_abs, exist_ok=True)
    train_bin_path = os.path.join(output_dir_abs, "train.bin")
    val_bin_path = os.path.join(output_dir_abs, "valid.bin")

    # ==========================================
    # 1. 唤醒分词器引擎 (使用最新的 JSON 接口)
    # ==========================================
    print(f"⚙️ 正在加载 Tokenizer 引擎...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_file_abs, 
        merges_filepath=merges_file_abs, 
        special_tokens=["<|endoftext|>"]
    )
    print("✅ Tokenizer 唤醒成功！")

    # ==========================================
    # 2. 核心压缩函数
    # ==========================================
# ==========================================
    # 2. 核心压缩函数 (流式分块防爆版)
    # ==========================================
    def process_and_save(txt_path: str, bin_path: str, split_name: str):
        from tqdm import tqdm # 确保引入了进度条
        
        print(f"\n📖 正在流式读取 {split_name} 语料: {txt_path}")
        
        total_tokens = 0
        chunk_lines = []
        chunk_size = 10000  # 每次处理 10000 行（吃进去一点）
        
        # 🌟 核心：读模式打开纯文本，写模式(wb)打开二进制
        with open(txt_path, "r", encoding="utf-8") as f_in, open(bin_path, "wb") as f_out:
            
            # 使用 tqdm 包装 f_in 提供进度条感
            for line in tqdm(f_in, desc=f"🚀 压缩 {split_name}"):
                chunk_lines.append(line)
                
                # 当攒够 10000 行时，执行一次分词并落盘
                if len(chunk_lines) >= chunk_size:
                    text_chunk = "".join(chunk_lines)
                    token_ids = tokenizer.encode(text_chunk)
                    
                    if token_ids:
                        # 转成 uint16 并直接写进物理硬盘（吐出来一点）
                        arr = np.array(token_ids, dtype=np.uint16)
                        f_out.write(arr.tobytes())
                        total_tokens += len(token_ids)
                        
                    # 清空内存，准备吃下一口
                    chunk_lines = []
            
            # 🌟 扫尾工作：处理最后不足 10000 行的残余数据
            if chunk_lines:
                text_chunk = "".join(chunk_lines)
                token_ids = tokenizer.encode(text_chunk)
                if token_ids:
                    arr = np.array(token_ids, dtype=np.uint16)
                    f_out.write(arr.tobytes())
                    total_tokens += len(token_ids)
                    
        print(f"✅ {split_name} 编码完成，共计 {total_tokens:,} 个 Tokens.")
        print(f"💾 {split_name} 已硬核写入: {bin_path} (大小: {os.path.getsize(bin_path) / 1024 / 1024:.2f} MB)")

    # ==========================================
    # 3. 独立处理两份文件
    # ==========================================
    process_and_save(train_input_abs, train_bin_path, "训练集 (Train)")
    process_and_save(val_input_abs, val_bin_path, "验证集 (Valid)")
    
    print("\n🎉 全线贯通！你的 train.bin 和 valid.bin 已经准备完毕！")

if __name__ == "__main__":
    main()