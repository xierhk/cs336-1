import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def main():
    data_dir = "/home/cs336/projects/cs336-assignment1/data"
    input_txt = os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt")
    output_json = os.path.join(data_dir, "tokenizer.json")
    
    print("🚀 [1/3] 初始化 Hugging Face Tokenizer (Rust 引擎)...")
    tokenizer = Tokenizer(BPE())
    # 采用 GPT 系列标准的 Byte-Level 预处理，完美解决各种乱码和生僻字
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 配置训练器：词表大小 10000，且把休止符放在第一个（ID=0）
    trainer = BpeTrainer(
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        show_progress=True
    )

    print("⏳ [2/3] 开始统计词频并训练 BPE...")
    # 架构师技巧：只取前 100 万行训练分词器，不仅速度极快，而且词频统计已经绝对准确了（符合齐普夫定律）
    def get_training_corpus():
        with open(input_txt, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 1_000_000:
                    break
                yield line.strip()

    # 启动训练
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    print(f"💾 [3/3] 保存模型至 {output_json}...")
    tokenizer.save(output_json)
    
    # 抽查测试
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    print(f"✅ 完成！词表大小: {tokenizer.get_vocab_size()}")
    print(f"🔍 测试: <|endoftext|> 的 ID 是 {eot_id}")

if __name__ == "__main__":
    main()