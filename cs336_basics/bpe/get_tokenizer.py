import re
from typing import Iterator, Iterable

import regex

class Tokenizer:
    # 🌟 官方 GPT-2 预分词正则表达式
    # 它的作用是把长句按标点、空格、字母类型极其严谨地切成碎块
    GPT2_PAT = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def __init__(self, vocab, merges, special_tokens=None):
        # ... 保持不变
        self.vocab = vocab
        
        # ⚡ 双重保险：确保 self.merges 绝对是字典，将 inside 查找优化到 O(1) 闪电速度
        if isinstance(merges, list):
            self.merges = {pair: i for i, pair in enumerate(merges)}
        else:
            self.merges = merges
            
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        import json
        
        # 1. 直接用 Python 官方的 json 库读取，彻底抛弃旧的 parse_vocab！
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
            
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            raw_merges = json.load(f)
            
        # 🌟 2. 逆向还原装甲：
        # 将 JSON 的 {str: str} 完美还原为底层的 {int: bytes}
        parsed_vocab = {int(k): v.encode("latin-1") for k, v in raw_vocab.items()}
        
        # 将 JSON 的 [[str, str]] 完美还原为底层的 [(bytes, bytes)]
        parsed_merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in raw_merges]
        
        return cls(vocab=parsed_vocab, merges=parsed_merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
            if not text:
                return []
                
            # 1. 依然保留你的完美特殊字符切分
            if self.special_tokens:
                sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
                # 注意这里把 re 换成了 regex
                special_pattern = "|".join(map(regex.escape, sorted_specials))
                pattern = regex.compile(f"({special_pattern})")
                parts = pattern.split(text)
            else:
                parts = [text]
                
            ids = []
            for part in parts:
                if not part:
                    continue
                
                if self.special_tokens and part in self.special_tokens:
                    # 特殊字符，直接查表
                    ids.append(self.vocab_reverse[part.encode("utf-8")])
                else:
                    # 🌟 2. 终极修复：在 BPE 合并前，用 GPT-2 正则把文本切成块！
                    # 比如 "\n\nA" 会被切成 ['\n', '\n', 'A']
                    # 这彻底阻止了 BPE 贪婪地把它们合并成 \n\n
                    for chunk in self.GPT2_PAT.findall(part):
                        ids.extend(self._encode_chunk(chunk))
                        
            return ids

    def _encode_chunk(self, text: str) -> list[int]:
        if not text:
            return []
            
        tokens = [bytes([b]) for b in text.encode("utf-8")]
        
        while True:
            if len(tokens) < 2:
                break

            pairs = set(zip(tokens, tokens[1:]))
            # 💡 只要 merges 是字典，这里的 in 判断就是瞬发，绝不卡死！
            valid_pairs = [p for p in pairs if p in self.merges]
            
            if not valid_pairs:
                break
                
            best_pair = min(valid_pairs, key=lambda p: self.merges[p])
            
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return [self.vocab_reverse[token] for token in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        combined_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return combined_bytes.decode("utf-8", errors="replace")

# 🚨🚨🚨 请检查你文件里是不是有这个函数，确保给它加上 special_tokens=special_tokens！
