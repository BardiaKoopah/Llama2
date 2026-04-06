import os
import sentencepiece as spm
from typing import List, Optional
from datasets import load_dataset


class LlamaTokenizer:

    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), f"Tokenizer model not found: {model_path}"
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

        self.vocab_size: int = self.sp.GetPieceSize()
        self.bos_id: int = self.sp.bos_id()
        self.eos_id: int = self.sp.eos_id()
        self.pad_id: int = self.sp.pad_id() if self.sp.pad_id() >= 0 else self.eos_id

    @classmethod
    def train_from_corpus(
        cls,
        corpus_path: str,
        vocab_size: int = 32_000,
        model_prefix: str = "llama_tok",
    ) -> "LlamaTokenizer":
        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            byte_fallback=True,
            bos_id=1,
            eos_id=2,
            pad_id=0,
            unk_id=3,
            split_digits=True,
            normalization_rule_name="identity",
        )
        return cls(f"{model_prefix}.model")

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> List[int]:
        tokens = self.sp.Encode(text)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, ids: List[int]) -> str:
        return self.sp.Decode(ids)

    def encode_batch(self, texts: List[str], bos: bool = True, eos: bool = False) -> List[List[int]]:
        return [self.encode(t, bos=bos, eos=eos) for t in texts]

    def __len__(self) -> int:
        return self.vocab_size

if __name__ == "__main__":
    path = '/Users/bardia/Desktop/llama2/llama_tok.model'

    if not os.path.isfile(path):
        corpus_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train+validation+test")
        with open("corpus.txt", "w", encoding="utf-8") as f:
            for line in corpus_dataset["text"]:
                if line.strip():
                    f.write(line + "\n")
        LlamaTokenizer.train_from_corpus(corpus_path='corpus.txt')
    else:
        print('TOKENIZER EXISTS!')