import torch
from typing import Optional
from tokenizer import LlamaTokenizer


class LlamaDataLoader:

    def __init__(
        self,
        corpus_path: str,
        tokenizer: LlamaTokenizer,
        seq_len: int,
        batch_size: int,
        device: str = "cpu",
        shuffle: bool = True,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        all_ids = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids = tokenizer.encode(line, bos=False, eos=False)
                    all_ids.extend(ids)

        all_ids = [tokenizer.bos_id] + all_ids + [tokenizer.eos_id]
        self.data = torch.tensor(all_ids, dtype=torch.long)

        self.n_windows = (len(self.data) - 1) // self.seq_len
        self.data = self.data[: self.n_windows * self.seq_len + 1]

        self._indices: Optional[torch.Tensor] = None
        self._pos = 0
        self.reset()

    def reset(self):
        self._indices = torch.arange(self.n_windows)
        if self.shuffle:
            self._indices = self._indices[torch.randperm(self.n_windows)]
        self._pos = 0

    def __len__(self) -> int:
        return self.n_windows // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos + self.batch_size > len(self._indices):
            raise StopIteration

        batch_idx = self._indices[self._pos : self._pos + self.batch_size]
        self._pos += self.batch_size

        starts = batch_idx * self.seq_len
        x = torch.stack([self.data[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])

        return x.to(self.device), y.to(self.device)
