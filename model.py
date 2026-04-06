import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(69)
global_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Embeddings(nn.Module):
    
    def __init__(self, vocab_size, d_model, device=global_device):
        super().__init__()
        self.embedded = nn.Embedding(vocab_size, d_model, device=device)
        with torch.no_grad():
            self.embedded.weight.normal_(mean=0, std=0.02)

    def forward(self, input):
        return self.embedded(input)

class RMSNorm(nn.Module):

    def __init__(self, d_model, device=global_device):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((d_model,), device=device))
    
    def forward(self, x):
        eps = torch.finfo(x.dtype).tiny
        numerator = x
        denominator = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        final = self.alpha * (numerator / denominator)
        return final

class GQA(nn.Module):

    def __init__(self, seq_len, d_model, n_kv_heads, num_heads, device=global_device, is_causal=True):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_kv_heads = n_kv_heads
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.groups = num_heads // self.n_kv_heads
        self.causal = is_causal
        
        self.Wq = nn.Linear(d_model, d_model, bias=False, device=device)
        self.Wk = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False, device=device)
        self.Wv = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False, device=device)
        self.Wo = nn.Linear(d_model, d_model, bias=False, device=device)

        #RoPE stuff so we don't have to keep recomputing
        pair_indices = torch.arange(self.d_k // 2, dtype=torch.float32).unsqueeze(-1).to(global_device)
        positions = torch.arange(self.seq_len).unsqueeze(-1).to(global_device)

        theta = 1 / (10000 ** ((2 * pair_indices) / self.d_k)).squeeze(-1).unsqueeze(0)
        m_theta = (positions * theta).unsqueeze(0).unsqueeze(1)
        self.register_buffer('theta', theta)
        self.register_buffer('m_theta', m_theta)

        mask = torch.triu(
            torch.full((seq_len, seq_len), fill_value=torch.finfo().min),
            diagonal=1
        ).view(1, 1, seq_len, seq_len).to(device=device)
        
        self.register_buffer('mask', mask)
    
    def RoPE(self, Q: torch.Tensor, K: torch.Tensor, cache):
        Q_copy = Q
        K_copy = K

        num_pairs = self.d_k // 2

        if cache:
            offset = cache[0].shape[2]
            actual_len = Q.shape[2]
            positions = torch.arange(offset, actual_len + offset, dtype=torch.float32).unsqueeze(-1).to(global_device)
            m_theta = (positions * self.theta).unsqueeze(0).unsqueeze(1)
        else:
            actual_len = Q.shape[2]
            m_theta = self.m_theta[:, :, :actual_len, :]

        Q_copy = torch.reshape(Q_copy, (Q_copy.shape[0], Q_copy.shape[1], Q_copy.shape[2], num_pairs, 2))
        K_copy = torch.reshape(K_copy, (K_copy.shape[0], K_copy.shape[1], K_copy.shape[2], num_pairs, 2))

        q0 = Q_copy[..., 0]
        q1 = Q_copy[..., 1]

        Q_rotated = torch.stack([
            q0 * torch.cos(m_theta) - q1 * torch.sin(m_theta),
            q0 * torch.sin(m_theta) + q1 * torch.cos(m_theta)
        ], dim=-1)

        k0 = K_copy[...,0]
        k1 = K_copy[...,1]

        K_rotated = torch.stack([
            k0 * torch.cos(m_theta) - k1 * torch.sin(m_theta),
            k0 * torch.sin(m_theta) + k1 * torch.cos(m_theta)
        ], dim=-1)

        Q_out = Q_rotated.reshape(Q.shape)
        K_out = K_rotated.reshape(K.shape)

        return Q_out, K_out

    def sdpa(self, Q, K, V):

        first_matmul = torch.divide(Q @ K.mT, self.d_k ** 0.5)
        
        if Q.shape[2] != 1:
            actual_len = Q.shape[2]
            mask = self.mask[:, :, :actual_len, :actual_len]
            masked = mask + first_matmul
        else:
            masked = first_matmul

        attention = torch.softmax(masked, dim=-1)

        output = attention @ V

        return output, attention

    def forward(self, x, cache=None):
        Q: torch.Tensor = self.Wq(x)
        K: torch.Tensor = self.Wk(x)
        V: torch.Tensor = self.Wv(x)

        B = x.shape[0]

        Q = torch.reshape(Q, (B, -1, self.num_heads, self.d_k)).transpose(1, 2)
        K = torch.reshape(K, (B, -1, self.n_kv_heads, self.d_k)).transpose(1,2)
        V = torch.reshape(V, (B, -1, self.n_kv_heads, self.d_k)).transpose(1,2)

        Q, K = self.RoPE(Q, K, cache)

        if cache is not None:
            K_cache = torch.cat([cache[0], K], dim=2)
            V_cache = torch.cat([cache[1], V], dim=2)
            cache = (K_cache, V_cache)
            K = K_cache
            V = V_cache
        else:
            cache = (K, V) if not self.training else None
        
        K = K.repeat_interleave(self.groups, dim=1)
        V = V.repeat_interleave(self.groups, dim=1)

        output, attention = self.sdpa(Q, K, V)

        reshaped = torch.transpose(output, 1, 2).reshape((B, -1, self.num_heads * self.d_k))

        final = self.Wo(reshaped)
        return final, cache

class SwiGLU(nn.Module):

    def __init__(self, seq_len, d_model, device=global_device):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        hidden_dims = int((4 * d_model) * (2 / 3))
        self.W1 = nn.Linear(d_model, hidden_dims, device=device, bias=False)
        self.W2 = nn.Linear(d_model, hidden_dims, device=device, bias=False)
        self.W3 = nn.Linear(hidden_dims, d_model, device=device, bias=False)

    def forward(self, x):
        swish_input = self.W1(x)
        swish = swish_input * torch.sigmoid(swish_input)
        return self.W3(swish * self.W2(x))
    

class DecoderBlock(nn.Module):

    def __init__(self, vocab_size, seq_len, d_model, num_heads, n_kv_heads, is_causal=True ,device=global_device):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads

        self.first_norm = RMSNorm(d_model=d_model, device=device)
        self.gqa = GQA(seq_len=seq_len, d_model=d_model, n_kv_heads=n_kv_heads, num_heads=num_heads, is_causal=is_causal, device=device)
        self.second_norm = RMSNorm(d_model=d_model, device=device)
        self.swiglu = SwiGLU(seq_len=seq_len, d_model=d_model, device=device)

    def forward(self, x, cache=None):
        first_skip = x
        first_rms = self.first_norm(x)
        attention, cache = self.gqa(first_rms, cache)

        first_skip_output = first_skip + attention
        second_skip = first_skip_output
        second_norm = self.second_norm(first_skip_output)
        ffn = self.swiglu(second_norm)

        third_skip_output = second_skip + ffn
        return third_skip_output, cache


class Llama2(nn.Module):

    def __init__(self, vocab_size, seq_len, d_model, num_heads, n_kv_heads, num_layers=32, is_causal=True, device=global_device):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.causal = is_causal
        self.layers = num_layers
        self.caches = [None] * num_layers
        self.is_causal = is_causal

        self.embeddings = Embeddings(vocab_size=vocab_size, d_model=d_model, device=device)
        self.norm = RMSNorm(d_model=d_model, device=device)
        self.decoder = nn.ModuleList([DecoderBlock(vocab_size, seq_len, d_model, num_heads, n_kv_heads, is_causal, device) for _ in range(num_layers)])
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False, device=device)
        with torch.no_grad():    
            self.output_projection.weight.normal_(mean=0, std=0.02)

    def reset_cache(self):
        self.caches = [None] * self.layers
    
    def forward(self, x):
        embedded_text = self.embeddings(x)
        """
        if self.training:
            for idx, block in enumerate(self.decoder):
                block_output, cache = torch.utils.checkpoint.checkpoint(block, embedded_text, use_reentrant=False)
                embedded_text = block_output
        """
        #else:
        for idx, block in enumerate(self.decoder):
            block_output, cache = block(embedded_text, self.caches[idx])
            self.caches[idx] = cache
            embedded_text = block_output
        
        normed = self.norm(embedded_text)

        logits = self.output_projection(normed)

        return logits


