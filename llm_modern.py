"""
llm_modern.py — мини-LLM «как у LLaMA».
Современные блоки: RoPE, RMSNorm, SwiGLU, KV-cache, Top-p sampling.
"""
import math, urllib.request, torch, torch.nn as nn, torch.nn.functional as F

# ---- Конфиг --------------------------------------------------------------
BLOCK_SIZE = 256
BATCH_SIZE = 64
N_EMBD     = 384
N_HEAD     = 6
N_LAYER    = 6
DROPOUT    = 0.2
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# ---- Данные ---------------------------------------------------------------
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = urllib.request.urlopen(url).read().decode("utf-8")
chars = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join([itos[i] for i in ids])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i+BLOCK_SIZE]     for i in ix])
    y = torch.stack([d[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ---- RoPE ----------------------------------------------------------------
def precompute_rope_cache(head_dim, max_seq_len, base=10000.0, device=DEVICE):
    """
    Кэш поворотных матриц (cos, sin) для каждой позиции и пары измерений.
    Вектор x размерности head_dim делится на пары: (x_0, x_1), (x_2, x_3), ...
    Каждая пара вращается на угол θ_i * pos, где θ_i = 1 / base^(2i / head_dim).
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)   # кэш на всю сессию

def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim/2)
    T = x.size(-2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)     # (1, 1, T, head_dim/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)   # пары (..., 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    # Стандартная формула поворота 2D-вектора:
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.stack([rot1, rot2], dim=-1).reshape_as(x)
    return out.to(x.dtype)

# ---- RMSNorm -------------------------------------------------------------
class RMSNorm(nn.Module):
    """y = x / rms(x) * weight, где rms(x) = sqrt(mean(x^2) + eps)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight

# ---- Causal Self-Attention с KV-cache и RoPE -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.qkv   = nn.Linear(n_embd, 3 * n_embd, bias=False)  # один матричный умножение на Q,K,V
        self.proj  = nn.Linear(n_embd, n_embd, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.block_size = block_size
        # Causal-mask: верхний треугольник = -inf (мы запрещаем видеть «будущее»)
        self.register_buffer("mask",
            torch.triu(torch.full((block_size, block_size), float("-inf")), diagonal=1))

    def forward(self, x, cos, sin, kv_cache=None, start_pos=0):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 × (B, n_head, T, head_dim)

        # --- KV-cache: на инференсе переиспользуем прошлые K, V -----------
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # Поворотное кодирование применяется только к Q и K
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Attention: (B, n_head, T_q, T_kv)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        T_kv = k.size(2)
        # Каузальная маска: разрешаем q[t] смотреть на k[0..t+start_pos]
        att = att.masked_fill(
            torch.triu(torch.full((T, T_kv), float("-inf"), device=x.device),
                       diagonal=start_pos + 1), 0)
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v                         # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), new_cache

# ---- SwiGLU Feed-Forward --------------------------------------------------
class SwiGLU(nn.Module):
    """
    SwiGLU: y = (silu(x @ W1) * (x @ W2)) @ W3
    silu(x) = x * sigmoid(x) — гладкая альтернатива ReLU.
    Скрытая размерность ~ 2.67 * n_embd (часто округляют до кратного 32 или 64).
    """
    def __init__(self, n_embd, mult=8/3):
        super().__init__()
        hidden = int(n_embd * mult)
        # делаем hidden кратным 32, чтобы тензорные ядра GPU работали эффективнее
        hidden = ((hidden + 31) // 32) * 32
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, n_embd, bias=False)
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

# ---- Трансформер-блок -----------------------------------------------------
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn  = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.norm2 = RMSNorm(n_embd)
        self.mlp   = SwiGLU(n_embd)
    def forward(self, x, cos, sin, kv_cache=None, start_pos=0):
        # Pre-Norm: x = x + attn(norm(x))
        h, new_cache = self.attn(self.norm1(x), cos, sin, kv_cache, start_pos)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, new_cache

# ---- Полная модель --------------------------------------------------------
class MiniLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.blocks  = nn.ModuleList(
            [Block(N_EMBD, N_HEAD, BLOCK_SIZE, DROPOUT) for _ in range(N_LAYER)])
        self.norm_f  = RMSNorm(N_EMBD)
        self.head    = nn.Linear(N_EMBD, vocab_size, bias=False)
        # tying: разделяем веса эмбеддингов и выходного слоя (как в LLaMA)
        self.head.weight = self.tok_emb.weight
        # Кэш RoPE пересчитываем максимум на block_size * 4 позиции
        cos, sin = precompute_rope_cache(N_EMBD // N_HEAD, BLOCK_SIZE * 4)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, idx, targets=None, kv_cache=None, start_pos=0):
        x = self.tok_emb(idx)
        caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, self.rope_cos, self.rope_sin,
                       None if kv_cache is None else kv_cache[i],
                       start_pos)
            caches.append(c)
        x = self.norm_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=1.0, top_p=0.9):
        # Жадный шаг для prompt, потом по одному токену с KV-cache
        for step in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            start = idx.size(1) - idx_cond.size(1)
            logits, _, caches = self(idx_cond, kv_cache=None if step == 0 else caches,
                                     start_pos=start)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # Top-p (nucleus) sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = sorted_probs.cumsum(dim=-1)
            mask = cum - sorted_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_tok = sorted_idx.gather(-1,
                torch.multinomial(sorted_probs, 1))
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

# ---- Запуск ---------------------------------------------------------------
if __name__ == "__main__":
    model = MiniLLM(vocab_size=len(chars)).to(DEVICE)
    print(f"Параметров: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print(f"Устройство: {DEVICE}")
    print(decode(model.generate(torch.zeros((1,1), dtype=torch.long, device=DEVICE),
                                 max_new_tokens=200, temperature=0.8, top_p=0.9)[0].tolist()))
