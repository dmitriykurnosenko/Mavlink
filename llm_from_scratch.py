"""
Минимальный LLM-блок: посимвольный токенизатор + Causal Self-Attention.
Запусти в Jupyter Notebook (вставь в ячейку, Shift+Enter).
Требует: pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

# ============================================================
# 1. ДАННЫЕ: Крошечный Шекспир (для первого эксперимента хватит)
# ============================================================
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
print(f"Длина текста: {len(text)} символов")
print(text[:200])

# ============================================================
# 2. ПОСИМВОЛЬНЫЙ ТОКЕНИЗАТОР (char-level)
# ============================================================
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # char -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int -> char

encode = lambda s: [stoi[c] for c in s]       # строка -> список int
decode = lambda l: "".join([itos[i] for i in l])  # список int -> строка

print(f"\nРазмер словаря: {vocab_size} уникальных символов")
print(f"encode('hello') = {encode('hello')}")
print(f"decode([46, 43, 50, 50, 53]) = {decode([46, 43, 50, 50, 53])}")

# Превращаем весь текст в тензор
data = torch.tensor(encode(text), dtype=torch.long)
print(f"\ndata.shape: {data.shape}, data.dtype: {data.dtype}")

# Делим на train/val
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ============================================================
# 3. БАТЧИ: выбираем случайные куски текста для обучения
# ============================================================
batch_size = 32      # сколько независимых примеров за раз
block_size = 64      # длина контекста (сколько символов модель "видит")

def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i : i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1 : i + block_size + 1] for i in ix])
    return x, y

# ============================================================
# 4. CAUSAL SELF-ATTENTION
#    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + M) V
#    M = маска: -inf в верхнем треугольнике (будущее скрыто)
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Одна большая матрица для Q, K, V сразу (эффективнее)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Каузальная маска: треугольная, верх = -inf
        # Зарегистрируем как буфер, чтобы не обучалась, но ездила с .to(device)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape  # batch, time (длина контекста), channels (embed_dim)
        qkv = self.qkv(x)                      # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)         # каждый (B, T, C)

        # Разбиваем на головы: (B, T, C) -> (B, T, nh, hd) -> (B, nh, T, hd)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Скалярное произведение: QK^T, делим на sqrt(d_k) для стабильности
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B, nh, T, T)

        # Применяем каузальную маску: -inf там, где 0
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)            # вероятности по прошлому
        att = self.dropout(att)

        # Взвешиваем значения V
        y = att @ v                             # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # склеиваем головы

        return self.out_proj(y)

# ============================================================
# 5. ОДИН БЛОК ТРАНСФОРМЕРА (Attention + MLP)
# ============================================================
class FeedForward(nn.Module):
    """Простой MLP: расширяем в 4 раза, активация, сужаем обратно."""
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dropout)

    def forward(self, x):
        # Pre-LN: нормализация -> подсеть -> residual
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# ============================================================
# 6. МИНИ-GPT: эмбеддинги токенов + позиций + блоки + голова
# ============================================================
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=96, num_heads=6,
                 num_layers=4, block_size=64, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)                              # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device)) # (T, C)
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                                  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Авторегрессионная генерация: токен за токеном."""
        for _ in range(max_new_tokens):
            # Берём только последние block_size токенов (контекстное окно)
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

# ============================================================
# 7. СОЗДАЁМ МОДЕЛЬ И ПРОВЕРЯЕМ
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nУстройство: {device}")

model = MiniGPT(
    vocab_size=vocab_size,
    embed_dim=128,    # размерность эмбеддинга
    num_heads=4,      # число голов внимания
    num_layers=4,     # число блоков трансформера
    block_size=block_size,
    dropout=0.1,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Параметров в модели: {n_params:,}")

# Быстрый тест: один проход вперёд + генерация ДО обучения
x, y = get_batch("train")
x, y = x.to(device), y.to(device)
logits, loss = model(x, y)
print(f"До обучения loss = {loss.item():.4f} (теоретически -ln(1/vocab_size) = {-torch.log(torch.tensor(1.0/vocab_size)):.4f})")

prompt = torch.tensor([encode("ROMEO:")], dtype=torch.long, device=device)
print("\nГенерация ДО обучения (случайный шум):")
print(decode(model.generate(prompt, max_new_tokens=150, top_k=50)[0].tolist()))
