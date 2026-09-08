import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim должен делиться на num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Размерность одной головы (например, 512 / 8 = 64)
        
        # Линейные слои для получения Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Финальный линейный слой после склейки голов
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x имеет форму: [Batch_size, Seq_len, Embed_dim]
        B, T, D = x.shape
        
        # Шаг 1: Проекция в Q, K, V
        Q = self.q_linear(x)  # [B, T, D]
        K = self.k_linear(x)  # [B, T, D]
        V = self.v_linear(x)  # [B, T, D]
        
        # Трюк с размерностями: разбиваем D на (num_heads, head_dim)
        # И меняем местами оси, чтобы головы шли СРАЗУ после Батча: [B, num_heads, T, head_dim]
        # Это позволяет PyTorch обрабатывать все головы параллельно одной операцией!
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Шаг 2: Расчет Scaled Dot-Product Attention для всех голов сразу
        # Матричное умножение последних двух осей (T, head_dim) на (head_dim, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, T, T]
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Умножаем веса на V
        out = torch.matmul(attention_weights, V)  # [B, num_heads, T, head_dim]
        
        # Шаг 3: Склеивание голов обратно (Concatenation)
        # Возвращаем оси на место: [B, T, num_heads, head_dim]
        out = out.transpose(1, 2).contiguous()
        # Схлопываем последние две оси обратно в D (num_heads * head_dim)
        out = out.view(B, T, self.embed_dim)  # [B, T, D]
        
        # Шаг 4: Финальная линейная проекция
        return self.out_linear(out)

# --- Пример проверки работы слоя ---
B, T, D = 2, 5, 512  # 2 предложения, по 5 слов, размерность эмбеддинга 512
num_heads = 8        # 8 голов внимания (каждая размерностью 64)

X = torch.randn(B, T, D)
mha = MultiHeadAttention(embed_dim=D, num_heads=num_heads)
output = mha(X)

print("Входной размер:", X.shape)
print("Выходной размер слоя MHA:", output.shape)  # Размерность должна сохраниться!
