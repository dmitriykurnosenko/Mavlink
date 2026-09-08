import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 1. Создаем матрицу из нулей размера [max_len, embed_dim]
        # max_len — это максимальная длина текста, которую мы закладываем (например, 5000 слов)
        pe = torch.zeros(max_len, embed_dim)
        
        # 2. Создаем вектор позиций [0, 1, 2, ..., max_len-1] и меняем его форму на [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. Вычисляем знаменатель формулы (шаг частоты) для синусоид.
        # Берем только четные индексы (поэтому шаг 2). Формула: 10000^(2i / embed_dim)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # 4. Заполняем матрицу:
        # Четные индексы (0, 2, 4...) заполняем синусами
        pe[:, 0::2] = torch.sin(position * div_term)
        # Нечетные индексы (1, 3, 5...) заполняем косинусами
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5. Добавляем размерность для батча, чтобы форма стала [1, max_len, embed_dim]
        pe = pe.unsqueeze(0)
        
        # register_buffer говорит PyTorch, что это состояние модели (константа),
        # которую нужно сохранять вместе с весами, но НЕ нужно обучать (для нее нет градиентов).
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x имеет форму: [Batch_size, Seq_len, Embed_dim]
        # Мы просто берем срез нашей матрицы позиций под длину текущего текста (Seq_len)
        # и ПРИБАВЛЯЕМ к вектору слов X.
        x = x + self.pe[:, :x.size(1)]
        return x

# --- Проверка работы кода ---
B, T, D = 1, 4, 6  # 1 предложение, 4 слова, размерность эмбеддинга 6
torch.manual_seed(42)

# Эмбеддинги слов (представим, что это выход слоя nn.Embedding)
word_embeddings = torch.randn(B, T, D)
print("Исходные эмбеддинги слов (до позиций):\n", word_embeddings)

# Создаем наш слой
pos_encoder = PositionalEncoding(embed_dim=D, max_len=10)
ready_embeddings = pos_encoder(word_embeddings)

print("\nЭмбеддинги ПОСЛЕ добавления Positional Encoding:\n", ready_embeddings)
