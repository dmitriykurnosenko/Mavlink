import torch
import torch.nn.functional as F

# Предложение из 3 слов: "ИИ(1) меняет(2) мир(3)"
B, T, D = 1, 3, 4  
d_k = D

torch.manual_seed(42)
Q = torch.randn(B, T, D)
K = torch.randn(B, T, D)
V = torch.randn(B, T, D)

# 1. Считаем обычные оценки сходства (Scores)
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
print("Исходные оценки (до маски):\n", scores)

# 2. Создаем верхнетреугольную маску (Causal Mask)
# torch.triu(..., diagonal=1) оставляет единицы только ВЫШЕ главной диагонали
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
print("\nМатрица маски (True — это будущее, которое надо скрыть):\n", mask)

# 3. Применяем маску: заменяем все True на -inf (минус бесконечность)
# .masked_fill_ принимает маску типа bool и заполняет нужные места
masked_scores = scores.masked_fill(mask, float('-inf'))
print("\nОценки ПОСЛЕ применения маски (-inf):\n", masked_scores)

# 4. Применяем Softmax
# Обратите внимание: там, где было -inf, теперь станут чистые 0.0!
attention_weights = F.softmax(masked_scores, dim=-1)
print("\nВеса внимания (Softmax). Посмотрите на нули в правом верхнем углу:\n", attention_weights)

# 5. Считаем финальный выход
output = torch.matmul(attention_weights, V)
