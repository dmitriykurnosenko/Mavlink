import torch
import torch.nn.functional as F

# Зададим случайные тензоры для примера (Batch_size=1, Sequence_length=3, Dimension=4)
# Например, предложение: "ИИ меняет мир" (3 слова, у каждого вектор из 4 чисел)
torch.manual_seed(42)  # Фиксируем генератор для воспроизводимости

B, T, D = 1, 3, 4  # Батч, Длина последовательности (слов), Размерность вектора
d_k = D            # Размерность ключей

# Создаем случайные матрицы Q, K, V
Q = torch.randn(B, T, D)
K = torch.randn(B, T, D)
V = torch.randn(B, T, D)

print("Матрица Query (Запросы):\n", Q[0])

# Шаг 1 & 2: Находим сходство (Scores) и масштабируем
# Используем transpose(-2, -1) для корректного матричного умножения последних осей K
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
print("\nМатрица оценок внимания (до Softmax):\n", scores[0])

# Шаг 3: Нормализация через Softmax по последней оси (строкам)
attention_weights = F.softmax(scores, dim=-1)
print("\nВеса внимания (Softmax) - строки суммируются в 1:\n", attention_weights[0])

# Шаг 4: Умножаем веса на значения (Value)
output = torch.matmul(attention_weights, V)
print("\nФинальный выход слоя внимания:\n", output[0])
