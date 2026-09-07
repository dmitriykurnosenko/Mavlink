"""
llm_numpy_demo.py — упрощённая демонстрация Causal Self-Attention на чистом NumPy.

Зачем: PyTorch на Termux (Android) либо не ставится, либо падает на импорте.
Этот файл показывает МАТЕМАТИКУ внимания без обучения, без CUDA, без тяжёлых
зависимостей. Идеален, чтобы проверить, что Jupyter в Termux вообще живой,
и «пощупать» формулы руками.

Запуск в Jupyter:
    %run llm_numpy_demo.py

Запуск из терминала Termux:
    python llm_numpy_demo.py
"""
import numpy as np


def softmax(x, axis=-1):
    """Численно стабильный softmax: вычитаем max перед exp."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_self_attention(X, W_q, W_k, W_v, n_head):
    """
    Multi-Head Causal Self-Attention в стиле GPT.

    X  : (B, T, C)  — вход (B батчей, T токенов, C каналов)
    W_*: (C, C)     — обучаемые матрицы Q, K, V
    n_head          — число «голов» внимания

    Возвращает (B, T, C) — контекстно-смешанное представление.
    """
    B, T, C = X.shape
    assert C % n_head == 0, "C должно делиться на n_head"
    Dh = C // n_head

    # Линейные проекции в Q, K, V
    Q = (X @ W_q).reshape(B, T, n_head, Dh).transpose(0, 2, 1, 3)  # (B, H, T, Dh)
    K = (X @ W_k).reshape(B, T, n_head, Dh).transpose(0, 2, 1, 3)
    V = (X @ W_v).reshape(B, T, n_head, Dh).transpose(0, 2, 1, 3)

    # Масштабированное скалярное произведение
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(Dh)             # (B, H, T, T)

    # Каузальная маска: токен на позиции t видит только 0..t
    mask = np.triu(np.full((T, T), -np.inf), k=1)                  # верхний треугольник = -inf
    weights = softmax(scores + mask, axis=-1)                      # после softmax маска даёт 0

    # Взвешенная сумма значений
    out = weights @ V                                              # (B, H, T, Dh)
    return out.transpose(0, 2, 1, 3).reshape(B, T, C)              # (B, T, C)


def demo():
    # ---- 1. Случайный вход: 2 примера, 5 токенов, 8 признаков ----------
    np.random.seed(0)
    B, T, C, H = 2, 5, 8, 2
    X = np.random.randn(B, T, C).astype(np.float32)
    W_q, W_k, W_v = [np.random.randn(C, C).astype(np.float32) * 0.1 for _ in range(3)]

    # ---- 2. Прогон -------------------------------------------------------
    out = causal_self_attention(X, W_q, W_k, W_v, n_head=H)

    # ---- 3. Печать результатов -----------------------------------------
    print("=" * 60)
    print("Causal Self-Attention — NumPy демо")
    print("=" * 60)
    print(f"Вход X:        {X.shape}  (B={B}, T={T}, C={C}, n_head={H})")
    print(f"Выход out:     {out.shape}")
    print(f"Среднее out:   {out.mean():+.4f}")
    print(f"Станд. откл.:  {out.std():+.4f}")
    print()
    print("Выход первого примера, первые 3 токена:")
    np.set_printoptions(precision=3, suppress=True)
    print(out[0, :3])
    print()

    # ---- 4. Проверка каузальности --------------------------------------
    # Меняем входной токен в ДАЛЁКОМ будущем (t=4) и смотрим,
    # изменился ли выход на РАННЕЙ позиции (t=0).
    # Если маска работает — измениться не должен.
    X2 = X.copy()
    X2[0, 4, :] += 10.0  # большой «шум» в будущем токене
    out2 = causal_self_attention(X2, W_q, W_k, W_v, n_head=H)

    diff_t0 = np.abs(out2[0, 0] - out[0, 0]).max()
    diff_t3 = np.abs(out2[0, 3] - out[0, 3]).max()
    diff_t4 = np.abs(out2[0, 4] - out[0, 4]).max()

    print("Тест каузальной маски:")
    print(f"  |Δ на t=0| = {diff_t0:.6f}  (должен быть ~0, видит только прошлое)")
    print(f"  |Δ на t=3| = {diff_t3:.6f}  (должен быть ~0, видит прошлое+настоящее)")
    print(f"  |Δ на t=4| = {diff_t4:.6f}  (должен измениться, видит будущее)")
    print()
    if diff_t0 < 1e-5 and diff_t4 > 1e-3:
        print("✅ Маска работает корректно!")
    else:
        print("❌ Что-то пошло не так с маской.")

    # ---- 5. Визуализация весов внимания (если есть matplotlib) ----------
    try:
        import matplotlib.pyplot as plt
        # Пересчитаем веса для одного примера, одной головы
        Q = (X[0:1] @ W_q).reshape(1, T, H, C // H).transpose(0, 2, 1, 3)
        K = (X[0:1] @ W_k).reshape(1, T, H, C // H).transpose(0, 2, 1, 3)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(C // H)
        mask = np.triu(np.full((T, T), -np.inf), k=1)
        weights = softmax(scores + mask, axis=-1)[0]   # (H, T, T)

        fig, axes = plt.subplots(1, H, figsize=(4 * H, 3.5), sharey=True)
        for h in range(H):
            ax = axes[h] if H > 1 else axes
            im = ax.imshow(weights[h], cmap="viridis", vmin=0, vmax=weights[h].max())
            ax.set_title(f"Head {h+1}")
            ax.set_xlabel("Ключ (позиция)")
            ax.set_ylabel("Запрос (позиция)" if h == 0 else "")
            for i in range(T):
                for j in range(T):
                    if weights[h, i, j] > 1e-3:
                        ax.text(j, i, f"{weights[h, i, j]:.2f}",
                                ha="center", va="center", color="white", fontsize=8)
        fig.suptitle("Матрица внимания: верхний треугольник = 0 (каузальная маска)")
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.tight_layout()
        plt.show()
        print("График весов внимания отрисован.")
    except ImportError:
        print("matplotlib не установлен — пропускаю визуализацию.")
        print("Поставь его командой:  pip install matplotlib")


if __name__ == "__main__":
    demo()
