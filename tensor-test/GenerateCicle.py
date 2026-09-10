import torch
import torch.nn.functional as F

# Наш игрушечный словарь (Vocabulary)
vocab = {0: "<PAD>", 1: "ИИ", 2: "меняет", 3: "мир", 4: "к", 5: "лучшему", 6: "<EOS>"}
inv_vocab = {v: k for k, v in vocab.items()}

# Стартовый промт
input_text = ["ИИ", "меняет"]
input_tokens = [inv_vocab[word] for word in input_text]

print("Стартовый промт:", input_text)
print("Стартовые токены:", input_tokens)

# Настройки генерации
max_generated_tokens = 4
eos_token_id = inv_vocab["<EOS>"]  # Токен конца текста (End Of Sequence)

# Эмуляция финального слоя модели, который выдает логиты для 7 слов словаря
def mock_transformer_model(tokens):
    # В реальности здесь работает вся сеть Transformer.
    # Для демонстрации мы вернем фиксированные логиты, имитирующие знание контекста:
    last_token = tokens[-1]
    logits = torch.zeros(len(vocab))
    
    if last_token == inv_vocab["меняет"]:
        logits[inv_vocab["мир"]] = 10.0      # Делаем слово "мир" самым вероятным
    elif last_token == inv_vocab["мир"]:
        logits[inv_vocab["к"]] = 8.0         # Затем слово "к"
    elif last_token == inv_vocab["к"]:
        logits[inv_vocab["лучшему"]] = 12.0  # Затем "лучшему"
    else:
        logits[eos_token_id] = 5.0           # В конце ставим маркер завершения
    return logits

# --- ФИНАЛЬНЫЙ ЦИКЛ ГЕНЕРАЦИИ ---
for step in range(max_generated_tokens):
    # 1. Берем текущие токены
    current_tokens = torch.tensor(input_tokens)
    
    # 2. Передаем в модель и получаем логиты для СЛЕДУЮЩЕГО слова
    logits = mock_transformer_model(current_tokens)
    
    # 3. Превращаем логиты в вероятности (Softmax)
    probabilities = F.softmax(logits, dim=-1)
    
    # 4. Выбираем самое вероятное слово (стратегия Greedy Search)
    next_token_id = torch.argmax(probabilities).item()
    
    # 5. Добавляем токен в наш список (вход для следующего шага)
    input_tokens.append(next_token_id)
    
    # Печатаем слово на экран в реальном времени
    generated_word = vocab[next_token_id]
    print(f"Шаг {step+1}: сгенерировано слово -> '{generated_word}'")
    
    # Если модель сгенерировала токен конца текста, останавливаем цикл
    if next_token_id == eos_token_id:
        print("Получен токен <EOS>. Генерация завершена.")
        break

# Итоговый результат
final_text = [vocab[t] for t in input_tokens if vocab[t] not in ["<PAD>", "<EOS>"]]
print("\nИтоговый результат:", " ".join(final_text))
