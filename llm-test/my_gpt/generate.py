import torch
from mingpt.model import GPT

# 1. Загружаем сохраненный файл (чекпоинт)
print("Загрузка модели из файла...")
checkpoint = torch.load('checkpoint.pt', map_location='cpu', weights_only=False)

# 2. Восстанавливаем словари
stoi = checkpoint['stoi']
itos = checkpoint['itos']

# 3. Создаем точно такую же структуру модели по сохраненному конфигу
model_config = checkpoint['model_config']
model = GPT(model_config)

# 4. Загружаем в модель обученные веса
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Переводим модель в режим оценки/генерации
print("Модель успешно загружена и готова к работе!")

# 5. Функция для генерации текста по вашей подсказке
def generate_text(prompt, max_tokens=150, temperature=0.8):
    # Проверяем, все ли символы из подсказки есть в нашем словаре
    # Если символа нет, заменяем его на пробел, чтобы скрипт не упал
    clean_prompt = "".join([s if s in stoi else " " for s in prompt])
    
    # Переводим текст в тензор чисел
    x = torch.tensor([stoi[s] for s in clean_prompt], dtype=torch.long).unsqueeze(0)
    
    # Генерируем новые токены (символы)
    with torch.no_grad(): # Отключаем подсчет градиентов для экономии памяти
        y = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
    
    # Переводим числа обратно в текст
    completion = ''.join([itos[int(i)] for i in y[0]])
    return completion

# --- Запуск генерации ---
print("\n" + "="*40)
prompt_text = "ROMEO:" # Ваша начальная фраза
print(f"Подсказка: {prompt_text}")
print("Результат генерации:")

result = generate_text(prompt_text, max_tokens=200, temperature=0.9)
print(result)
print("="*40)
