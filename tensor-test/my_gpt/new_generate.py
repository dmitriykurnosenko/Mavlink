import torch
from mingpt.model import GPT

# 1. Загружаем сохраненный файл (чекпоинт)
print("Загрузка модели из файла...")
try:
    checkpoint = torch.load('checkpoint.pt', map_location='cpu', weights_only=False)
except FileNotFoundError:
    print("\nОшибка: Файл checkpoint.pt не найден! Сначала запустите обучение через 'python train.py'.")
    exit(1)

# 2. Восстанавливаем словари и конфигурацию
stoi = checkpoint['stoi']
itos = checkpoint['itos']
model_config = checkpoint['model_config']

# 3. Инициализируем модель и загружаем веса
model = GPT(model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Модель успешно загружена и готова к чату!")

# 4. Функция генерации текста
def generate_text(prompt, max_tokens=150, temperature=0.8):
    if not prompt:
        return ""
    # Фильтруем символы, которых нет в словаре модели (чтобы избежать KeyError)
    clean_prompt = "".join([s if s in stoi else " " for s in prompt])
    
    x = torch.tensor([stoi[s] for s in clean_prompt], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
    
    return ''.join([itos[int(i)] for i in y])

# --- Интерактивный цикл в консоли ---
print("\n" + "="*50)
print("  ИНТЕРАКТИВНЫЙ РЕЖИМ МИНИ-GPT")
print("  Введите текст-подсказку для модели.")
print("  Для выхода из программы введите слово: exit")
print("="*50)

while True:
    try:
        user_input = input("\nВы: ").strip()
        
        if user_input.lower() == 'exit':
            print("Выход из программы. Пока!")
            break
            
        if not user_input:
            continue
            
        print("\nGPT генерирует ответ...")
        # Передаем текст пользователя в модель
        result = generate_text(user_input, max_tokens=120, temperature=0.8)
        
        print("-" * 40)
        print(result)
        print("-" * 40)
        
    except KeyboardInterrupt:
        # Позволяет выйти из цикла по нажатию Ctrl+C
        print("\nПрограмма принудительно остановлена. Пока!")
        break
