import torch
import torch.nn as nn
from torch.utils.data import Dataset
from mingpt.model import GPT
from mingpt.trainer import Trainer

# 1. Локальная генерация текста вместо скачивания
print("Создание локальных данных...")
base_text = """
ROMEO: O God, O God! shall we ever meet again?
JULIET: I doubt it not; and all these woes shall serve
For sweet discourses in our time to come.
ROMEO: Alack, there lies more peril in thine eye
Than twenty of their swords: look thou but sweet,
And I am proof against their enmity.
"""
text = base_text * 1500 # Размножаем фразу, чтобы набралось достаточно символов

# 2. Создаем простой символьный датасет
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print(f'Данные содержат {data_size} символов, уникальных: {vocab_size}')
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

block_size = 64 # Длина контекста (сколько символов модель видит за раз)
train_dataset = CharDataset(text, block_size)

# 3. Настройка СУПЕР-МАЛЕНЬКОЙ модели (чтобы телефон справился)
model_config = GPT.get_default_config()
model_config.model_type = None # Отключаем пресеты GPT-2
model_config.vocab_size = train_dataset.vocab_size
model_config.block_size = block_size
# Кастомные ультра-легкие параметры:
model_config.n_layer = 2     # Всего 2 слоя трансформера (вместо 12)
model_config.n_head = 2      # 2 головы внимания
model_config.n_embd = 64     # Размерность эмбеддингов 64

model = GPT(model_config)

# 4. Настройка параметров обучения
trainer_config = Trainer.get_default_config()
trainer_config.learning_rate = 5e-4
trainer_config.max_iters = 500   # 500 шагов обучения (займет пару минут)
trainer_config.batch_size = 16   # Маленький размер батча для экономии памяти
trainer_config.num_workers = 0   # Важно для Termux, чтобы не ломались потоки

trainer = Trainer(trainer_config, model, train_dataset)

# Функция для генерации текста в процессе обучения
def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"Шаг {trainer.iter_num}: loss = {trainer.loss.item():.4f}")
        # Тестовая генерация
        context = "O God, O God!"
        x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long).unsqueeze(0)
        y = model.generate(x, max_new_tokens=100, temperature=1.0, do_sample=True)[0]
        completion = ''.join([train_dataset.itos[int(i)] for i in y])
        print('-'*30 + f'\nГенерация:\n{completion}\n' + '-'*30)

trainer.set_callback('on_batch_end', batch_end_callback)

# 5. Запуск обучения!
print("Начало обучения на CPU...")
trainer.run()

# 6. Сохранение результатов после завершения trainer.run()
print("Сохранение модели...")

# Создаем словарь (checkpoint), куда упакуем всё необходимое
checkpoint = {
    'model_state_dict': model.state_dict(), # Сами веса обученной нейросети
    'model_config': model_config,           # Конфигурация слоев модели
    'stoi': train_dataset.stoi,             # Словарь: символ -> число
    'itos': train_dataset.itos              # Словарь: число -> символ
}

# Записываем всё в один файл
torch.save(checkpoint, 'checkpoint.pt')
print("Модель успешно сохранена в файл checkpoint.pt!")
