import numpy as np
import tensorflow as tf

# ==========================================
# 1. СОЗДАНИЕ И ОБУЧЕНИЕ ИГРУШЕЧНОЙ МОДЕЛИ
# ==========================================
# Представим, что мы обучаем простую модель для TinyML
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Компилируем и эмулируем обучение на случайных данных
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
x_train = np.random.randn(100, 10).astype(np.float32)
y_train = np.random.randint(0, 2, size=(100,))
model.fit(x_train, y_train, epochs=1, verbose=0)

print("1. Базовая модель успешно создана и обучена.")

# ==========================================
# 2. ПОДГОТОВКА ДАННЫХ ДЛЯ КВАНТОВАНИЯ
# ==========================================
# Генератор репрезентативных данных (из нашей обучающей выборки)
# Нужно передать хотя бы 100-500 примеров, чтобы калибровка была точной
def representative_data_gen():
    for i in range(100):
        # Добавляем размерность батча (1, 10) и приводим к float32
        data = np.expand_dims(x_train[i], axis=0)
        yield [data]

# ==========================================
# 3. НАСТРОЙКА И ЗАПУСК КВАНТОВАНИЯ
# ==========================================
# Создаем конвертер из нашей Keras модели
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Включаем режим оптимизации по размеру
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Передаем наш генератор данных для калибровки активаций
converter.representative_dataset = representative_data_gen

# Жестко требуем, чтобы и входы, и выходы, и внутренние операции были строго INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTIN_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Конвертируем модель
tflite_quant_model = converter.convert()
print("2. Модель успешно квантована в INT8.")

# ==========================================
# 4. СОХРАНЕНИЕ МОДЕЛИ
# ==========================================
with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_quant_model)
print("3. Файл 'model_quantized.tflite' сохранен.")
