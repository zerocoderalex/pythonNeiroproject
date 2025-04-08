import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # Для загрузки и обработки изображений

# Загрузите свою модель
loaded_model = tf.keras.models.load_model('mnist_model.keras')

# Функция для подготовки изображения
def prepare_image(image_path):
    # Откройте изображение и преобразуйте его в оттенки серого
    img = Image.open(image_path).convert('L')
    # Измените размер изображения до 28x28 пикселей
    img = img.resize((28, 28))
    # Преобразуйте изображение в массив numpy
    img_array = np.array(img)
    # Нормализуйте значения пикселей (если это необходимо)
    img_array = img_array / 255.0
    # Преобразуйте в плоский массив, если модель ожидает плоские данные
    img_flat = img_array.flatten()
    # Добавьте измерение для партии
    img_flat = img_flat.reshape(1, -1)
    return img_flat

# Укажите путь к своему изображению
image_path = 'images/image7.png'
prepared_image = prepare_image(image_path)

# Предскажите класс для вашего изображения
predictions = loaded_model.predict(prepared_image)
predicted_class = np.argmax(predictions[0])
print(f"Predicted class: {predicted_class}")

# Отображаем изображение и его предсказанный класс
plt.imshow(np.array(Image.open(image_path)), cmap='gray')
plt.title(f"Predicted: {predicted_class}")
plt.show()