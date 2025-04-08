import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# **Подготовьте данные**: Для примера можно использовать набор данных MNIST, который уже включен в Keras:

from tensorflow.keras.datasets import mnist

# Загрузите данные
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализуйте данные
x_train, x_test = x_train / 255.0, x_test / 255.0

# **Создайте модель**:

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # 784 = 28*28, размер изображения MNIST
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 классов для цифр от 0 до 9
])

# **Скомпилируйте модель**:

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# **Обучите модель**:

        # Преобразуйте изображения в векторы
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

model.fit(x_train_flat, y_train, epochs=5)

model.save('mnist_model.h5')

#  **Оцените модель**:

test_loss, test_accuracy = model.evaluate(x_test_flat, y_test)
print(f"Test accuracy: {test_accuracy}")