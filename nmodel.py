import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from main import x_test_flat, x_test, y_test
loaded_model = tf.keras.models.load_model('mnist_model.keras')


# Предскажите классы для тестовой выборки
predictions = loaded_model.predict(x_test_flat)
predicted_class = np.argmax(predictions[0])
print(f"Predicted class: {predicted_class}")

# Отображаем первое изображение из тестового набора и его предсказанный класс
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted: {predicted_class}, Actual: {y_test[0]}")
plt.show()
