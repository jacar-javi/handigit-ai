# Load Libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical


# 1. Load and prepare MNIST Dataset:

# Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train / 255
x_test = x_test / 255

# Convert Labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)




# 2. Create Artificial Neuronal Network:

# Create Sequential Model
model = Sequential()

# Add Layers to Model
model.add(Flatten(input_shape=(28, 28)))  # Flattening Layer
model.add(Dense(128, activation='relu'))  # Hidden Layer
model.add(Dense(10, activation='softmax'))  # Output Layer



# 3. Model Compilation:

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# 4. Model Training:

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)



# 5. Model Evaluation:

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")



# 6. Model performance durihng training:

# Chart Creation
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.show()


# Save trained model
model.save('trained_model.h5')