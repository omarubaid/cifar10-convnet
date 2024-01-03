from tensorflow import keras
from tensorflow.keras import datasets, layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

shuffle_index = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

validation_size = 10000
X_val, y_val = X_train[:validation_size], y_train[:validation_size]
X_train, y_train = X_train[validation_size:], y_train[validation_size:]

model = keras.Sequential([
    layers.BatchNormalization(input_shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64, verbose=1)
model.evaluate(X_test, y_test)

history_frame = pd.DataFrame(history.history)
history_frame[['loss', 'val_loss']].plot(title='Training and Validation Loss')
history_frame[['accuracy', 'val_accuracy']].plot(title='Training and Validation Accuracy')

plt.show()
