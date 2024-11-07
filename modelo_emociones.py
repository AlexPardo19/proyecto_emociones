import numpy as np
from tensorflow.keras import layers, models

class ModeloEmociones:
    def __init__(self):
        self.modelo = self._crear_modelo()

    def _crear_modelo(self):
        modelo = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(7, activation='softmax')
        ])
        return modelo

    def compilar(self):
        self.modelo.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    def entrenar(self, X, y, epochs=10, validation_split=0.2):
        return self.modelo.fit(X, y, epochs=epochs, validation_split=validation_split)

    def predecir(self, imagen):
        prediccion = self.modelo.predict(imagen)
        return np.argmax(prediccion)
