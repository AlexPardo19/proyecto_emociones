import cv2
import numpy as np

class ProcesadorImagen:
    @staticmethod
    def preprocesar(imagen_path):
        imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen en la ruta: {imagen_path}")
        imagen = cv2.resize(imagen, (48, 48))
        imagen = np.expand_dims(imagen, axis=[0, -1])
        return imagen.astype(np.float32)  # Aseguramos que sea float32
