import numpy as np

class CargadorDatos:
    @staticmethod
    def cargar_datos():
        # Aquí cargarías tus datos reales
        # Por simplicidad, generamos datos aleatorios
        X = np.random.rand(1000, 48, 48, 1).astype(np.float32)
        y = np.random.randint(0, 7, 1000)
        return X, y
