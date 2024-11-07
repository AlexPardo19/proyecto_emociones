from cargador_datos import CargadorDatos
from modelo_emociones import ModeloEmociones
from procesador_imagen import ProcesadorImagen

class ReconocedorEmociones:
    def __init__(self):
        self.cargador = CargadorDatos()
        self.modelo = ModeloEmociones()
        self.procesador = ProcesadorImagen()
        self.emociones = ['Enojo', 'Disgusto', 'Miedo', 'Felicidad', 'Tristeza', 'Sorpresa', 'Neutral']

    def entrenar_modelo(self):
        X, y = self.cargador.cargar_datos()
        self.modelo.compilar()
        self.modelo.entrenar(X, y)

    def predecir_emocion(self, imagen_path):
        imagen = self.procesador.preprocesar(imagen_path)
        indice_emocion = self.modelo.predecir(imagen)
        return self.emociones[indice_emocion]
