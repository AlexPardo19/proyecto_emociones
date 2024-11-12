import cv2  # Importar OpenCV para acceder a la cámara
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

    def predecir_emocion(self, imagen):
        indice_emocion = self.modelo.predecir(imagen)
        return self.emociones[indice_emocion]

    def iniciar_reconocimiento(self):
        cap = cv2.VideoCapture(0)  # Abrir la cámara
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            imagen = self.procesador.preprocesar(frame)  # Procesar la imagen del frame
            emocion = self.predecir_emocion(imagen)
            cv2.putText(frame, emocion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Mostrar la emoción
            cv2.imshow('Reconocimiento de Emociones', frame)  # Mostrar el frame

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Salir con 'q'
                break
        cap.release()
        cv2.destroyAllWindows()
