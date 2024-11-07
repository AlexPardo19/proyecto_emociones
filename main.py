from reconocedor_emociones import ReconocedorEmociones

def main():
    reconocedor = ReconocedorEmociones()
    reconocedor.entrenar_modelo()
    """""
     Ejemplo de predicción
     Asegúrar de proporcionar una ruta válida a una imagen
     emocion = reconocedor.predecir_emocion('ruta/a/tu/imagen.jpg')
     print(f"La emoción predicha es: {emocion}")
    """

if __name__ == "__main__":
    main()
