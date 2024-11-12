from reconocedor_emociones import ReconocedorEmociones

def main():
    reconocedor = ReconocedorEmociones()
    reconocedor.entrenar_modelo()
    reconocedor.iniciar_reconocimiento()  # Iniciar el reconocimiento de emociones

if __name__ == "__main__":
    main()
