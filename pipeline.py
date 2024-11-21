from utils.normalización import normalice_and_save

def pipeline(output_folder):
    print("Iniciando normalización")
    normalice_and_save(output_folder)
    print("Normalización finalizada")
    return


if __name__ == "__main__":
    OUTPUT_PATH = "output/"

    pipeline(OUTPUT_PATH)
