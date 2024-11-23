from utils.normalización import normalice_and_save
from utils.data_load import load_data
from utils.experiment_generator import run_experiments

def pipeline(output_folder):
    print("Iniciando normalización")
    normalice_and_save(output_folder)
    print("Normalización finalizada")

    print("Creando dataloaders")
    train_loader, test_loader = load_data(output_folder, 8)
    print("Dataloaders creados")

    print("Iniciando entrenamiento")
    run_experiments(test_loader, train_loader)
    print("Entrenamiento finalizado")

    return


if __name__ == "__main__":
    OUTPUT_PATH = "output/"

    pipeline(OUTPUT_PATH)
