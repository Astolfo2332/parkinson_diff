from utils.normalización import normalice_and_save
from utils.data_load import load_data
from utils.experiment_generator import run_experiments
import json

def pipeline(output_folder):
    with open("parameters.json", "r",encoding="utf-8") as file:
        parameters = json.load(file)
    
    print("Iniciando normalización")
    normalice_and_save(output_folder)
    print("Normalización finalizada")

    AUGMENTATION = True
    ALL_DATA = False
    WEIGHTED_LOSS = False

    if "no_augmentation" in parameters:
        AUGMENTATION = False
        WEIGHTED_LOSS = True

    if "no_all_data" in parameters:
        ALL_DATA = False
        WEIGHTED_LOSS = True

    print("Creando dataloaders")
    train_loader, test_loader = load_data(output_folder, parameters["batches"],
                                          augmentation=AUGMENTATION, all_data=ALL_DATA)
    print("Dataloaders creados")

    
    print("Iniciando entrenamiento")
    run_experiments(test_loader, train_loader, parameters, w_loss=WEIGHTED_LOSS)
    print("Entrenamiento finalizado")

    return


if __name__ == "__main__":
    OUTPUT_PATH = "output/"

    pipeline(OUTPUT_PATH)
