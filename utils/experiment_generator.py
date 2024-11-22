import os
import json
import torch
from torch.utils.data import DataLoader
from torch import nn
from utils.engine import set_seed
from utils.model_generator import TinyVGG
from utils.model_generator import get_model
from utils.summary_utils import select_optimizer, create_write, train



def run_experiments(test_dataloader: DataLoader, train_dataloader: DataLoader):

    torch.cuda.empty_cache()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed()
    
    with open("parameters.json", "r",encoding="utf-8") as file:
        parameters = json.load(file)

    num_epochs = parameters["epochs"]
    optimizers = parameters["optimizers"]
    experiment_number = 0
    loss_fn = nn.CrossEntropyLoss().to(device)


    for model_name in parameters["models"]:
        best_test_acc = 0
        best_checkpoint = None
        for epochs in num_epochs:
            for optimizer_name in optimizers:
                experiment_number += 1
                print(f"[INFO] Experiment number: {experiment_number}")
                print(f"[INFO] model: {model_name}")
                print(f"[INFO] Optimizer: {optimizer_name}")
                print(f"[INFO] Epochs: {epochs}")
                model = get_model(model_name).to(device)
                writer = create_write(name=optimizer_name,
                model=model_name, extra=str(epochs))
                #log_dir = os.path.join("log", timestamp + optimizer + name + str(epochs))
                optimizer = select_optimizer(model, optimizer_name)
                results, best_model, test_acc = train(model, test_dataloader, train_dataloader, loss_fn, optimizer, device,
                epochs, writer)
                print("-" * 50 + "\n")

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_checkpoint = best_model
                    best_optimizer = optimizer_name
                    best_epochs = epochs
                    best_model_name = model_name
        
        if best_checkpoint:
            path = f"./models/{best_model_name}/{best_optimizer}/{best_epochs}"
            os.makedirs(path, exist_ok=True)
            torch.save(best_checkpoint, path + f"/best_model_{best_model_name}.pth")
            

    torch.cuda.empty_cache()
    return