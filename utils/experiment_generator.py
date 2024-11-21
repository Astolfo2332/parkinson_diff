import json
import torch
from torch.utils.data import DataLoader
from torch import nn
from utils.engine import set_seed
from utils.model_generator import TinyVGG
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

    for epochs in num_epochs:
        for optimizer_name in optimizers:
            experiment_number += 1
            model_name = "TinyVGG"
            print(f"[INFO] Experiment number: {experiment_number}")
            print(f"[INFO] model: {model_name}")
            print(f"[INFO] Optimizer: {optimizer_name}")
            print(f"[INFO] Epochs: {epochs}")
            model = TinyVGG(input_shape=1, hidden_units=64, output_shape=2).to(device)
            writer = create_write(name=optimizer_name, 
            model=model_name, extra=str(epochs))
            #log_dir = os.path.join("log", timestamp + optimizer + name + str(epochs))
            optimizer = select_optimizer(model, optimizer_name)
            results = train(model, test_dataloader, train_dataloader, loss_fn, optimizer, device,
            epochs, writer)
            print("-" * 50 + "\n")
    torch.cuda.empty_cache()
    return