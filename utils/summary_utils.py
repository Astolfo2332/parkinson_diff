from typing import Dict, List, Tuple
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from datetime import datetime
from utils.engine import train_func, test_func

def train_func_profiler(model: nn.Module, data: DataLoader, loss_fn:nn.Module,
               optimizer: torch.optim.Optimizer, device: torch.device
               , log_dir: str) -> Tuple[float, float]:
    """
    Trains a model for a single epoch

    Args
    ----
        model: A Pytorch model
        data: a DataLoader object with the train data
        loss_fn: loss function to minimized
        optimizer: A Pytorch optimizer
        device: A target device to perform the operations ("cuda" or "cpu")

    Returns
    ------
        A tuple with the loss and accuracy of the training epoch like
        (train_loss, train_acc)
    """
    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    model.train()
    train_loss, train_acc = 0, 0
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=1,
            repeat=1),
        #on_trace_ready=trace_handler,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        with_stack=True
    ) as profiler:
        for _ , (x, y) in enumerate(data):
            x, y = x.to(device), y.to(device)
            y_logit = model(x)
            loss = loss_fn(y_logit, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            profiler.step()
            y_pred = torch.softmax(y_logit, 1).argmax(1)
            train_acc += (y_pred == y).sum().item() / len (y_pred)
            #We can use another function if we want
    train_loss = train_loss / len(data)
    train_acc = train_acc / len(data)
    return train_loss, train_acc

def train(model: nn.Module, test_data: DataLoader, train_data: DataLoader, loss_fn:nn.Module,
        optimizer: torch.optim.Optimizer, device: torch.device, epochs: int
        , writer: SummaryWriter) -> Dict[str, List]:
    """
    Trains and test a Pytorch model

    Args:
    -----
        model: A Pytorch model
        train_data: a DataLoader object with the train data
        test_data: a DataLoader object with the train data
        loss_fn: loss function to minimized
        optimizer: A Pytorch optimizer
        device: A target device to perform the operations ("cuda" or "cpu")
        epochs: A integre with the number of epochs that the model will be train
    Returns:
    --------
        A dictionary with the train and test loss and accuracy for every epoch
        in the form of 
        {"train_loss": [...],
        "train_acc": [...],
        "test_loss": [...],
        "test_acc": [...]}
    """

    best_test_acc = 0
    best_model = None

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_func(
            model,train_data,loss_fn,optimizer, device)
        test_loss, test_acc = test_func(
            model, test_data, loss_fn, device)
        print(
            f"Epoch {epoch+1} |"
            f"train_loss :{train_loss: .4f} |"
            f"train_acc :{train_acc: .4f} |"
            f"test_loss :{test_loss: .4f} |"
            f"test_acc :{test_acc: .4f} "
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)



        ##Add of the writer
        if writer:
            writer.add_scalars(main_tag="Loss", 
            tag_scalar_dict={"train_loss": train_loss, "tests_loss": test_loss},
            global_step=epoch
            )

            writer.add_scalars(main_tag="Accuracy", 
            tag_scalar_dict={"train_acc": train_acc, "tests_acc": test_acc},
            global_step=epoch
            )

            writer.close()
        # Para guardar el mejor modelo
        if test_acc > best_test_acc:
            best_test_acc = train_acc
            best_model = model.state_dict()

    return results, best_model, best_test_acc

def create_write(name: str, model: str, extra: str=None) -> SummaryWriter():

    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        log_dir = os.path.join("runs", timestamp, name, model, extra)
    else:
        log_dir = os.path.join("runs", timestamp, name, model)
    print(f"[INFO] create summary writer, saving to: {log_dir}")
    return SummaryWriter(log_dir=log_dir)

def select_optimizer(model: nn.Module, optimizer: str):
    if optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=0.001)
    if optimizer == "SGD": 
        return torch.optim.SGD(model.parameters(), lr=0.001)
    if optimizer == "Adamw":
        return torch.optim.AdamW(model.parameters(), lr=0.001)
