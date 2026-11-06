# %%
#%matplotlib inline

import os
import torch
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import copy
import re
import math

from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2
from functools import partial
from matplotlib.ticker import PercentFormatter

torch.manual_seed(42)

# Switch to True if you want to retrain all models and keep False if it can use the earlier found results
FORCE_RETRAIN = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# %% [markdown]
# # Task 1

# %% [markdown]
# ## Data Loading

# %%
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

# Download training data from open datasets.
training_fashion = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

# Download test data from open datasets.
test_fashion = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

# Split 10% of the training data for validation
generator = torch.Generator().manual_seed(42)
train_fashion, val_fashion = random_split(training_fashion, [54000, 6000], generator=generator)


# %%
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(train_fashion, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_fashion, batch_size=batch_size)
val_dataloader = DataLoader(val_fashion, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# %% [markdown]
# ## Defining MLP and CNN classes

# %%
class MLP(nn.Module):
    def __init__(self, activation=nn.ReLU, width=512, dropout=0, depth=2, init=None, in_features= 28 * 28):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        for _ in range(depth):
            layers += [
                nn.Linear(in_features, width),
                activation(),
                nn.Dropout(dropout),
            ]
            in_features = width
        layers.append(nn.Linear(in_features, 10))

        self.layers = nn.Sequential(*layers)

        # Initialization
        if init is not None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# %%
class CNN(nn.Module):
    def __init__(self, in_channels=1, image_size=28, activation=nn.ReLU, dropout=0.0, init = None, channels=(6, 16), kernel_size=5, classification_size= 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size)

        # Fully connected layers
        size = ((image_size - kernel_size + 1) // 2 - kernel_size + 1) // 2
        self.fc1 = nn.Linear(channels[1] * size * size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classification_size)

        self.act = activation()
        self.dropout = nn.Dropout(dropout)

        # Initialization
        if init is not None:
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    init(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        x = self.fc3(x)
        return x

# %% [markdown]
# ## Training and Evaluation Functions

# %%
def train_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)


    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        if not isinstance(loss_fn, nn.CrossEntropyLoss):
            pred = pred.squeeze(-1)
            assert pred.shape == y.shape, (
                f"Shape mismatch: {pred.shape} vs {y.shape}"
            )
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        # Updates during process by giving loss every 100th batch
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return total_loss / num_batches

def evaluate(model, dataloader, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if not isinstance(loss_fn, nn.CrossEntropyLoss):
                pred = pred.squeeze(-1)
                assert pred.shape == y.shape, (
                    f"Shape mismatch: {pred.shape} vs {y.shape}"
                )
            test_loss += loss_fn(pred, y).item()

            if isinstance(loss_fn, nn.CrossEntropyLoss):
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = test_loss / num_batches

    
    accuracy  = correct / size

    return avg_loss, accuracy

# %% [markdown]
# ## Creating baselines for MLP and CNN

# %%
learning_rate = 1e-3
batch_size = 64
epochs = 15

mlp_baseline_path = "results/mlp_baseline_history.json"

if os.path.exists(mlp_baseline_path) and not FORCE_RETRAIN:
    with open(mlp_baseline_path) as f:
        mlp_baseline_history = json.load(f)
    print(f"Loaded mlp_baseline_history from {mlp_baseline_path}")

else: 
    torch.manual_seed(42)

    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlp_baseline_history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_dataloader, loss_fn)

        mlp_baseline_history["train_loss"].append(train_loss)
        mlp_baseline_history["val_loss"].append(val_loss)
        mlp_baseline_history["val_acc"].append(val_acc)
        
        print(f"epoch {epoch+1}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

    with open("results/mlp_baseline_history.json", "w") as f:
        json.dump(mlp_baseline_history, f, indent=1)

# %%
cnn_baseline_path = "results/cnn_baseline_history.json"

if os.path.exists(cnn_baseline_path) and not FORCE_RETRAIN:
    with open(cnn_baseline_path) as f:
        cnn_baseline_history = json.load(f)
    print(f"Loaded cnn_baseline_history from {cnn_baseline_path}")

else: 
    torch.manual_seed(42)

    model = CNN().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    cnn_baseline_history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_dataloader, loss_fn)

        cnn_baseline_history["train_loss"].append(train_loss)
        cnn_baseline_history["val_loss"].append(val_loss)
        cnn_baseline_history["val_acc"].append(val_acc)

        print(f"epoch {epoch+1}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")


    with open("results/cnn_baseline_history.json", "w") as f:
        json.dump(cnn_baseline_history, f, indent=1)

# %% [markdown]
# ## Hyperparameter experiments on Fashion-MNIST dataset 

# %%
# All hyperparameters we are going to experiment with
activations = {
    "relu": nn.ReLU,
    "leaky_relu": partial(nn.LeakyReLU, 0.01),
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}
inits = {
    "default": None,                                    # PyTorch's built-in init
    "kaiming": nn.init.kaiming_normal_,
    "xavier": nn.init.xavier_uniform_,
    "normal": partial(nn.init.normal_, mean=0.0, std=0.01),
}
optimizers = {
    "sgd": partial(optim.SGD, momentum=0.9),
    "adam": optim.Adam,                                 # weight_decay here = L2 penalty
    "adamw": optim.AdamW,                               # weight_decay here = decoupled decay
}

# Base Configuration 
base = dict(
    init="default", activation="relu",
    optimizer="adam", lr=1e-3,
    dropout=0.0, weight_decay=0.0,
    depth=2, width=512,
    epochs=15, patience=3,
)

# %%
# Trains a model with early stopping and saves the results of the run
def run_epochs_with_patience_save_results(model, train_dataloader, val_dataloader, loss_fn, optimizer, results, cfg, name):    
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    bad_epochs = 0
    best_epoch = None
    best_state = None                                   

    for epoch in range(cfg["epochs"]):
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_dataloader, loss_fn)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())

        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"Early stopping - epoch {epoch}")
                break

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"epoch {epoch+1}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

    if best_state is not None:                            
        model.load_state_dict(best_state)

    results.append({
        "name": name, **cfg,
        "config":cfg,
        "best_val_loss": best_val_loss,
        "best_val_acc": history["val_acc"][best_epoch] if best_epoch is not None else None,
        "best_epoch": best_epoch + 1 if best_epoch is not None else None,
        "params": sum(p.numel() for p in model.parameters()),
        "history": history,
    })

    return model

# %%
def train_MLP(cfg, train_dataloader, val_dataloader, results, in_features, name):
    
    print(f'training MLP with config {cfg}')

    torch.manual_seed(42)
    loss_fn = nn.CrossEntropyLoss()

    model = MLP(activation=activations[cfg["activation"]], 
                width=cfg["width"], 
                dropout=cfg["dropout"], 
                depth=cfg["depth"],
                init=inits[cfg["init"]],
                in_features=in_features
                ).to(device)

    optimizer = optimizers[cfg["optimizer"]](
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    run_epochs_with_patience_save_results(model, train_dataloader, val_dataloader, loss_fn, optimizer, results, cfg, name)

    return model

# %%
def train_CNN(cfg, train_dataloader, val_dataloader, results, name, in_channels=1, image_size=28, classification_size=10, loss_fn_overide=None):
    
    print(f'training CNN with config {cfg}')

    torch.manual_seed(42)

    if loss_fn_overide is None:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = loss_fn_overide
        
    model = CNN(in_channels=in_channels,           
                image_size=image_size,              
                channels=cfg["channels"],           
                kernel_size=cfg["kernel_size"],     
                activation=activations[cfg["activation"]], 
                dropout=cfg["dropout"], 
                init=inits[cfg["init"]],
                classification_size=classification_size
                ).to(device)

    optimizer = optimizers[cfg["optimizer"]](
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    run_epochs_with_patience_save_results(model, train_dataloader, val_dataloader, loss_fn, optimizer, results, cfg, name)

    return model

# %%
# Each entry changes only what differs from the baseline (one factor at a time) for MLP
variations_mlp = [
    {},                                                 # baseline
    # parameter initialization
    {"init": "kaiming"}, {"init": "xavier"}, {"init": "normal"},
    # activation functions
    {"activation": "leaky_relu"}, {"activation": "gelu"}, {"activation": "tanh"},
    # optimizers + learning rates (paired, since good lr depends on the optimizer)
    {"optimizer": "sgd", "lr": 0.1}, {"optimizer": "sgd", "lr": 0.01},
    {"optimizer": "adam", "lr": 1e-4},
    {"optimizer": "adamw", "lr": 1e-3}, {"optimizer": "adamw", "lr": 1e-4},
    # regularization
    {"dropout": 0.2}, {"dropout": 0.5},
    {"weight_decay": 1e-4},                             # L2 penalty via Adam
    {"optimizer": "adamw", "weight_decay": 1e-2},       # decoupled weight decay
    {"dropout": 0.2, "optimizer": "adamw", "weight_decay": 1e-2},
    # architecture
    {"depth": 1}, {"depth": 4},
    {"width": 128}, {"width": 1024},
]

configs_mlp = [base | v for v in variations_mlp]

mlp_sweep_path = "results/mlp_sweep.json"

if os.path.exists(mlp_sweep_path) and not FORCE_RETRAIN:
    with open(mlp_sweep_path) as f:
        results_mlp = json.load(f)
    print(f"Loaded results from {mlp_sweep_path}")

else: 
    results_mlp = []

    for i, (cfg, change) in enumerate(zip(configs_mlp, variations_mlp), 1):
        name = ", ".join(f"{k}={v}" for k, v in change.items()) or "baseline"
        print(f"\n=== {i}/{len(configs_mlp)}: {name} ===")

        train_MLP(cfg=cfg,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    results=results_mlp,
                    in_features=28 * 28,
                    name=name)

df_mlp = pd.DataFrame(results_mlp).drop(columns="history").sort_values("best_val_loss")

# Save results: CSV for reading/report, JSON incl. per-epoch history for plots
os.makedirs("results", exist_ok=True)
df_mlp.to_csv("results/mlp_sweep.csv", index=False)
with open("results/mlp_sweep.json", "w") as f:
    json.dump(results_mlp, f, indent=1)

df_mlp



# %%
base_cnn = {k: v for k, v in base.items() if k not in ("depth", "width")} | dict(
    channels=(6, 16), kernel_size=5,
)

# Each entry changes only what differs from the baseline (one factor at a time) for CNN
variations_cnn = [
    {},                                                 # baseline
    # parameter initialization
    {"init": "kaiming"}, {"init": "xavier"}, {"init": "normal"},
    # activation functions
    {"activation": "leaky_relu"}, {"activation": "gelu"}, {"activation": "tanh"},
    # optimizers + learning rates (paired, since good lr depends on the optimizer)
    {"optimizer": "sgd", "lr": 0.1}, {"optimizer": "sgd", "lr": 0.01},
    {"optimizer": "adam", "lr": 1e-4},
    {"optimizer": "adamw", "lr": 1e-3}, {"optimizer": "adamw", "lr": 1e-4},
    # regularization
    {"dropout": 0.2}, {"dropout": 0.5},
    {"weight_decay": 1e-4},                             # L2 penalty via Adam
    {"optimizer": "adamw", "weight_decay": 1e-2},       # decoupled weight decay
    {"dropout": 0.2, "optimizer": "adamw", "weight_decay": 1e-2},
    # architecture: number of filters (kernel size fixed at 5)
    {"channels": (3, 8)},  {"channels": (16, 32)},  {"channels": (32, 64)},
    # architecture: kernel size (filters fixed at 6/16)
    {"kernel_size": 3},    {"kernel_size": 7},
]
configs_cnn = [base_cnn | v for v in variations_cnn]

cnn_sweep_path = "results/cnn_sweep.json"

if os.path.exists(cnn_sweep_path) and not FORCE_RETRAIN:
    with open(cnn_sweep_path) as f:
        results_cnn = json.load(f)
    print(f"Loaded results from {cnn_sweep_path}")

else: 
    results_cnn = []

    for i, (cfg, change) in enumerate(zip(configs_cnn, variations_cnn), 1):
        name = ", ".join(f"{k}={v}" for k, v in change.items()) or "baseline"
        print(f"\n=== {i}/{len(configs_cnn)}: {name} ===")

        train_CNN(cfg=cfg,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    results=results_cnn,
                    name=name)

df_cnn = pd.DataFrame(results_cnn).drop(columns="history").sort_values("best_val_loss")

# Save results: CSV for reading/report, JSON incl. per-epoch history for plots
os.makedirs("results", exist_ok=True)
df_cnn.to_csv("results/cnn_sweep.csv", index=False)
with open("results/cnn_sweep.json", "w") as f:
    json.dump(results_cnn, f, indent=1)

df_cnn

# %%
def train_cnn_with_different_configuration_save_results(train_dataloader, val_dataloader, classification_size, image_size=28, results_file_name= "cnn_sweep.json", epoch_overide = None, loss_function = None):

    os.makedirs("results", exist_ok=True)
    results_file_name_json = "results/"+results_file_name+".json"

    if os.path.exists(results_file_name_json) and not FORCE_RETRAIN:
        with open(results_file_name_json) as f:
            results_cnn = json.load(f)
        print(f"Loaded results from {results_file_name_json}")

    else: 
        results_cnn = []

        for i, (cfg, change) in enumerate(zip(configs_cnn, variations_cnn), 1):
            name = ", ".join(f"{k}={v}" for k, v in change.items()) or "baseline"
            print(f"\n=== {i}/{len(configs_cnn)}: {name} ===")
            cfg = cfg.copy()
            
            if epoch_overide:
                cfg["epochs"] = epoch_overide

            model = train_CNN(cfg=cfg,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        results=results_cnn,
                        name=name,
                        image_size=image_size,
                        classification_size=classification_size,
                        loss_fn_overide=loss_function)

            model_file_name = Path(f"models/{results_file_name.replace('-75','')}/{name}.pth")
            model_file_name.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_file_name)

    df_cnn = pd.DataFrame(results_cnn).drop(columns="history").sort_values("best_val_loss")

    # Save results: CSV for reading/report, JSON incl. per-epoch history for plots
    df_cnn.to_csv(results_file_name_json.replace(".json", ".csv"), index=False)
    with open(results_file_name_json, "w") as f:
        json.dump(results_cnn, f, indent=1)

    df_cnn

# %% [markdown]
# ## Using top 3 configurations for CIFAR-10

# %%
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

# Download training data from open datasets.
training_cifar = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

# Download test data from open datasets.
test_cifar = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

# Split 10% of the training data for validation
generator = torch.Generator().manual_seed(42)
train_cifar, val_cifar = random_split(training_cifar, [45000, 5000], generator=generator)

# %%
# Create dataloaders and finds top 3 configurations
batch_size = 64

train_cifar_dataloader = DataLoader(train_cifar, batch_size=batch_size, shuffle=True)
test_cifar_dataloader = DataLoader(test_cifar, batch_size=batch_size)
val_cifar_dataloader = DataLoader(val_cifar, batch_size=batch_size)

# Opens results from the sweeps and finds top 3 configurations for MLP and CNN 
with open("results/mlp_sweep.json") as f:
    mlp_loaded = json.load(f)

with open("results/cnn_sweep.json") as f:
    cnn_loaded = json.load(f)

sorted_loss_mlp = sorted(mlp_loaded, key=lambda s: s["best_val_loss"]) 
top_3_mlp = sorted_loss_mlp[:3]

sorted_loss_cnn = sorted(cnn_loaded, key=lambda s: s["best_val_loss"]) 
top_3_cnn = sorted_loss_cnn[:3]

# %%
# MLP on CIFAR dataset with top 3 config
mlp_cifar_path = "results/cifar_mlp.json"
cnn_cifar_path = "results/cifar_cnn.json"

if os.path.exists(mlp_cifar_path) and os.path.exists(cnn_cifar_path) and not FORCE_RETRAIN:
    with open(mlp_cifar_path) as f:
        results_cifar_mlp = json.load(f)
    print(f"Loaded results_cifar_mlp from {mlp_cifar_path}")

    with open(cnn_cifar_path) as f:
        results_cifar_cnn = json.load(f)
    print(f"Loaded results_cifar_cnn from {cnn_cifar_path}")


else: 
    results_cifar_mlp = []
    for configuration in top_3_mlp:
        train_MLP(cfg=configuration["config"], 
                train_dataloader=train_cifar_dataloader, 
                val_dataloader=val_cifar_dataloader, 
                results=results_cifar_mlp, 
                in_features= 32 * 32 * 3, 
                name=f"CIFAR - {configuration['name']}")

    # CNN on CIFAR dataset with top 3 config
    results_cifar_cnn = []
    for configuration in top_3_cnn:
        train_CNN(cfg=configuration["config"], 
                train_dataloader=train_cifar_dataloader, 
                val_dataloader=val_cifar_dataloader, 
                results=results_cifar_cnn, 
                in_channels=3, 
                image_size=32,
                name=f"CIFAR - {configuration['name']}")


# Save results and show corresponding tables
os.makedirs("results", exist_ok=True)

df_cifar_mlp = pd.DataFrame(results_cifar_mlp).drop(columns="history").sort_values("best_val_loss")
df_cifar_cnn = pd.DataFrame(results_cifar_cnn).drop(columns="history").sort_values("best_val_loss")

df_cifar_mlp.to_csv("results/cifar_mlp.csv", index=False)
df_cifar_cnn.to_csv("results/cifar_cnn.csv", index=False)

with open("results/cifar_mlp.json", "w") as f:
    json.dump(results_cifar_mlp, f, indent=1)

with open("results/cifar_cnn.json", "w") as f:
    json.dump(results_cifar_cnn, f, indent=1)

display(df_cifar_mlp)
display(df_cifar_cnn)

# %% [markdown]
# ## Using test set

# %%
# Evaluate the top 3 configurations of each model on the test sets 
test_path = "results/test_results.json"

if os.path.exists(test_path) and not FORCE_RETRAIN:
    with open(test_path) as f:
        test_results = json.load(f)
    print(f"Loaded test results from {test_path}")

else:
    test_results = []
    loss_fn = nn.CrossEntropyLoss()

    # MLP on Fashion-MNIST
    for configuration in top_3_mlp:
        config_results = []
        model = train_MLP(cfg=configuration["config"], 
                        train_dataloader=train_dataloader, 
                        val_dataloader=val_dataloader, 
                        results=config_results, 
                        in_features=28 * 28, 
                        name=f"Fashion-MNIST - {configuration['name']}")
        
        test_loss, test_acc = evaluate(model, test_dataloader, loss_fn)
        test_results.append({ 
             "model": "MLP",
             "dataset": "Fashion-MNIST",
             "name": configuration["name"],
             "val_acc": config_results[-1]["best_val_acc"],
             "test_acc": test_acc,
             "test_loss": test_loss,
             "params": config_results[-1]["params"],
        })

    # MLP on CIFAR-10
    for configuration in top_3_mlp:
        config_results = []
        model = train_MLP(cfg=configuration["config"], 
                        train_dataloader=train_cifar_dataloader, 
                        val_dataloader=val_cifar_dataloader, 
                        results=config_results, 
                        in_features=32 * 32 * 3, 
                        name=f"CIFAR-10 - {configuration['name']}")
        
        test_loss, test_acc = evaluate(model, test_cifar_dataloader, loss_fn)
        test_results.append({ 
             "model": "MLP",
             "dataset": "CIFAR-10",
             "name": configuration["name"],
             "val_acc": config_results[-1]["best_val_acc"],
             "test_acc": test_acc,
             "test_loss": test_loss,
             "params": config_results[-1]["params"],
        })

    # CNN on Fashion-MNIST
    for configuration in top_3_cnn:
        config_results = []
        model = train_CNN(cfg=configuration["config"], 
                        train_dataloader=train_dataloader, 
                        val_dataloader=val_dataloader, 
                        results=config_results, 
                        name=f"Fashion-MNIST - {configuration['name']}")
        
        test_loss, test_acc = evaluate(model, test_dataloader, loss_fn)
        test_results.append({ 
             "model": "CNN",
             "dataset": "Fashion-MNIST",
             "name": configuration["name"],
             "val_acc": config_results[-1]["best_val_acc"],
             "test_acc": test_acc,
             "test_loss": test_loss,
             "params": config_results[-1]["params"],
        })

    # CNN on CIFAR-10
    for configuration in top_3_cnn:
        config_results = []
        model = train_CNN(cfg=configuration["config"], 
                        train_dataloader=train_cifar_dataloader, 
                        val_dataloader=val_cifar_dataloader, 
                        results=config_results, 
                        in_channels=3,
                        image_size=32,
                        name=f"CIFAR-10 - {configuration['name']}")
        
        test_loss, test_acc = evaluate(model, test_cifar_dataloader, loss_fn)
        test_results.append({ 
             "model": "CNN",
             "dataset": "CIFAR-10",
             "name": configuration["name"],
             "val_acc": config_results[-1]["best_val_acc"],
             "test_acc": test_acc,
             "test_loss": test_loss,
             "params": config_results[-1]["params"],
        })

    with open(test_path, "w") as f:
        json.dump(test_results, f, indent=1)

df_test = pd.DataFrame(test_results)
df_test.to_csv("results/test_results.csv", index=False)
display(df_test)

# %% [markdown]
# ## Making plots for task 1

# %% [markdown]
# ### Baseline plots

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)

for ax, history, title in [(axes[0], mlp_baseline_history, "MLP baseline"),
                           (axes[1], cnn_baseline_history, "CNN baseline")]:
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], marker="o", markersize=3, label="Training loss")
    ax.plot(epochs, history["val_loss"], marker="o", markersize=3, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Cross-entropy loss")
axes[0].legend()

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/baselines_loss.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Sweep plots

# %%
category_order = ["Baseline", "Initialisation", "Activation", "Optimiser", "Regularisation", "Architecture"]
category_colors = dict(zip(category_order, plt.cm.tab10.colors))

LABEL_NAMES = {
    "init": {"kaiming": "Kaiming init", "xavier": "Xavier init", "normal": "N(0, 0.01²) init"},
    "activation": {"leaky_relu": "Leaky ReLU", "gelu": "GELU", "tanh": "tanh"},
    "optimizer": {"sgd": "SGD", "adam": "Adam", "adamw": "AdamW"},
}

def format_label(name):
    if name == "baseline":
        return "Baseline"
    parts = []
    for item in re.split(r", (?=\w+=)", name):           # split on ', ' only before 'key='. This keeps '(32, 64)' intact
        key, value = item.split("=", 1)
        if key in LABEL_NAMES:
            parts.append(LABEL_NAMES[key][value])
        elif key == "lr":
            parts.append(f"lr 1e{round(math.log10(float(value)))}")
        elif key == "weight_decay":
            parts.append(f"wd {float(value):g}")
        elif key == "channels":
            parts.append(f"filters {value}")
        elif key == "kernel_size":
            parts.append(f"kernel {value}")
        else:                                              
            parts.append(f"{key} {value}")
    return ", ".join(parts)

def sweep_category(name):
    if name == "baseline":
        return "Baseline"
    if "dropout" in name or "weight_decay" in name:
        return "Regularisation"
    if name.startswith("init"):
        return "Initialisation"
    if name.startswith("activation"):
        return "Activation"
    if "optimizer" in name or "lr" in name:
        return "Optimiser"
    if any(key in name for key in ("depth", "width", "channels", "kernel_size")):
        return "Architecture"
    return "Other"

def plot_sweep_panel(ax, df, title):
    d = df.copy()
    d["category"] = d["name"].apply(sweep_category)
    d["category_rank"] = d["category"].map(category_order.index)
    d = d.sort_values(["category_rank", "best_val_acc"], ascending=[True, False]).reset_index(drop=True)
    d["y"] = range(len(d))

    for cat in category_order:
        sub = d[d["category"] == cat]
        ax.scatter(sub["best_val_acc"], sub["y"], color=category_colors[cat], label=cat, s=40, zorder=3)

    baseline_acc = d.loc[d["name"] == "baseline", "best_val_acc"].iloc[0]
    ax.axvline(baseline_acc, linestyle="--", color="grey", linewidth=1, label="Baseline accuracy")

    ax.set_yticks(d["y"])
    ax.set_yticklabels([format_label(n) for n in d["name"]], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Validation accuracy at lowest validation loss")
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharex=True)
plot_sweep_panel(axes[0], df_mlp, "MLP (Fashion-MNIST)")
plot_sweep_panel(axes[1], df_cnn, "CNN (Fashion-MNIST)")

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.05))

fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("figures/sweep_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Fashion-MNIST vs CIFAR-10

# %%
def plot_transfer_test(ax, df_test, model_type, title):
    d = df_test[df_test["model"] == model_type]

    for rank, name in enumerate(d["name"].unique()):          # same order as the top-3 (rank 1, 2, 3)
        fmnist = d[(d["name"] == name) & (d["dataset"] == "Fashion-MNIST")]["test_acc"].iloc[0]
        cifar = d[(d["name"] == name) & (d["dataset"] == "CIFAR-10")]["test_acc"].iloc[0]
        ax.plot([0, 1], [fmnist, cifar], marker="o", color=plt.cm.tab10.colors[rank], linewidth=2,
                label=format_label(name))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Fashion-MNIST", "CIFAR-10"])
    ax.set_xlim(-0.2, 1.2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Test accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
plot_transfer_test(axes[0], df_test, "MLP", "MLP: top-3 configurations")
plot_transfer_test(axes[1], df_test, "CNN", "CNN: top-3 configurations")

fig.tight_layout()
fig.savefig("figures/transfer_fmnist_cifar_test.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Task 2

# %%
# Create CustomDataset such that we can apply randomsplits 
class CustomDataset(Dataset):
    def __init__(self, x, y, dtype_y, transform):
      self.x = x
      self.y = y
      self.dtype_y = dtype_y
      self.transform = transform

    def __getitem__(self, index):
      added_channel = np.expand_dims(self.x, axis=-1)  # Add color channel to the end such that it has the expexted np shape
      transformed_x = self.transform(added_channel[index])
      return (transformed_x, torch.tensor(self.y[index], dtype=self.dtype_y))

    def __len__(self):
        return len(self.x)


""" 
Problems with using high cardinality classification:
- Each bin is independent of eachother. Meaning that times that are 1 minute apart have no connection and close guesses don't play a role in the cross entropy loss
- Some classes might have very few or no training examples
- 
"""



# %%
import re, sys, zipfile, requests
from pathlib import Path

url = "https://surfdrive.surf.nl/index.php/s/Nznt5c48Mzlb2HY/download"
dest = Path("./data/Clock")
dest.mkdir(parents=True, exist_ok=True)

if any(dest.iterdir()):
    print(f"{dest} is not empty, skipping download")
else:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        cd = r.headers.get("content-disposition", "")
        m = re.search(r'filename="?([^";]+)"?', cd)
        archive = dest / (m.group(1) if m else "download.zip")
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(archive, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    msg = f"Downloading: {done/1e6:.1f} / {total/1e6:.1f} MB ({done*100/total:.1f}%)"
                else:
                    msg = f"Downloading: {done/1e6:.1f} MB"
                sys.stdout.write("\r" + msg)
                sys.stdout.flush()
    print()

    # --- Extract top-level archive into /data/Clock ---
    if zipfile.is_zipfile(archive):
        print(f"Extracting {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        archive.unlink()

    # --- Recursively extract nested zips ---
    while True:
        nested = [p for p in dest.rglob("*.zip") if zipfile.is_zipfile(p)]
        if not nested:
            break
        for z in nested:
            target = z.with_suffix("")
            target.mkdir(exist_ok=True)
            print(f"Extracting {z.relative_to(dest)}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)
            z.unlink()

    print("Done:", sum(1 for _ in dest.rglob("*")), "items in", dest)

# %%
# Download the datasets and open them here
clock_75_images = np.load("./data/Clock/A1 Data/A1_data_75/images.npy")
clock_75_labels = np.load("./data/Clock/A1 Data/A1_data_75/labels.npy")

clock_150_images = np.load("./data/Clock/A1 Data/A1_data_150/images.npy")
clock_150_labels = np.load("./data/Clock/A1 Data/A1_data_150/labels.npy")

# %%
def classification_labels(y, num_classes=720):
    y_minutes = y[:,0]*60 + y[:,1]
    total_minutes = 60*12

    # create bin edges
    bin_edges = np.linspace(0, total_minutes, num_classes+1)
    # bin the data
    bin_indices = np.digitize(y_minutes, bin_edges) - 1

    return bin_indices

# %%
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

# Downsampled dataset
cls_dataset_30min_75 = CustomDataset(x=clock_75_images, y=classification_labels(clock_75_labels, num_classes=24), dtype_y=torch.long, transform=transform)
cls_dataset_15min_75 = CustomDataset(x=clock_75_images, y=classification_labels(clock_75_labels, num_classes=48), dtype_y=torch.long, transform=transform)
cls_dataset_5min_75 = CustomDataset(x=clock_75_images, y=classification_labels(clock_75_labels, num_classes=144), dtype_y=torch.long, transform=transform)
cls_dataset_1min_75 = CustomDataset(x=clock_75_images, y=classification_labels(clock_75_labels, num_classes=720), dtype_y=torch.long, transform=transform)


len_dataset = len(clock_75_images)
# Downsampled dataset
# Split 10% of the training data for validation and 10% for testing
generator = torch.Generator().manual_seed(42)
train_30min_75, val_30min_75, test_30min_75 = random_split(cls_dataset_30min_75, [int(0.8*len_dataset), int(0.1*len_dataset), int(0.1*len_dataset)], generator=generator)

generator = torch.Generator().manual_seed(42)
train_15min_75, val_15min_75, test_15min_75 = random_split(cls_dataset_15min_75, [int(0.8*len_dataset), int(0.1*len_dataset), int(0.1*len_dataset)], generator=generator)

generator = torch.Generator().manual_seed(42)
train_5min_75, val_5min_75, test_5min_75 = random_split(cls_dataset_5min_75, [int(0.8*len_dataset), int(0.1*len_dataset), int(0.1*len_dataset)], generator=generator)

generator = torch.Generator().manual_seed(42)
train_1min_75, val_1min_75, test_1min_75 = random_split(cls_dataset_1min_75, [int(0.8*len_dataset), int(0.1*len_dataset), int(0.1*len_dataset)], generator=generator)

"""
Important do not remove the generator before each random_split. This ensures that we split the dataset on the same indices each time which is needed for the common sense error
"""
print(test_30min_75.indices == test_1min_75.indices)
print(test_15min_75.indices == test_1min_75.indices)
print(test_5min_75.indices == test_1min_75.indices)

# %%
batch_size = 64

# Create data loaders
train_dataloader_30min_75 = DataLoader(train_30min_75, batch_size=batch_size, shuffle=True)
test_dataloader_30min_75 = DataLoader(test_30min_75, batch_size=batch_size)
val_dataloader_30min_75 = DataLoader(val_30min_75, batch_size=batch_size)

train_dataloader_15min_75 = DataLoader(train_15min_75, batch_size=batch_size, shuffle=True)
test_dataloader_15min_75 = DataLoader(test_15min_75, batch_size=batch_size)
val_dataloader_15min_75 = DataLoader(val_15min_75, batch_size=batch_size)

train_dataloader_5min_75 = DataLoader(train_5min_75, batch_size=batch_size, shuffle=True)
test_dataloader_5min_75 = DataLoader(test_5min_75, batch_size=batch_size)
val_dataloader_5min_75 = DataLoader(val_5min_75, batch_size=batch_size)

train_dataloader_1min_75 = DataLoader(train_1min_75, batch_size=batch_size, shuffle=True)
test_dataloader_1min_75 = DataLoader(test_1min_75, batch_size=batch_size)
val_dataloader_1min_75 = DataLoader(val_1min_75, batch_size=batch_size)

for X, y in test_dataloader_30min_75:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# %%
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

# Normal dataset
cls_dataset_30min_150 = CustomDataset(x=clock_150_images, y=classification_labels(clock_150_labels, num_classes=24), dtype_y=torch.long, transform=transform)
cls_dataset_15min_150 = CustomDataset(x=clock_150_images, y=classification_labels(clock_150_labels, num_classes=48), dtype_y=torch.long, transform=transform)
cls_dataset_5min_150 = CustomDataset(x=clock_150_images, y=classification_labels(clock_150_labels, num_classes=144), dtype_y=torch.long, transform=transform)
cls_dataset_1min_150 = CustomDataset(x=clock_150_images, y=classification_labels(clock_150_labels, num_classes=720), dtype_y=torch.long, transform=transform)


len_dataset = len(clock_150_images)
# Normal dataset
generator = torch.Generator().manual_seed(42)
train_30min_150, val_30min_150, test_30min_150 = random_split(cls_dataset_30min_150, [int(0.8*len_dataset), int(0.1*len_dataset), int(0.1*len_dataset)], generator=generator)

generator = torch.Generator().manual_seed(42)
train_15min_150, val_15min_150, test_15min_150 = random_split(cls_dataset_15min_150, [int(0.8*len_dataset), int(0.1*len_dataset), int(0.1*len_dataset)], generator=generator)

generator = torch.Generator().manual_seed(42)
train_5min_150, val_5min_150, test_5min_150 = random_split(cls_dataset_5min_150, [int(0.8*len_dataset), int(0.1*len_dataset), int(0.1*len_dataset)], generator=generator)

generator = torch.Generator().manual_seed(42)
train_1min_150, val_1min_150, test_1min_150 = random_split(cls_dataset_1min_150, [int(0.8*len_dataset), int(0.1*len_dataset), int(0.1*len_dataset)], generator=generator)




# %%
batch_size = 64

# Create data loaders
train_dataloader_30min_150 = DataLoader(train_30min_150, batch_size=batch_size, shuffle=True)
test_dataloader_30min_150 = DataLoader(test_30min_150, batch_size=batch_size)
val_dataloader_30min_150 = DataLoader(val_30min_150, batch_size=batch_size)

train_dataloader_15min_150 = DataLoader(train_15min_150, batch_size=batch_size, shuffle=True)
test_dataloader_15min_150 = DataLoader(test_15min_150, batch_size=batch_size)
val_dataloader_15min_150 = DataLoader(val_15min_150, batch_size=batch_size)

train_dataloader_5min_150 = DataLoader(train_5min_150, batch_size=batch_size, shuffle=True)
test_dataloader_5min_150 = DataLoader(test_5min_150, batch_size=batch_size)
val_dataloader_5min_150 = DataLoader(val_5min_150, batch_size=batch_size)

train_dataloader_1min_150 = DataLoader(train_1min_150, batch_size=batch_size, shuffle=True)
test_dataloader_1min_150 = DataLoader(test_1min_150, batch_size=batch_size)
val_dataloader_1min_150 = DataLoader(val_1min_150, batch_size=batch_size)

for X, y in test_dataloader_30min_150:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# %%
def common_sense_error_classification(y_pred, y_real, num_classes):
    # We take the bincenters as distance measure unless there is a prediction label for each minute
    y_pred = y_pred.float()
    y_real = y_real.float()

    if num_classes != 720:
        size_interval = 720 / num_classes
        pred_bin_center = y_pred * size_interval + 0.5 * size_interval

        diff = torch.remainder(torch.abs(y_real-pred_bin_center), 720)  # If difference is bigger than full clock cycle take the remainder
        cse = torch.minimum(diff, 720-diff)

    else:
        diff = torch.remainder(torch.abs(y_real-y_pred), 720)
        cse = torch.minimum(diff, 720-diff)
    return torch.mean(cse)

def test_error_classification(model, num_classes, dataloader, per_minute_dataloader=test_dataloader_1min_75, error_fn = common_sense_error_classification):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_error, correct = 0, 0

    with torch.no_grad():
        for X, y in per_minute_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_error += error_fn(pred.argmax(1), y, num_classes).item() * len(y)

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_error = test_error / size
    accuracy  = correct / size

    return avg_error, accuracy

# %%
clock_data_75={
    "30min":{
            "validation":val_dataloader_30min_75,
            "train":train_dataloader_30min_75,
            "test":test_dataloader_30min_75,
            "num_classes":24,
            "epochs": 30
    },
    "15min":{
            "validation":val_dataloader_15min_75,
            "train":train_dataloader_15min_75,
            "test":test_dataloader_15min_75,
            "num_classes":48,
            "epochs": 30
    },
    "5min":{
            "validation":val_dataloader_5min_75,
            "train":train_dataloader_5min_75,
            "test":test_dataloader_5min_75,
            "num_classes":144,
            "epochs": 30
    },
    "1min":{
            "validation":val_dataloader_1min_75,
            "train":train_dataloader_1min_75,
            "test":test_dataloader_1min_75,
            "num_classes":720,
            "epochs": 30
    }
}


clock_data_150={
    "30min":{
            "validation":val_dataloader_30min_150,
            "train":train_dataloader_30min_150,
            "test":test_dataloader_30min_150,
            "num_classes":24,
            "epochs": 30
    },
    "15min":{
            "validation":val_dataloader_15min_150,
            "train":train_dataloader_15min_150,
            "test":test_dataloader_15min_150,
            "num_classes":48,
            "epochs": 30
    },
    "5min":{
            "validation":val_dataloader_5min_150,
            "train":train_dataloader_5min_150,
            "test":test_dataloader_5min_150,
            "num_classes":144,
            "epochs": 30
    },
    "1min":{
            "validation":val_dataloader_1min_150,
            "train":train_dataloader_1min_150,
            "test":test_dataloader_1min_150,
            "num_classes":720,
            "epochs": 30
    }
}

# %%
for time_divider, splits in clock_data_75.items():
    print(f"split {time_divider} with {splits['num_classes']} classes")

    train_cnn_with_different_configuration_save_results(
        train_dataloader=splits["train"], 
        val_dataloader=splits["validation"],  
        image_size = 75, 
        classification_size=splits["num_classes"], 
        results_file_name=time_divider+"-75",
        epoch_overide=splits["epochs"]
    )

# %%
for time_divider, splits in clock_data_75.items():
    print(f"calculating error rate for {time_divider}")

    result_directory = Path(f"models/{time_divider}")
    error_output = []

    for i, (cfg, change) in enumerate(zip(configs_cnn, variations_cnn), 1):
        name = ", ".join(f"{k}={v}" for k, v in change.items()) or "baseline"

        model_config = next(result_directory.rglob(f"{name}.pth"), None)   

        model = CNN( image_size=75,              
                channels=cfg["channels"],           
                kernel_size=cfg["kernel_size"],     
                activation=activations[cfg["activation"]], 
                dropout=cfg["dropout"], 
                init=inits[cfg["init"]],
                classification_size=splits['num_classes']
                ).to(device)
        
        model.load_state_dict(torch.load(model_config, weights_only=True))
        avg_error, accuracy = test_error_classification(model, splits['num_classes'], splits['test'])

        error_message = f"{name} - Average Test Error: {avg_error:.1f} minutes | Accuracy: {accuracy:.3f}"

        error_output.append(error_message)
        print(error_message)

    with open(f"{result_directory}/error_summary.txt", "w") as f:
        f.write("\n".join(error_output))

# %%
for time_divider, splits in clock_data_150.items():
    print(f"split {time_divider} with {splits['num_classes']} classes")

    train_cnn_with_different_configuration_save_results(
        train_dataloader=splits["train"], 
        val_dataloader=splits["validation"],  
        image_size = 150, 
        classification_size=splits["num_classes"], 
        results_file_name=time_divider+"-150"
    )

# %%
for time_divider, splits in clock_data_150.items():
    print(f"calculating error rate for {time_divider}")

    result_directory = Path(f"models/{time_divider}-150")
    error_output = []

    for i, (cfg, change) in enumerate(zip(configs_cnn, variations_cnn), 1):
        name = ", ".join(f"{k}={v}" for k, v in change.items()) or "baseline"

        model_config = next(result_directory.rglob(f"{name}.pth"), None)   

        model = CNN( image_size=150,              
                channels=cfg["channels"],           
                kernel_size=cfg["kernel_size"],     
                activation=activations[cfg["activation"]], 
                dropout=cfg["dropout"], 
                init=inits[cfg["init"]],
                classification_size=splits['num_classes']
                ).to(device)
        
        model.load_state_dict(torch.load(model_config, weights_only=True))
        avg_error, accuracy = test_error_classification(model, splits['num_classes'], splits['test'], per_minute_dataloader=test_dataloader_1min_150)

        error_message = f"{name} - Average Test Error: {avg_error:.1f} minutes | Accuracy: {accuracy:.3f}"

        error_output.append(error_message)
        print(error_message)

    with open(f"{result_directory}/error_summary.txt", "w") as f:
        f.write("\n".join(error_output))

# %% [markdown]
# Plots for taks 2 - classification

# %%
def plot_error_panel(ax, df, title, chance=180):
    d = df.copy()
    d["category"] = d["name"].apply(sweep_category)
    d["category_rank"] = d["category"].map(category_order.index)
    d = d.sort_values(["category_rank", "test_error_min"]).reset_index(drop=True)
    d["y"] = range(len(d))

    for cat in category_order:
        sub = d[d["category"] == cat]
        ax.scatter(sub["test_error_min"], sub["y"], color=category_colors[cat], label=cat, s=45, zorder=3)
        for x, y in zip(sub["test_error_min"], sub["y"]):
            ax.annotate(f"{x:.1f}", (x, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7)

    failed = d[d["test_error_min"].isna()]
    for y in failed["y"]:
        ax.annotate("diverged (NaN)", (chance, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7, color="red")

    baseline_err = d.loc[d["name"] == "baseline", "test_error_min"].iloc[0]
    ax.axvline(baseline_err, linestyle="--", color="grey", linewidth=1, label="Baseline error")
    ax.axvline(chance, linestyle=":", color="red", linewidth=1, label="Chance (~180 min)")

    ax.set_yticks(d["y"])
    ax.set_yticklabels(d["name"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, chance * 1.15)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Average test error (minutes)")
    ax.set_title(title)


# %%
def plot_minutes_error(title, path):

    raw = Path(path).read_text()

    err_df = (
        pd.Series(raw.splitlines())
        .str.extract(r"(?P<name>.+) - Average Test Error: (?P<test_error_min>[\d.]+|nan) minutes")
        .dropna(subset=["name"])
        .astype({"test_error_min": float})
        .reset_index(drop=True)
    )


    fig, ax = plt.subplots(figsize=(8, 7))
    plot_error_panel(ax, err_df, f"{title} - Average Test Error")
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        fontsize=9,
        frameon=False
    )

    # Reserve space for the legend
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    plt.show()

# %%
plot_minutes_error("720 classes", "models/1min-150/error_summary.txt")
plot_minutes_error("144 classes", "models/5min-150/error_summary.txt")
plot_minutes_error("48 classes", "models/15min-150/error_summary.txt")
plot_minutes_error("24 classes", "models/30min-150/error_summary.txt")

# %%

def load_result_in_df(path):
    if os.path.exists(path) and not FORCE_RETRAIN:
        dt= pd.read_json(path)
        print(f"Loaded results from {path}")
        return dt


# %%
dt_1min_150 = load_result_in_df("./results/1min-150.json")
dt_5min_150 = load_result_in_df("./results/5min-150.json")
dt_15min_150 = load_result_in_df("./results/15min-150.json")
dt_30min_150 = load_result_in_df("./results/30min-150.json")


fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, layout="constrained")
plot_sweep_panel(axes[0], dt_1min_150, "Clock 1min - 720 class")
plot_sweep_panel(axes[1], dt_5min_150, "Clock 5min - 144 class")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7), sharex=True, layout="constrained")
plot_sweep_panel(axes2[0], dt_1min_150, "Clock 15min - 48 class")
plot_sweep_panel(axes2[1], dt_5min_150, "Clock 30min - 24 class")

# %% [markdown]
# Run Train Best Configuration from sweep

# %%

## train
def train_one_config_save_model_test_results(best_configuration_from_sweep_cnn, clock_data, image_size, per_minute_dataloader):

    name_best="best";

    for time_divider, splits in clock_data.items():
        print(f"split {time_divider} with {splits['num_classes']} classes")

        os.makedirs(f"results/best/{image_size}/", exist_ok=True)
        results_file_name_json_best = f"results/best/{image_size}/{time_divider}.json"

        if os.path.exists(results_file_name_json_best) and not FORCE_RETRAIN:
            with open(results_file_name_json_best) as f:
                results_best_sweep_cnn = json.load(f)
            print(f"Loaded results from {results_file_name_json_best}")
        else:
            results_best_sweep_cnn = []

            model = train_CNN(cfg=best_configuration_from_sweep_cnn,
                        train_dataloader=splits["train"],
                        val_dataloader=splits["validation"],
                        results=results_best_sweep_cnn,
                        name=name_best,
                        image_size=image_size,
                        classification_size=splits["num_classes"])

            model_file_name = Path(f"models/best/{image_size}/{time_divider}/{name_best}.pth")
            model_file_name.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_file_name)

            df_results_best_sweep_cnn = pd.DataFrame(results_best_sweep_cnn).drop(columns="history").sort_values("best_val_loss")
            
            df_results_best_sweep_cnn.to_csv(results_file_name_json_best.replace(".json", ".csv"), index=False)
            with open(results_file_name_json_best, "w") as f:
                json.dump(results_best_sweep_cnn, f, indent=1)


    # ##validate test data

    for time_divider, splits in clock_data.items():
        model_config = Path(f"models/best/{image_size}/{time_divider}/{name_best}.pth")  

        error_output_best = []

        model = CNN( image_size=image_size,              
                channels=best_configuration_from_sweep_cnn["channels"],           
                kernel_size=best_configuration_from_sweep_cnn["kernel_size"],     
                activation=activations[best_configuration_from_sweep_cnn["activation"]], 
                dropout=best_configuration_from_sweep_cnn["dropout"], 
                init=inits[best_configuration_from_sweep_cnn["init"]],
                classification_size=splits['num_classes']
                ).to(device)
            
        model.load_state_dict(torch.load(model_config, weights_only=True))
        avg_error, accuracy = test_error_classification(model, splits['num_classes'], splits['test'], per_minute_dataloader)

        error_message_best = f"{name} - Average Test Error: {avg_error:.1f} minutes | Accuracy: {accuracy:.3f}"

        error_output_best.append(error_message_best)
        print(error_message_best)

        with open(f"{model_config.parent}/error_summary.txt", "w") as f:
            f.write("\n".join(error_output_best))

# %%
best_configuration_from_sweep_cnn = dict(
    init="xavier", activation="leaky_relu",
    optimizer="adamw", lr=1e-3,
    dropout=0.2,  
    channels= (3, 8), kernel_size= 5,
    epochs=50, patience=10,
    weight_decay=0.0,
    num_classes= 75
)

train_one_config_save_model_test_results(best_configuration_from_sweep_cnn, clock_data_75, 75, test_dataloader_1min_75)
train_one_config_save_model_test_results(best_configuration_from_sweep_cnn, clock_data_150, 150, test_dataloader_1min_150)

# %% [markdown]
# ### Regression

# %%
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    def forward(self, predictions, targets):
        assert predictions.shape == targets.shape, (
            f"Shape mismatch: predictions {predictions.shape}, "
            f"targets {targets.shape}"
        )
        diff = torch.remainder(torch.abs(predictions-targets), 12)  # If difference is bigger than full clock cycle take the remainder
        cse = torch.minimum(diff, 12 - diff)
        mse_loss = torch.mean(cse ** 2)
        return mse_loss 

# %%
def regression_labels(y):
   y_scalar = y[:,0] + y[:,1]/60
   return y_scalar

def common_sense_error_regression(y_pred, y_real):
    diff = torch.remainder(torch.abs(y_pred-y_real), 12)  # If difference is bigger than full clock cycle take the remainder
    cse = torch.minimum(diff, 12 - diff)
    return torch.mean(cse)

def test_error_regression(model, dataloader, error_fn = common_sense_error_regression):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_error = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze(-1)

            assert pred.shape == y.shape

            test_error += error_fn(pred, y).item() * len(y)

    avg_error = 60* test_error / size 

    return avg_error



# %%
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
dataset_reg_75 = CustomDataset(x=clock_75_images, y=regression_labels(clock_75_labels), dtype_y=torch.float32, transform=transform)
dataset_reg_150 = CustomDataset(x=clock_150_images, y=regression_labels(clock_150_labels), dtype_y=torch.float32, transform=transform)

len_dataset = len(clock_75_images)
# Split 10% of the training data for validation and 10% for testing
generator = torch.Generator().manual_seed(42)
train_reg_75, val_reg_75, test_reg_75 = random_split(dataset_reg_75, [int(0.8*len_dataset), int(0.1*len_dataset), int(0.1*len_dataset)], generator=generator)
generator = torch.Generator().manual_seed(42)
train_reg_150, val_reg_150, test_reg_150 = random_split(dataset_reg_150, [int(0.8*len_dataset), int(0.1*len_dataset), int(0.1*len_dataset)], generator=generator)

# %%
batch_size = 64

# Create data loaders
train_dataloader_reg_75 = DataLoader(train_reg_75, batch_size=batch_size, shuffle=True)
test_dataloader_reg_75 = DataLoader(test_reg_75, batch_size=batch_size)
val_dataloader_reg_75 = DataLoader(val_reg_75, batch_size=batch_size)

train_dataloader_reg_150 = DataLoader(train_reg_150, batch_size=batch_size, shuffle=True)
test_dataloader_reg_150 = DataLoader(test_reg_150, batch_size=batch_size)
val_dataloader_reg_150 = DataLoader(val_reg_150, batch_size=batch_size)

# %%
def error_summary(name, test_data_loader, image_size):
        result_directory = Path(f"models/{name}")
        error_output = []

        for i, (cfg, change) in enumerate(zip(configs_cnn, variations_cnn), 1):
                name_file = ", ".join(f"{k}={v}" for k, v in change.items()) or "baseline"

                model_config = next(result_directory.rglob(f"{name_file}.pth"), None)   

                model = CNN( image_size=image_size,              
                        channels=cfg["channels"],           
                        kernel_size=cfg["kernel_size"],     
                        activation=activations[cfg["activation"]], 
                        dropout=cfg["dropout"], 
                        init=inits[cfg["init"]],
                        classification_size=1
                        ).to(device)

                model.load_state_dict(torch.load(model_config, weights_only=True))
                avg_error = test_error_regression(model, test_data_loader, error_fn=common_sense_error_regression)

                error_message = f"{name_file} - Average Test Error: {avg_error:.1f} minutes"

                error_output.append(error_message)
                print(error_message)

        with open(f"{result_directory}/error_summary.txt", "w") as f:
                f.write("\n".join(error_output))

# %%
train_cnn_with_different_configuration_save_results(
        train_dataloader=train_dataloader_reg_75, 
        val_dataloader=val_dataloader_reg_75,  
        image_size = 75, 
        classification_size=1, 
        results_file_name="regression-75",
        loss_function=CustomLoss(),
        epoch_overide=30
)

error_summary("regression", test_dataloader_reg_75, 75)

train_cnn_with_different_configuration_save_results(
        train_dataloader=train_dataloader_reg_75, 
        val_dataloader=val_dataloader_reg_75,  
        image_size = 75, 
        classification_size=1, 
        results_file_name="regression-MSELoss-75",
        loss_function=nn.MSELoss(),
        epoch_overide=30
)

error_summary("regression-MSELoss", test_dataloader_reg_75, 75)

train_cnn_with_different_configuration_save_results(
        train_dataloader=train_dataloader_reg_75, 
        val_dataloader=val_dataloader_reg_75,  
        image_size = 75, 
        classification_size=1, 
        results_file_name="regression-L1Loss-75",
        loss_function=nn.L1Loss(),
        epoch_overide=30
)

error_summary("regression-L1Loss", test_dataloader_reg_75, 75)

train_cnn_with_different_configuration_save_results(
        train_dataloader=train_dataloader_reg_75, 
        val_dataloader=val_dataloader_reg_75,  
        image_size = 75, 
        classification_size=1, 
        results_file_name="regression-SmoothL1Loss-75",
        loss_function=nn.SmoothL1Loss(),
        epoch_overide=30
)

error_summary("regression-SmoothL1Loss", test_dataloader_reg_75, 75)

# %%
train_cnn_with_different_configuration_save_results(
        train_dataloader=train_dataloader_reg_150, 
        val_dataloader=val_dataloader_reg_150,  
        image_size = 150, 
        classification_size=1, 
        results_file_name="regression-150",
        loss_function=CustomLoss(),
        epoch_overide=30
)

error_summary("regression-150", test_dataloader_reg_150, 150)

train_cnn_with_different_configuration_save_results(
        train_dataloader=train_dataloader_reg_150, 
        val_dataloader=val_dataloader_reg_150,  
        image_size = 150, 
        classification_size=1, 
        results_file_name="regression-MSELoss-150",
        loss_function=nn.MSELoss(),
        epoch_overide=30
)

error_summary("regression-MSELoss-150", test_dataloader_reg_150, 150)

train_cnn_with_different_configuration_save_results(
        train_dataloader=train_dataloader_reg_150, 
        val_dataloader=val_dataloader_reg_150,  
        image_size = 150, 
        classification_size=1, 
        results_file_name="regression-L1Loss-150",
        loss_function=nn.L1Loss(),
        epoch_overide=30
)

error_summary("regression-L1Loss-150", test_dataloader_reg_150, 150)

train_cnn_with_different_configuration_save_results(
        train_dataloader=train_dataloader_reg_150, 
        val_dataloader=val_dataloader_reg_150,  
        image_size = 150, 
        classification_size=1, 
        results_file_name="regression-SmoothL1Loss-150",
        loss_function=nn.SmoothL1Loss(),
        epoch_overide=30
)

error_summary("regression-SmoothL1Loss-150", test_dataloader_reg_150, 150)

# %%
##charts - regression

dt_regression = load_result_in_df("results/regression-150.json")
dt_L1Loss = load_result_in_df("results/regression-L1Loss-150.json")
dt_MSELoss= load_result_in_df("results/regression-MSELoss-150.json")
dt_SmoothL1Loss = load_result_in_df("results/regression-SmoothL1Loss-150.json")


fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, layout="constrained")
plot_sweep_panel(axes[0], dt_1min_150, "Clock Regression - CustomLoss")
plot_sweep_panel(axes[1], dt_5min_150, "Clock Regression - L1Loss")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7), sharex=True, layout="constrained")
plot_sweep_panel(axes2[0], dt_1min_150, "Clock Regression - MSELoss")
plot_sweep_panel(axes2[1], dt_5min_150, "Clock Regression - SmoothL1Loss")

# %%
plot_minutes_error("Custom", "models/regression-150/error_summary.txt")
plot_minutes_error("L1Loss", "models/regression-L1Loss-150/error_summary.txt")
plot_minutes_error("MSELoss", "models/regression-MSELoss/error_summary.txt")
plot_minutes_error("SmoothL1Loss", "models/regression-SmoothL1Loss-150/error_summary.txt")

# %% [markdown]
# Best regression from sweep

# %%

## train
def train_one_config_save_model_test_results_regression(best_configuration, test_data_loader, validation_dl, train_dl, image_size, per_minute_dataloader, loss_function, loss_name):

    name_best="best";

    os.makedirs(f"results/best/regression/", exist_ok=True)
    results_file_name_json_best = f"results/best/regression/{loss_name}.json"

    if os.path.exists(results_file_name_json_best) and not FORCE_RETRAIN:
        with open(results_file_name_json_best) as f:
            results_best = json.load(f)
        print(f"Loaded results from {results_file_name_json_best}")
    else:
        results_best = []

        model = train_CNN(cfg=best_configuration,
                    train_dataloader=train_dl,
                    val_dataloader=validation_dl,
                    results=results_best,
                    name=name_best,
                    image_size=image_size,
                    classification_size=1,
                    loss_fn_overide=loss_function)

        model_file_name = Path(f"models/best/regression/{loss_name}.pth")
        model_file_name.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_file_name)

        df_results_best_sweep_cnn = pd.DataFrame(results_best).drop(columns="history").sort_values("best_val_loss")
        
        df_results_best_sweep_cnn.to_csv(results_file_name_json_best.replace(".json", ".csv"), index=False)
        with open(results_file_name_json_best, "w") as f:
            json.dump(results_best, f, indent=1)


    # ##validate test data

    model_config = Path(f"models/best/regression/{loss_name}.pth")  

    error_output_best = []

    model = CNN( image_size=image_size,              
            channels=best_configuration["channels"],           
            kernel_size=best_configuration["kernel_size"],     
            activation=activations[best_configuration["activation"]], 
            dropout=best_configuration["dropout"], 
            init=inits[best_configuration["init"]],
            classification_size=1
            ).to(device)
        
    model.load_state_dict(torch.load(model_config, weights_only=True))
    avg_error, accuracy = test_error_classification(model, 1, test_data_loader, per_minute_dataloader)

    error_message_best = f"{name} - Average Test Error: {avg_error:.1f} minutes | Accuracy: {accuracy:.3f}"

    error_output_best.append(error_message_best)
    print(error_message_best)

    with open(f"{model_config.parent}/error_summary_{loss_name}.txt", "w") as f:
        f.write("\n".join(error_output_best))

# %%
best_configuration_regression = dict(
    init="xavier", activation="leaky_relu",
    optimizer="adamw", lr=1e-3,
    dropout=0.2,  
    channels= (3, 8), kernel_size= 5,
    epochs=50, patience=10,
    weight_decay=0.0
)

train_one_config_save_model_test_results_regression(best_configuration_regression, test_dataloader_reg_150, val_dataloader_reg_150, train_dataloader_reg_150, 150, test_dataloader_1min_150, CustomLoss(), "Custom Loss")
train_one_config_save_model_test_results_regression(best_configuration_regression, test_dataloader_reg_150, val_dataloader_reg_150, train_dataloader_reg_150, 150, test_dataloader_1min_150, nn.L1Loss(), "L1Loss")
train_one_config_save_model_test_results_regression(best_configuration_regression, test_dataloader_reg_150, val_dataloader_reg_150, train_dataloader_reg_150, 150, test_dataloader_1min_150, nn.MSELoss(), "MSELoss")
train_one_config_save_model_test_results_regression(best_configuration_regression, test_dataloader_reg_150, val_dataloader_reg_150, train_dataloader_reg_150, 150, test_dataloader_1min_150, nn.SmoothL1Loss(), "SmoothL1Loss")

# %% [markdown]
# DOUBLE HEAD

# %%
import copy
import torch
import torch.nn as nn


def make_activation(activation):
    """Accept a class (nn.ReLU), a factory (lambda: nn.LeakyReLU(0.1)), or an instance."""
    if isinstance(activation, nn.Module):
        return copy.deepcopy(activation)
    return activation()


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, activation, n_convs=2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            layers += [
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size,
                          padding=kernel_size // 2, bias=False),  # BN makes the bias redundant
                nn.BatchNorm2d(out_ch),
                make_activation(activation),
            ]
        layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MultiHeadCNN(nn.Module):
    def __init__(self, image_size=150, channels=(3, 32, 64, 128, 256), kernel_size=3,
                 activation=nn.ReLU, dropout=0.3, init=None,
                 n_hours=12, n_minutes=60, pool_size=4, hidden=256, convs_per_block=2):
        super().__init__()
        # image_size is kept for compatibility; adaptive pooling makes the model size-agnostic

        self.features = nn.Sequential(*[
            ConvBlock(c_in, c_out, kernel_size, activation, convs_per_block)
            for c_in, c_out in zip(channels[:-1], channels[1:])
        ])
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        flat_dim = channels[-1] * pool_size * pool_size

        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, hidden),
            nn.BatchNorm1d(hidden),
            make_activation(activation),
        )
        self.hour_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            make_activation(activation),
            nn.Linear(hidden // 2, n_hours),
        )
        self.minute_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            make_activation(activation),
            nn.Linear(hidden // 2, n_minutes),
        )

        if init is not None:
            self.apply(lambda m: self._init_weights(m, init))

    @staticmethod
    def _init_weights(m, init):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.trunk(self.pool(self.features(x)))
        return self.hour_head(z), self.minute_head(z)

# %%
def double_head_labels(y):
    y_hour = y[:, 0] % 12      # 0–11 (maps 12 to 0 if hours are 1–12)
    y_minute = y[:, 1]         # 0–59
    return y_hour, y_minute

# %%
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
dataset_double_head_75 = CustomDataset(x=clock_75_images, y=clock_75_labels, dtype_y=torch.float32, transform=transform)
dataset_double_head_150 = CustomDataset(x=clock_150_images, y=clock_150_labels, dtype_y=torch.float32, transform=transform)

len_dataset = len(clock_75_images)
# Split 10% of the training data for validation and 10% for testing
generator = torch.Generator().manual_seed(42)
train_dh_75, val_dh_75, test_dh_75 = random_split(dataset_double_head_75, [0.8, 0.1, 0.1], generator=generator)
train_dh_150, val_dh_150, test_dh_150 = random_split(dataset_double_head_150, [0.8, 0.1, 0.1], generator=generator)

train_loader_dh_75 = DataLoader(train_dh_75, batch_size=batch_size, shuffle=True)
val_loader_dh_75   = DataLoader(val_dh_75,   batch_size=batch_size)

train_loader_dh_150 = DataLoader(train_dh_150, batch_size=batch_size, shuffle=True)
val_loader_dh_150   = DataLoader(val_dh_150,   batch_size=batch_size)
test_loader_dh_150 = DataLoader(test_dh_150, batch_size=batch_size)


# %%
class MultiHeadLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        out_hour, out_minute = outputs
        y_hour, y_minute = targets
        return self.ce(out_hour, y_hour) + self.alpha * self.ce(out_minute, y_minute)

# %%
def train_epoch_db(model, dataloader, loss_fn, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for batch, (X, y) in enumerate(dataloader):
        x = X.to(device)
        y_hour = (y[:, 0] % 12).long().to(device)
        y_minute = y[:, 1].long().to(device)
        out_hour, out_minute = model(x)
        loss = loss_fn((out_hour, out_minute), (y_hour, y_minute))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()          # one step per batch for OneCycleLR
        total_loss += loss.item()

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]")

    return total_loss / num_batches

def evaluate_db(model, dataloader, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    test_loss, hour_correct, min_correct, both_correct = 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            x = X.to(device)
            y_hour = (y[:, 0] % 12).long().to(device)
            y_minute = y[:, 1].long().to(device)
            out_hour, out_minute = model(x)
            test_loss += loss_fn((out_hour, out_minute), (y_hour, y_minute)).item()
            h_ok = out_hour.argmax(1) == y_hour
            m_ok = out_minute.argmax(1) == y_minute
            hour_correct += h_ok.sum().item()
            min_correct  += m_ok.sum().item()
            both_correct += (h_ok & m_ok).sum().item()
    avg_loss = test_loss / len(dataloader)
    return avg_loss, both_correct / size, hour_correct/size, min_correct/size

# %%

from torch.optim.lr_scheduler import OneCycleLR

best_configuration = dict(
    init="xavier", activation="leaky_relu",
    optimizer="adamw", lr=1e-3,
    dropout=0.3,
    channels=(1, 32, 64, 128, 256), kernel_size=3,
    epochs=50, patience=10,
    weight_decay=1e-4,
)

torch.manual_seed(42)
epochs = best_configuration["epochs"]  # 50

model = MultiHeadCNN(
    image_size=150,
    channels=best_configuration["channels"],
    kernel_size=best_configuration["kernel_size"],
    activation=activations[best_configuration["activation"]],
    dropout=best_configuration["dropout"],
    init=inits[best_configuration["init"]],
).to(device)

loss_fn = MultiHeadLoss(alpha=1.0)
optimizer = optimizers[best_configuration["optimizer"]](
    model.parameters(), lr=best_configuration["lr"],
    weight_decay=best_configuration["weight_decay"],
)
scheduler = OneCycleLR(
    optimizer,
    max_lr=best_configuration["lr"],          # peak LR = your tuned 1e-3
    epochs=epochs,
    steps_per_epoch=len(train_loader_dh_150),
    pct_start=0.2,                            # first 20% warmup, then cosine decay
)

cnn_multihead_history = {"train_loss": [], "val_loss": [], "val_acc": [],
                         "hour_correct": [], "min_correct": [], "lr": []}
best_acc = 0.0

for epoch in range(epochs):
    train_loss = train_epoch_db(model, train_loader_dh_150, loss_fn, optimizer, scheduler)
    val_loss, val_acc, hour_acc, min_acc = evaluate_db(model, val_loader_dh_150, loss_fn)

    cnn_multihead_history["train_loss"].append(train_loss)
    cnn_multihead_history["val_loss"].append(val_loss)
    cnn_multihead_history["val_acc"].append(val_acc)
    cnn_multihead_history["hour_correct"].append(hour_acc)
    cnn_multihead_history["min_correct"].append(min_acc)
    cnn_multihead_history["lr"].append(scheduler.get_last_lr()[0])

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "results/cnn_multihead_best.pt")

    print(f"epoch {epoch+1}: lr {scheduler.get_last_lr()[0]:.2e} | train {train_loss:.4f} | "
          f"val {val_loss:.4f} | both {val_acc:.4f} | hour {hour_acc:.4f} | min {min_acc:.4f}")

with open("results/cnn_multihead_history.json", "w") as f:
    json.dump(cnn_multihead_history, f, indent=1)



# %%
def common_sense_error_minutes_db(pred_minutes, true_minutes, period=12 * 60):
    diff = (pred_minutes - true_minutes).abs() % period
    return torch.minimum(diff, period - diff)


def test_error_double_head(model, dataloader, error_fn=common_sense_error_minutes_db):
    model.eval()
    size = len(dataloader.dataset)
    total_error, both_correct, hour_correct, min_correct = 0.0, 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_hour = (y[:, 0] % 12).long().to(device)
            y_minute = y[:, 1].long().to(device)

            out_hour, out_minute = model(X)
            pred_hour = out_hour.argmax(1)
            pred_minute = out_minute.argmax(1)

            # Combine both heads into minutes since 12:00 so the error wraps around the dial
            pred_total = pred_hour * 60 + pred_minute
            true_total = y_hour * 60 + y_minute
            total_error += error_fn(pred_total, true_total).sum().item()

            h_ok = pred_hour == y_hour
            m_ok = pred_minute == y_minute
            hour_correct += h_ok.sum().item()
            min_correct += m_ok.sum().item()
            both_correct += (h_ok & m_ok).sum().item()

    avg_error = total_error / size
    return avg_error, both_correct / size, hour_correct / size, min_correct / size

# %%
############ TEST

model = model = MultiHeadCNN(
    image_size=150,
    channels=best_configuration["channels"],
    kernel_size=best_configuration["kernel_size"],
    activation=activations[best_configuration["activation"]],
    dropout=best_configuration["dropout"],
    init=inits[best_configuration["init"]],
).to(device)
        
model.load_state_dict(torch.load("results/cnn_multihead_best.pt", weights_only=True, map_location=device))
avg_error, both, hour, min = test_error_double_head(model,test_loader_dh_150)

error_message_best = f"Double Head - Average Test Error: {avg_error:.1f} minutes | both {both:.4f} | hour {hour:.4f} | min {min:.4f}"
print(error_message_best)

# %% [markdown]
# Plots double head

# %%
fig, ax = plt.subplots(figsize=(6, 3.8))

history = cnn_multihead_history
epochs = range(1, len(history["train_loss"]) + 1)

ax.plot(epochs, history["train_loss"], marker="o", markersize=3, label="Training loss")
ax.plot(epochs, history["val_loss"], marker="o", markersize=3, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MultiHeadLoss")
ax.set_title("CNN multihead with scheduler")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/baselines_cnn_mh_loss.png", dpi=150, bbox_inches="tight")
plt.show()


