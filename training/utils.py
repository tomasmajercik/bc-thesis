import yaml
import torch
from torch.utils.data import random_split

class ConsoleColors:
    ORANGE = "\033[33m"
    GREEN  = "\033[32m"
    RED    = "\033[31m"
    RESET  = "\033[00m"

    INFO = f"{GREEN}[INFO] {RESET}"
    WARN = f"{ORANGE}[WARN] {RESET}"
    ERR  = f"{RED}[ERR] {RESET}"

def np_2_tensor(raw_numpy, device):
    return(
        torch.from_numpy(raw_numpy)
        .permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        .unsqueeze(0)     # (1, H, W, C) - adds batch dim
        .float().to(device)
    )
    
def load_params(path="config/training_cfg.yaml"):
    with open(path, "r") as f:
        CFG = yaml.safe_load(f)

    print(ConsoleColors.INFO + "Config loaded")

    return CFG

def split_ds(train_ratio, dataset):
    train_size = int(train_ratio * len(dataset))
    val_size   = len(dataset) - train_size

    return random_split(dataset, [train_size, val_size])
