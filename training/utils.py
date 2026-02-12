import torch

class ConsoleColors:
    def __init__(self):
        self.ORANGE = "\033[33m"
        self.GREEN  = "\033[32m"
        self.RED    = "\033[31m"
        self.RESET  = "\033[00m"

        self.INFO = f"{self.GREEN}[INFO]{self.RESET}"
        self.WARN = f"{self.ORANGE}[WARN]{self.RESET}"
        self.ERR  = f"{self.RED}[ERR]{self.RESET}"

def np_2_tensor(raw_numpy, device):
    return(
        torch.from_numpy(raw_numpy)
        .permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        .unsqueeze(0)     # (1, H, W, C) - adds batch dim
        .float().to(device)
    )
    