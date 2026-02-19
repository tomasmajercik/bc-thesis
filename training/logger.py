import wandb
from training.utils import ConsoleColors as cc

class WandbLogger:
    def __init__(self, cfg):
        wandb_cfg = cfg.get("wandb", {})
        self.enabled = wandb_cfg.get("use_wandb", False)

        if not self.enabled:
            self.run = None
            print(cc.WARN + "Experiment not tracked by wandb")
            return
        
        self.run = wandb.init(
            project = wandb_cfg.get("project", "bc-thesis"),
            name    = wandb_cfg.get("run_name"),
            entity  = "xmajercik-fiit-stu",
            config  = cfg
        )
        self.log_interval = wandb_cfg.get("log_interval", 1)
        
        print(cc.INFO + "Wandb initialized")

    def log(self, data: dict, epoch: int):
        if not self.enabled:
            return
        
        if epoch % self.log_interval == 0:
            wandb.log(data)
    
    def finish(self):
        if self.enabled and self.run is not None:
            wandb.finish()
