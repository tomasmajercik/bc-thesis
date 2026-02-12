import torch
from model.model import MultiEncoderUNet
from torch.utils.data import DataLoader
from training.datasets import PETSDataset
from training.utils import ConsoleColors as cc, load_params, split_ds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    ## Config ##
    CFG = load_params("training/config/training_cfg.yaml")
    
    ## Load dataset
    dataset = PETSDataset()
    train_ds, val_ds = split_ds(CFG['train_ratio'], dataset)
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False)

    ## Load model
    model = MultiEncoderUNet(
        past_channels = 1,
        impassable_channels = 1,
        context_channels = 3,
        zoom_channels = 3
    ).to(DEVICE)

    for batch in train_loader: 
        past, imp, ctx, zoom, target = [x.to(DEVICE) for x in batch]

        with torch.no_grad():
            model_out = model(past, imp, ctx, zoom)

        print(cc.INFO + f"Output shape: {model_out.shape}")
        print(cc.INFO + f"Output shape: {target.shape}")
        break




    