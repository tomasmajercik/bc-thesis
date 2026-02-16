import torch
import torch.optim as optim
from training.losses import DiceLoss, NonZeroDiceLoss, WingLoss, FocalMSELoss, WeightedMSELoss, SparseHeatmapLoss
from training.logger import WandbLogger
from torch.utils.data import DataLoader
from model.model import MultiEncoderUNet
from training.datasets import PETSDataset
from training.utils import ConsoleColors as cc, load_params, split_ds

import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    ## Config ##
    CFG    = load_params("training/config/training_cfg.yaml")
    logger = WandbLogger(CFG) 

    if DEVICE == "cpu": print(cc.WARN + f"Using cpu as a device\n")
    else: print(cc.INFO + f"Using gpu as a device\n")
    
    ## Load dataset
    dataset = PETSDataset(scale=CFG['image_scale'])
    if CFG['debug']: dataset = torch.utils.data.Subset(dataset, range(20)) # debug
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
    
    # criterion = DiceLoss()
    # criterion = WingLoss()
    # criterion = torch.nn.MSELoss()
    # criterion = FocalMSELoss(alpha=20.0, beta=1.0, gamma=2.0)
    # criterion = WeightedMSELoss()
    # criterion = SparseHeatmapLoss(nonzero_weight=150.0, sparsity_weight=10.0) # best one so far
    criterion = NonZeroDiceLoss(smooth=1e-6, threshold=0.01)


    optimizer = optim.Adam(model.parameters(), lr=float(CFG['learning_rate'])) if CFG['optimizer'] == "adam" else None

    ## Main training loop
    for epoch in range(CFG['num_epochs']):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0

        for batch in train_loader: 
            past, imp, ctx, zoom, target = [x.to(DEVICE) for x in batch]

            optimizer.zero_grad()

            model_out = torch.sigmoid(model(past, imp, ctx, zoom))
            loss = criterion(model_out, target.float())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------- VALID ----------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                past, imp, ctx, zoom, target = [x.to(DEVICE) for x in batch]

                model_out = torch.sigmoid(model(past, imp, ctx, zoom))
                loss = criterion(model_out, target.float())

                val_loss += loss.item()
        val_loss /= len(val_loader)

        logger.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, epoch+1)


        # debug: print some predictions every 2 epochs, or first 5 epochs
        if epoch % 2 == 0 or epoch < 5: 
            model.eval()
            with torch.no_grad():
                # Get first validation batch
                val_batch = next(iter(val_loader))
                past, imp, ctx, zoom, target = [x.to(DEVICE) for x in val_batch]
                pred = torch.sigmoid(model(past, imp, ctx, zoom))
                
                print(f" Predictions at epoch {epoch+1}:")
                print(f" Pred mean: {pred.mean():.6f} (target: {target.mean():.6f})")
                print(f" Pred max: {pred.max():.3f} (target: {target.max():.3f})")
                print(f" Pred >0.01: {100*(pred > 0.01).float().mean():.2f}% (target: {100*(target > 0.01).float().mean():.2f}%)")
                print(f" Pred >0.1: {100*(pred > 0.1).float().mean():.2f}%")
                print(f" Pred >0.5: {100*(pred > 0.5).float().mean():.2f}%")
            model.train()


        print(cc.INFO + f"Epoch {epoch+1}/{CFG['num_epochs']}")
        print(cc.INFO + f"Train loss: {train_loss:.4f}")
        print(cc.INFO + f"Val loss:   {val_loss:.4f}")
        print("-"*30)

    logger.finish()




    