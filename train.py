import torch
import torch.optim as optim
from training.losses import DiceLoss
from training.logger import WandbLogger
from torch.utils.data import DataLoader
from model.model import MultiEncoderUNet
from training.datasets import PETSDataset
from training.utils import ConsoleColors as cc, load_params, split_ds
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    ## Config ##
    CFG    = load_params("training/config/training_cfg.yaml")
    logger = WandbLogger(CFG) 

    if DEVICE == "cpu": print(cc.WARN + f"Using cpu as a device\n")
    else: print(cc.INFO + f"Using gpu as a device\n")
    
    ## Load dataset
    dataset = PETSDataset()
    if CFG['debug']: dataset = torch.utils.data.Subset(dataset, range(200)) # debug
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
    critetion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(CFG['learning_rate'])) if CFG['optimizer'] == "adam" else None

    ## Main training loop
    for epoch in range(CFG['num_epochs']):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0

        for batch in train_loader: 
            past, imp, ctx, zoom, target = [x.to(DEVICE) for x in batch]

            optimizer.zero_grad()
            model_out = model(past, imp, ctx, zoom)
            loss = critetion(model_out, target.float())

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

                model_out = model(past, imp, ctx, zoom)
                loss = critetion(model_out, target.float())

                val_loss += loss.item()
        val_loss /= len(val_loader)

        logger.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, 
            epoch+1
        )


        print(cc.INFO + f"Epoch {epoch+1}/{CFG['num_epochs']}")
        print(cc.INFO + f"Train loss: {train_loss:.4f}")
        print(cc.INFO + f"Val loss:   {val_loss:.4f}")
        print("-"*30)

    logger.finish()




    