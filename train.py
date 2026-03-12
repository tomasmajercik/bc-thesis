import torch
from pathlib import Path
import torch.optim as optim
from training.logger import WandbLogger
from torch.utils.data import DataLoader
from model.model import MultiEncoderUNet
from training.datasets import PETSDataset
from training.losses import DiceLoss, NonZeroDiceLoss, SparseIoULoss, SparseHeatmapLoss
from training.utils import ConsoleColors as cc, load_params, split_ds, split_ds_w_test, split_ds_sequential, log_predictions_to_wandb

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
    
    # train_ds, val_ds = split_ds(CFG['train_ratio'], dataset)
    # train_ds, val_ds, test_ds = split_ds_w_test(CFG['train_ratio'], dataset, 0.1)
    train_ds, val_ds, test_ds = split_ds_sequential(dataset, CFG['train_ratio'], CFG['val_ratio'])

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CFG['batch_size'], shuffle=False)
    # train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)
    # val_loader   = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False)
    # test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False)

    ## Load model
    model = MultiEncoderUNet(
        past_channels = 1,
        obstacle_channels = 1,
        context_channels = 3,
        zoom_channels = 3
    ).to(DEVICE)
    
    # criterion = DiceLoss()
    # criterion = WingLoss()
    # criterion = torch.nn.MSELoss()
    # criterion = NonZeroDiceLoss(smooth=1e-6, threshold=0.01)
    # criterion = SparseIoULoss()

    criterion = SparseHeatmapLoss(CFG['nonzero_weight'], CFG['sparsity_weight'])
    optimizer = optim.Adam(model.parameters(), lr=float(CFG['learning_rate']), weight_decay=float(CFG['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',        # minimize val loss
        factor=0.5,        # halve the LR
        patience=3,        # wait 3 epochs of no improvement
        min_lr=1e-6
    )

    ## Setup for saving best model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_val_loss = float('inf')


    ## Main training loop
    for epoch in range(CFG['num_epochs']):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0

        for batch in train_loader: 
            past, imp, ctx, zoom, target = [x.to(DEVICE) for x in batch]

            optimizer.zero_grad()

            model_out = model(past, imp, ctx, zoom)
            loss = criterion(model_out, target.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                loss = criterion(model_out, target.float())

                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # ---------------- LOGGING ----------------
        logger.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "predictions": log_predictions_to_wandb(model, val_loader, epoch + 1, DEVICE, num_samples=3),
            'lr': optimizer.param_groups[0]['lr']
        }, epoch+1)

        # ---------------- SAVE BEST MODEL ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"{CFG['wandb']['run_name']}" / "best_model.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': CFG,
            }, checkpoint_path)
            
            print(cc.INFO + f"New best model saved. Val loss: {val_loss:.4f} (epoch {epoch+1})")


        # debug: print some predictions every 2 epochs, or first 5 epochs
        if epoch % 2 == 0 or epoch < 5: 
            model.eval()
            with torch.no_grad():
                # Get first validation batch
                val_batch = next(iter(val_loader))
                past, imp, ctx, zoom, target = [x.to(DEVICE) for x in val_batch]
                pred = model(past, imp, ctx, zoom)
                
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

    print(cc.INFO + "Running final test evaluation...")


    # ---------------- FINAL TEST EVALUATION ----------------
    best_checkpoint_path = checkpoint_dir / f"{CFG['wandb']['run_name']}" / "best_model.pth"
    checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            past, imp, ctx, zoom, target = [x.to(DEVICE) for x in batch]

            model_out = model(past, imp, ctx, zoom)
            loss = criterion(model_out, target.float())

            test_loss += loss.item()

    test_loss /= len(test_loader)

    print(cc.INFO + f"Final Test loss: {test_loss:.4f}")

    logger.log({
        "test_loss": test_loss,
    }, CFG['num_epochs'] + 1)

    logger.finish()




    