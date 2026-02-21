import torch
from pathlib import Path
import torch.optim as optim
from training.logger import WandbLogger
from model.model import MultiEncoderUNet
from training.datasets import PETSDataset
from sklearn.model_selection import KFold
from training.losses import SparseHeatmapLoss
from torch.utils.data import DataLoader, Subset
from training.utils import ConsoleColors as cc, load_params, log_predictions_to_wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    ## Config ##
    CFG    = load_params("training/config/cross_val_cfg.yaml")
    logger = WandbLogger(CFG) 

    if DEVICE == "cpu": print(cc.WARN + f"Using cpu as a device\n")
    else: print(cc.INFO + f"Using gpu as a device\n")
    
    ## Load dataset
    dataset = PETSDataset(scale=CFG['image_scale'])
    if CFG['debug']: # debug
        dataset = Subset(dataset, range(20))

    kf = KFold(n_splits=CFG['kf_n_splits'], shuffle=True, random_state=42)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    global_step = 0
    fold_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)): # pyright: ignore[reportArgumentType]
        print(cc.INFO + f"\n--- Fold {fold+1}/{CFG['kf_n_splits']} ---")

        train_ds = Subset(dataset, train_idx.tolist())
        val_ds   = Subset(dataset, val_idx.tolist())

        train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False)

        model = MultiEncoderUNet(
            past_channels = 1,
            impassable_channels = 1,
            context_channels = 3,
            zoom_channels = 3
        ).to(DEVICE)

        criterion = SparseHeatmapLoss(nonzero_weight=150.0, sparsity_weight=50.0) # best one so far
        optimizer = optim.Adam(model.parameters(), lr=float(CFG['learning_rate']))

        CV_EPOCHS = CFG['num_epochs_in_k']
        best_val_loss = float('inf')

        ## Main training loop
        for epoch in range(CV_EPOCHS):
            # ---------------- TRAIN ----------------
            model.train()
            train_loss = 0

            for batch in train_loader: 
                past, imp, ctx, zoom, target = [x.to(DEVICE) for x in batch]

                optimizer.zero_grad()

                model_out = model(past, imp, ctx, zoom)
                loss = criterion(model_out, target.float())

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)

            # ---------------- VALID ----------------
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    past, imp, ctx, zoom, target = [x.to(DEVICE) for x in batch]

                    model_out = model(past, imp, ctx, zoom)
                    loss = criterion(model_out, target.float())

                    val_loss += loss.item()
            val_loss   /= max(len(val_loader), 1)

            # ---------------- LOGGING ----------------
            fold_metrics = {
                "fold": fold + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            if epoch == CV_EPOCHS - 1:
                fold_metrics["predictions"] = log_predictions_to_wandb(
                    model, val_loader, epoch + 1, DEVICE
                )

            logger.log(fold_metrics, global_step)
            global_step += 1

            print(cc.INFO + f"Fold {fold+1} | Epoch {epoch+1}/{CV_EPOCHS}")
            print(cc.INFO + f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        fold_val_losses.append(best_val_loss)
        logger.log(
            {"fold_best_val_loss": best_val_loss, "fold": fold + 1},
            global_step
        )
        global_step += 1
        print(cc.INFO + f"Fold {fold+1} best val loss: {best_val_loss:.4f}")

    # ---------------- FINAL CV SUMMARY ----------------
    mean_loss = sum(fold_val_losses) / len(fold_val_losses)

    logger.log({
        "cv_mean_val_loss": mean_loss,
        "cv_fold_losses": fold_val_losses
    }, global_step)

    print(cc.INFO + "\nCross-validation complete")
    print(cc.INFO + f"Mean val loss: {mean_loss:.4f}")

    logger.finish()




    