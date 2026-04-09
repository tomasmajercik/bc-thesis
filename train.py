import torch
from pathlib import Path
import torch.optim as optim
from training.logger import WandbLogger
from torch.utils.data import DataLoader
from model.model import MultiEncoderUNet
from training.metrics import EMDMetric, KLDMetric, FDEMetric, MRMetric
from training.datasets import PETSDataset, StMarcDataset, SherbrookeDataset, AtriumDataset, RouenDataset, MOTS16_02Dataset
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
    use_lstm = CFG.get('use_lstm', False)

    if CFG['dataset'] == "pets":
        dataset = PETSDataset(scale=CFG['image_scale'], return_coords=CFG['return_coords'], return_past_coords=use_lstm)
    elif CFG['dataset'] == "stmarc":
        dataset = StMarcDataset(scale=CFG['image_scale'], return_coords=CFG['return_coords'], return_past_coords=use_lstm)
    elif CFG['dataset'] == "sherbrooke":
        dataset = SherbrookeDataset(scale=CFG['image_scale'], return_coords=CFG['return_coords'], return_past_coords=use_lstm)
    elif CFG['dataset'] == "atrium":
        dataset = AtriumDataset(scale=CFG['image_scale'], return_coords=CFG['return_coords'], return_past_coords=use_lstm)
    elif CFG['dataset'] == "rouen":
        dataset = RouenDataset(scale=CFG['image_scale'], return_coords=CFG['return_coords'], return_past_coords=use_lstm)
    elif CFG['dataset'] == "mots16_02":
        dataset = MOTS16_02Dataset(scale=(CFG['image_scale'] - 0.15), return_coords=CFG['return_coords'], return_past_coords=use_lstm)

    if CFG['debug']: dataset = torch.utils.data.Subset(dataset, range(20))

    train_ds, val_ds, test_ds = split_ds_sequential(dataset, CFG['train_ratio'], CFG['val_ratio'])

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=CFG['batch_size'], shuffle=False)

    ## Load model
    model = MultiEncoderUNet(
        past_channels     = 1,
        obstacle_channels = 1,
        context_channels  = 3,
        zoom_channels     = 3,
        width             = CFG['model_size'],
        use_lstm          = use_lstm,
    ).to(DEVICE)


    # from training.losses import EdgeLoss
    # criterion = EdgeLoss().to(DEVICE)
    from training.losses import SparseHeatmapLoss
    criterion = SparseHeatmapLoss().to(DEVICE)

    # fourier = FourierLoss().to(DEVICE)
    # edge    = EdgeLoss().to(DEVICE)
    # sparse   = SparseHeatmapLoss().to(DEVICE)
    # emd     = EMDLoss().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=float(CFG['learning_rate']), weight_decay=float(CFG['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    ## Metrics
    emd_metric = EMDMetric()
    kld_metric = KLDMetric()
    fde_metric = FDEMetric()
    mr_metric  = MRMetric()

    ## Fixed viz batch — grab the middle batch of val so it's always the same samples
    all_val_batches  = list(val_loader)
    fixed_viz_batch  = all_val_batches[len(all_val_batches) // 2]

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
            if use_lstm:
                past, imp, ctx, zoom, target, _, past_coords = [x.to(DEVICE) for x in batch]
                model_out = model(past, imp, ctx, torch.zeros_like(zoom), past_coords)
            else:
                past, imp, ctx, zoom, target, _ = [x.to(DEVICE) for x in batch]
                model_out = model(past, imp, ctx, torch.zeros_like(zoom))

            optimizer.zero_grad()
            loss = criterion(model_out, target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------- VALID ----------------
        model.eval()
        val_loss = 0
        val_emd = val_kld = val_fde = val_mr = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if use_lstm:
                    past, imp, ctx, zoom, target, coords, past_coords = [x.to(DEVICE) for x in batch]
                    model_out = model(past, imp, ctx, torch.zeros_like(zoom), past_coords)
                else:
                    past, imp, ctx, zoom, target, coords = [x.to(DEVICE) for x in batch]
                    model_out = model(past, imp, ctx, torch.zeros_like(zoom))
                loss = criterion(model_out, target.float())

                val_loss += loss.item()
                val_emd  += emd_metric(model_out, target).item()
                val_kld  += kld_metric(model_out, target).item()
                val_fde  += fde_metric(model_out, target, coords).item()
                val_mr   += mr_metric(model_out,  target, coords).item()

                # edge_loss = edge(model_out, target.float()).item()
                # fourier_loss = fourier(model_out, target.float()).item()
                # sparse_loss = sparse(model_out, target.float()).item()
                # emd_loss = emd(model_out, target.float()).item()

        n = len(val_loader)
        val_loss /= n
        val_emd  /= n
        val_kld  /= n
        val_fde  /= n
        val_mr   /= n

        # print(cc.INFO + f"Component losses - Fourier: {fourier_loss:.4f}  Edge: {edge_loss:.4f}  Sparse: {sparse_loss:.4f}  EMD: {emd_loss:.4f}")



        # print(f"sparse: {val_loss:.4f}  EMD: {val_emd:.4f}")

        scheduler.step(val_loss)

        # ---------------- LOGGING ----------------
        logger.log({
            "train_loss":  train_loss,
            "val_loss":    val_loss,
            "val_emd":     val_emd,
            "val_kld":     val_kld,
            "val_fde":     val_fde,
            "val_mr":      val_mr,
            "predictions": log_predictions_to_wandb(model, val_loader, epoch + 1, DEVICE, num_samples=3, fixed_batch=fixed_viz_batch),
            "lr":          optimizer.param_groups[0]['lr'],
        }, epoch + 1)

        # ---------------- SAVE BEST MODEL ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"{CFG['wandb']['run_name']}" / "best_model.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss':           train_loss,
                'val_loss':             val_loss,
                'config':               CFG,
            }, checkpoint_path)

            print(cc.INFO + f"New best model saved. Val loss: {val_loss:.4f} (epoch {epoch+1})")

        print(cc.INFO + f"Epoch {epoch+1}/{CFG['num_epochs']}")
        print(cc.INFO + f"Train loss: {train_loss:.4f}")
        print(cc.INFO + f"Val loss:   {val_loss:.4f}  |  EMD: {val_emd:.4f}  KLD: {val_kld:.4f}  FDE: {val_fde:.2f}px  MR: {val_mr:.3f}")
        print("-" * 30)

    print(cc.INFO + "Running final test evaluation...")

    # ---------------- FINAL TEST EVALUATION ----------------
    best_checkpoint_path = checkpoint_dir / f"{CFG['wandb']['run_name']}" / "best_model.pth"
    checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    test_loss = test_emd = test_kld = test_fde = test_mr = 0.0  # fix: initialize before loop

    with torch.no_grad():
        for batch in test_loader:
            if use_lstm:
                past, imp, ctx, zoom, target, coords, past_coords = [x.to(DEVICE) for x in batch]
                model_out = model(past, imp, ctx, torch.zeros_like(zoom), past_coords)
            else:
                past, imp, ctx, zoom, target, coords = [x.to(DEVICE) for x in batch]
                model_out = model(past, imp, ctx, torch.zeros_like(zoom))
            loss = criterion(model_out, target.float())

            test_loss += loss.item()
            test_emd  += emd_metric(model_out, target).item()
            test_kld  += kld_metric(model_out, target).item()
            test_fde  += fde_metric(model_out, target, coords).item()
            test_mr   += mr_metric(model_out,  target, coords).item()

    n = len(test_loader)
    test_loss /= n
    test_emd  /= n
    test_kld  /= n
    test_fde  /= n
    test_mr   /= n

    print(cc.INFO + f"Final Test loss: {test_loss:.4f}  |  EMD: {test_emd:.4f}  KLD: {test_kld:.4f}  FDE: {test_fde:.2f}px  MR: {test_mr:.3f}")

    logger.log({
        "test_loss": test_loss,
        "test_emd":  test_emd,
        "test_kld":  test_kld,
        "test_fde":  test_fde,
        "test_mr":   test_mr,
    }, CFG['num_epochs'] + 1)

    logger.finish()