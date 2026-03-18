import argparse
import importlib
import json
import os
import sys
import time
import uuid

source_folder = "/beegfs/halder/GITHUB/RESEARCH/crop-yield-forecasting-germany/src"
sys.path.append(source_folder)

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataset import CropFusionNetDataset
from loss.loss import QuantileLoss
from models.CropFusionNet.model_ablation import CropFusionNet
from utils.utils import evaluate_and_save_outputs, load_config, set_seed

set_seed(42)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="CropFusionNet Ablation Study Training"
    )

    # Identity & Logging
    parser.add_argument(
        "--crop", type=str, required=True, help="Crop name (e.g. winter_wheat)"
    )
    parser.add_argument(
        "--job_id", type=str, default=str(uuid.uuid4())[:8], help="Unique Job ID"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Where to save results"
    )
    parser.add_argument(
        "--ablation_name",
        type=str,
        default="full_model",
        help="Human-readable ablation scenario name (e.g. no_lstm, no_attention)",
    )

    # Ablation flags — each component defaults to ON (1); pass 0 to disable
    parser.add_argument(
        "--use_vsn",
        type=int,
        default=1,
        help="Enable Variable Selection Networks (0/1)",
    )
    parser.add_argument(
        "--use_temporal_conv",
        type=int,
        default=1,
        help="Enable Multi-Scale Temporal Convolution (0/1)",
    )
    parser.add_argument(
        "--use_lstm", type=int, default=1, help="Enable LSTM encoder (0/1)"
    )
    parser.add_argument(
        "--use_attention",
        type=int,
        default=1,
        help="Enable Multi-Head Self-Attention (0/1)",
    )
    parser.add_argument(
        "--use_static_enrichment",
        type=int,
        default=1,
        help="Enable Static Enrichment (0/1)",
    )
    parser.add_argument(
        "--use_pyramidal_pooling",
        type=int,
        default=1,
        help="Enable Dynamic Pyramidal Pooling (0/1)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Train function (identical to main training script)
# ---------------------------------------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    patience,
    scheduler=None,
    checkpoint_dir="checkpoints",
    exp_name="CropFusionNet_ablation",
):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_id = f"run_{exp_name}_{timestamp}"
    log_dir = os.path.join("runs", log_id)
    writer = SummaryWriter(log_dir=log_dir)

    save_folder = os.path.join(checkpoint_dir, log_id)
    os.makedirs(save_folder, exist_ok=True)

    print(f"📘 TensorBoard logs: {log_dir}")
    print(f"💾 Checkpoints: {save_folder}")

    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # --- TRAINING PHASE ---
        model.train()
        train_loss_accum = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for batch in train_pbar:
            optimizer.zero_grad()

            inputs = {
                "inputs": batch["inputs"].to(device),
                "identifier": batch["identifier"].to(device),
                "mask": batch["mask"].to(device),
                "variable_mask": (
                    batch.get("variable_mask").to(device)
                    if batch.get("variable_mask") is not None
                    else None
                ),
            }
            targets = batch["target"].to(device)

            output_dict = model(inputs)
            preds = output_dict["prediction"]

            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_accum += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = {
                    "inputs": batch["inputs"].to(device),
                    "identifier": batch["identifier"].to(device),
                    "mask": batch["mask"].to(device),
                    "variable_mask": (
                        batch.get("variable_mask").to(device)
                        if batch.get("variable_mask") is not None
                        else None
                    ),
                }
                targets = batch["target"].to(device)

                output_dict = model(inputs)
                preds = output_dict["prediction"]

                loss = criterion(preds, targets)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)

        # --- LOGGING & SCHEDULING ---
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} "
            f"| LR: {current_lr:.2e} | T: {elapsed:.1f}s"
        )

        writer.add_scalars(
            "Loss", {"Train": avg_train_loss, "Val": avg_val_loss}, epoch
        )
        writer.add_scalar("LR", current_lr, epoch)

        if scheduler:
            scheduler.step(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(save_folder, "best_model.pt"))
            print(f"✨ New best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}")
                break

    writer.close()
    return best_val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Parse Arguments
    args = parse_args()

    # 2. Load config for the specified crop
    cfg, model_config, train_config = load_config(args.crop)
    device = model_config["device"]

    # 3. Inject ablation flags into model config
    ablation_flags = {
        "use_vsn": bool(args.use_vsn),
        "use_temporal_conv": bool(args.use_temporal_conv),
        "use_lstm": bool(args.use_lstm),
        "use_attention": bool(args.use_attention),
        "use_static_enrichment": bool(args.use_static_enrichment),
        "use_pyramidal_pooling": bool(args.use_pyramidal_pooling),
    }
    model_config.update(ablation_flags)

    # Update experiment name
    train_config["exp_name"] = (
        f"ablation_{args.crop}_{args.ablation_name}_{args.job_id}"
    )

    # Pretty-print configuration
    enabled = [k for k, v in ablation_flags.items() if v]
    disabled = [k for k, v in ablation_flags.items() if not v]

    print(f"🚀 Starting Ablation Job {args.job_id}")
    print(f"   Crop:      {args.crop}")
    print(f"   Scenario:  {args.ablation_name}")
    print(f"   Enabled:   {enabled}")
    print(f"   Disabled:  {disabled}")
    print(f"   Output:    {args.output_dir}")

    # ---------------------------------------------------------
    # 4. INITIALIZE OBJECTS
    # ---------------------------------------------------------
    train_dataset = CropFusionNetDataset(cfg, mode="train", scale=True)
    val_dataset = CropFusionNetDataset(cfg, mode="val", scale=True)
    test_dataset = CropFusionNetDataset(cfg, mode="test", scale=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Initialize Model (with ablation flags already in model_config)
    model = CropFusionNet(model_config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Params:    {trainable_params:,} trainable / {total_params:,} total")

    optimizer = Adam(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config.get("weight_decay", 1e-5),
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=1e-4, min_lr=1e-6
    )

    criterion = QuantileLoss(quantiles=model_config["quantiles"]).to(device)

    # ---------------------------------------------------------
    # 5. RUN TRAINING
    # ---------------------------------------------------------
    best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=train_config.get("num_epochs", 500),
        patience=train_config.get("early_stopping_patience", 10),
        scheduler=scheduler,
        exp_name=train_config["exp_name"],
    )

    # Save the trained model
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, f"model_{args.job_id}.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Trained model saved to {model_save_path}")

    # ---------------------------------------------------------
    # 6. EVALUATE AND SAVE OUTPUTS
    # ---------------------------------------------------------
    print("🔍 Evaluating and saving outputs...")

    evaluate_and_save_outputs(
        model, train_loader, criterion, device, args.output_dir, "train"
    )
    evaluate_and_save_outputs(
        model, val_loader, criterion, device, args.output_dir, "validation"
    )
    evaluate_and_save_outputs(
        model, test_loader, criterion, device, args.output_dir, "test"
    )

    # ---------------------------------------------------------
    # 7. SAVE ABLATION RESULTS (for plotting later)
    # ---------------------------------------------------------
    results = {
        "job_id": args.job_id,
        "crop": args.crop,
        "ablation_name": args.ablation_name,
        "ablation_flags": ablation_flags,
        "model_config": {
            k: v
            for k, v in model_config.items()
            if k != "device"  # device is not JSON-serializable
        },
        "train_config": train_config,
        "model_params": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "metrics": {
            "best_val_loss": best_val_loss,
        },
    }

    save_path = os.path.join(args.output_dir, f"result_{args.job_id}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Ablation results saved to {save_path}")
