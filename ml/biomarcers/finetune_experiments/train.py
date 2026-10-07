"""Fine-tune exactly one folder: checkpoint 1 -> experiment 1, etc."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import torch
import pandas as pd

from .common import EXPERIMENTS, check_prepared, digest, experiment_frames, file_digest, write_csv, write_json
from .models import checkpoint_data, discover_group, load_model
from .scoring import evaluate_loader, metric_frame
from .training import (BudgetExpired, CombinedLoss, atomic_save, capture_rng, check_deadline,
                       make_loader, make_optimizer, make_scheduler, probe_amp, restore_rng, seed_everything, train_epoch)


def train_experiment(args, architecture, loss_kind, initial, experiment, run_dir, manifest, deadline):
    exp_dir = run_dir / f"experiment_{experiment}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    settings = {key: getattr(args, key) for key in ("epochs", "patience", "batch_size", "accumulation",
                "seed", "encoder_lr", "head_lr", "transunet_lr", "weight_decay", "amp", "smoke_patches")}
    signature_data = {"version": 1, "architecture": architecture, "loss": loss_kind,
                      "experiment": experiment, "initial_checkpoint": initial.name,
                      "initial_sha256": file_digest(initial), "data_fingerprint": manifest["fingerprint"],
                      "settings": settings}
    signature = digest(signature_data)
    last_path, best_path = exp_dir / "last.pth", exp_dir / "best.pth"
    saved = None
    if last_path.exists():
        if not args.resume:
            raise FileExistsError(f"{last_path} already exists. Use --resume or a different output directory.")
        saved = checkpoint_data(last_path)
        if saved.get("signature") != signature:
            raise ValueError(f"Resume configuration/checkpoint/dataset mismatch: {exp_dir}")
        if saved["completed"]:
            if not best_path.exists():
                raise FileNotFoundError(f"Completed run is missing {best_path}")
            print(f"Experiment {experiment}: already completed, skipping", flush=True)
            return
    elif best_path.exists() or (exp_dir / "history.csv").exists():
        raise ValueError(f"Incomplete output without last.pth: {exp_dir}; use a new directory")
    check_deadline(deadline)
    train, valid = experiment_frames(args.prepared_dir, experiment)
    if args.smoke_patches:
        train, valid = train.head(args.smoke_patches), valid.head(args.smoke_patches)
    device = torch.device(args.device)
    seed_everything(args.seed + experiment)
    model = load_model(last_path if saved is not None else initial, architecture).to(device)
    if saved is not None and architecture == "segformer" and list(saved["model_state_dict"]) != list(model.state_dict()):
        raise ValueError("SegFormer layout changed since this training run. Resume with the original "
                         "Transformers version: optimizer states depend on parameter order. "
                         "For a new experiment use a different output directory.")
    optimizer = make_optimizer(model, architecture, args.encoder_lr, args.head_lr, args.transunet_lr, args.weight_decay)
    train_batches = (len(train) + args.batch_size - 1) // args.batch_size
    scheduler = make_scheduler(optimizer, train_batches, args.accumulation, args.epochs)
    effective_amp = (saved["amp_enabled"] if saved is not None and "amp_enabled" in saved
                     else probe_amp(model, train, args.batch_size, device, args.amp))
    scaler = torch.amp.GradScaler("cuda", enabled=effective_amp and device.type == "cuda")
    criterion = CombinedLoss(loss_kind).to(device)
    epoch, best_dice, stale, history = 0, -1.0, 0, []
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        if effective_amp:
            scaler.load_state_dict(saved["scaler_state_dict"])
        epoch, best_dice, stale = saved["epoch"], saved["best_dice"], saved["epochs_no_improve"]
        history = saved["history"]
        restore_rng(saved["rng_state"])
        # Discard any row from an epoch whose last checkpoint was not committed.
        if history:
            write_csv(exp_dir / "history.csv", pd.DataFrame(history))
    del saved
    metadata = {**signature_data, "signature": signature, "initial_path": str(initial.resolve()),
                "protocol": EXPERIMENTS[experiment], "train_patches": len(train), "valid_patches": len(valid),
                "train_sources": train[["dataset", "source_id"]].drop_duplicates().to_dict("records"),
                "valid_sources": valid[["dataset", "source_id"]].drop_duplicates().to_dict("records"),
                "python": platform.python_version(), "torch": str(torch.__version__),
                "device": str(device), "amp_enabled": effective_amp,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    write_json(exp_dir / "config.json", metadata)

    def save_last(completed):
        atomic_save(last_path, {"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(), "rng_state": capture_rng(), "epoch": epoch,
                    "best_dice": best_dice, "epochs_no_improve": stale, "history": history,
                    "signature": signature, "completed": completed, "architecture": architecture,
                    "amp_enabled": effective_amp})

    if not last_path.exists():
        save_last(False)  # Allows a restart even if the first epoch is interrupted.
    try:
        while epoch < args.epochs and stale < args.patience:
            check_deadline(deadline)
            started = time.monotonic()
            loader = make_loader(train, args.batch_size, args.num_workers, train=True, seed=args.seed + experiment * 1000 + epoch)
            loss = train_epoch(model, loader, optimizer, scheduler, scaler, criterion, device,
                               args.accumulation, deadline=deadline, amp=effective_amp)
            valid_loader = make_loader(valid, args.batch_size, args.num_workers)
            confusion, _ = evaluate_loader(model, valid_loader, device,
                                            check_time=lambda: check_deadline(deadline), amp=False)
            dice = float(metric_frame(confusion).iloc[-1].dice)
            epoch += 1
            improved = dice > best_dice
            if improved:
                best_dice, stale = dice, 0
                atomic_save(best_path, {"model_state_dict": model.state_dict(), "epoch": epoch,
                            "val_dice": dice, "architecture": architecture, "signature": signature})
            else:
                stale += 1
            history.append({"epoch": epoch, "train_loss": loss, "val_dice": dice,
                            "best_dice": best_dice, "epochs_no_improve": stale,
                            "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - started})
            complete = epoch >= args.epochs or stale >= args.patience
            save_last(complete)
            write_csv(exp_dir / "history.csv", pd.DataFrame(history))
            write_json(exp_dir / "status.json", {"status": "completed" if complete else "training",
                       "epoch": epoch, "best_dice": best_dice,
                       "reason": "early_stopping" if stale >= args.patience else "epochs" if complete else None})
            print(f"Experiment {experiment} | epoch {epoch}/{args.epochs} | loss {loss:.6f} | "
                  f"valid Dice {dice:.6f} | best {best_dice:.6f}", flush=True)
    except BudgetExpired:
        write_json(exp_dir / "status.json", {"status": "paused_time_budget", "epoch": epoch,
                   "best_dice": best_dice, "resume": "Rerun with --resume; incomplete epoch will be repeated"})
        raise
    except (Exception, KeyboardInterrupt) as error:
        write_json(exp_dir / "status.json", {"status": "failed", "epoch": epoch,
                   "best_dice": best_dice, "error": f"{type(error).__name__}: {error}",
                   "resume": "last.pth contains the last complete epoch"})
        raise
    finally:
        del model, optimizer, scheduler, scaler, criterion
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True, help="ONE folder containing three checkpoints")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--transunet-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-hours", type=float, default=10.0, help="Training budget for this invocation; 0 disables")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-patches", type=int, default=0, help="Debug only: limit each split, use a separate *_smoke run")
    args = parser.parse_args()
    for key in ("epochs", "patience", "batch_size", "accumulation"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.smoke_patches < 0 or args.max_hours < 0:
        parser.error("workers, smoke-patches and max-hours must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Enable a GPU in Kaggle notebook settings.")
    deadline = time.monotonic() + args.max_hours * 3600 if args.max_hours else float("inf")
    architecture, loss_kind, mapping = discover_group(args.models_dir)
    manifest = check_prepared(args.prepared_dir)
    run_dir = args.output_dir / f"{architecture}_{loss_kind}{'_smoke' if args.smoke_patches else ''}"
    run_info = {"architecture": architecture, "loss": loss_kind, "group": args.models_dir.name,
                "data_fingerprint": manifest["fingerprint"], "smoke_patches": args.smoke_patches,
                "mapping": {str(n): path.name for n, path in sorted(mapping.items())}}
    run_path = run_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text(encoding="utf-8")) != run_info:
        raise ValueError(f"Output folder belongs to a different run: {run_dir}")
    write_json(run_path, run_info)
    print(f"Output: {run_dir.resolve()}", flush=True)
    for number in sorted(set(args.experiments)):
        print(f"Experiment {number} <- {mapping[number].name}", flush=True)
        try:
            train_experiment(args, architecture, loss_kind, mapping[number], number, run_dir, manifest, deadline)
        except BudgetExpired as error:
            print(f"{error}. Best completed-epoch models can now be evaluated.", flush=True)
            break


if __name__ == "__main__":
    main()
