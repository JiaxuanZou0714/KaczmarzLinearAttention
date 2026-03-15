import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial 
import os
import argparse
# import wandb
import numpy as np

# Adjust path to import from local directories
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Block, MBlock, Config
from lit_gpt.utils import num_parameters
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
from mqar_data import MQARDataset

def main(args):
    # Setup Logger
    if args.debug:
        wandb_logger = WandbLogger(project="mqar_linear_attn", mode='disabled', name=args.exp_name)
    else:
        wandb_logger = WandbLogger(project="mqar_linear_attn", name=args.exp_name, save_dir=args.wandb_dir)

    # Setup Fabric
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        strategy = "auto" # Single GPU or DDP is fine for small models
        fabric = L.Fabric(devices=args.devices, strategy=strategy, precision="bf16-mixed", loggers=[wandb_logger])
    else:
        fabric = L.Fabric(devices=1, strategy="auto", precision="32-true", loggers=[wandb_logger])
    
    fabric.launch()
    fabric.seed_everything(args.seed)

    if fabric.global_rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        fabric.print(args)

    # Load Data
    train_dataset = MQARDataset(data_path=os.path.join(args.data_dir, "train.npz"))
    val_dataset = MQARDataset(data_path=os.path.join(args.data_dir, "val.npz"))
    test_dataset = MQARDataset(data_path=os.path.join(args.data_dir, "test.npz"))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    train_dataloader, val_dataloader, test_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader, test_dataloader)

    # Load Model
    config = Config.from_name(args.model_name)
    
    # Override config params from args if provided
    if args.n_embd: config.n_embd = args.n_embd
    if args.n_head: config.n_head = args.n_head
    if args.n_layer: config.n_layer = args.n_layer
    
    # Ensure head_size consistency
    if config.n_embd % config.n_head != 0:
        config.n_head = config.n_embd // 64 # Default head size 64
        
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))
        
    model = fabric.setup(model)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    optimizer = fabric.setup_optimizers(optimizer)

    # Training Loop
    state = {
        "model": model, 
        "optimizer": optimizer, 
        "iter_num": 0, 
        "best_val_acc": 0.0,
        "no_improve_count": 0
    }
    
    loss_func = FusedCrossEntropyLoss()
    
    best_model_path = train(fabric, state, train_dataloader, val_dataloader, loss_func, args)
    
    # Test with best model
    if best_model_path and os.path.exists(best_model_path):
        fabric.print(f"Loading best model from {best_model_path} for testing...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model"])
        
        test_acc = validate(fabric, model, test_dataloader)
        fabric.print(f"Test Acc: {test_acc:.4f}")
        fabric.log_dict({"test/acc": test_acc})
        
        # Save results to JSON
        import json
        results = {
            "model_name": args.model_name,
            "test_acc": float(test_acc),
            "best_val_acc": float(state["best_val_acc"]),
            "args": vars(args)
        }
        if fabric.global_rank == 0:
            with open(os.path.join(args.out_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
    else:
        fabric.print("No best model found or training failed.")

def train(fabric, state, train_dataloader, val_dataloader, loss_func, args):
    model = state["model"]
    optimizer = state["optimizer"]
    
    max_steps = args.max_steps
    val_interval = args.val_interval
    save_interval = args.save_interval
    
    step = 0
    train_iter = iter(train_dataloader)
    
    best_checkpoints = [] # Keep track of (acc, path)
    best_model_path = None
    
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
            
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        logits = model(input_ids)
        
        # Flatten for loss
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        loss = loss_func(logits_flat, labels_flat)
        
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Compute accuracy on this batch (only on masked tokens)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            correct = (preds == labels) & mask
            acc = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
        
        state["iter_num"] += 1
        step += 1
        
        # Logging
        if step % args.log_interval == 0:
            fabric.log_dict({
                "train/loss": loss.item(),
                "train/acc": acc.item(),
                "train/lr": optimizer.param_groups[0]['lr']
            }, step=step)
            if fabric.global_rank == 0:
                print(f"Step {step}: Loss {loss.item():.4f}, Acc {acc.item():.4f}")

        # Validation
        if step % val_interval == 0:
            val_acc = validate(fabric, model, val_dataloader)
            fabric.log_dict({"val/acc": val_acc}, step=step)
            if fabric.global_rank == 0:
                print(f"Step {step}: Val Acc {val_acc:.4f}")
            
            # Early stopping check
            if val_acc > state["best_val_acc"]:
                state["best_val_acc"] = val_acc
                state["no_improve_count"] = 0
                
                # Save best model
                save_path = os.path.join(args.out_dir, f"best_model_step_{step}.pth")
                fabric.save(save_path, state)
                best_model_path = save_path
                
                # Maintain top 5
                best_checkpoints.append((val_acc, save_path))
                best_checkpoints.sort(key=lambda x: x[0], reverse=True)
                if len(best_checkpoints) > 5:
                    to_remove = best_checkpoints.pop()
                    if os.path.exists(to_remove[1]):
                        os.remove(to_remove[1])
            else:
                state["no_improve_count"] += 1
                if state["no_improve_count"] >= args.early_stop_patience:
                    if fabric.global_rank == 0:
                        print(f"Early stopping at step {step}")
                    break
        
        # Checkpoint
        if step % save_interval == 0:
            save_path = os.path.join(args.out_dir, f"ckpt_step_{step}.pth")
            fabric.save(save_path, state)
            
    return best_model_path

@torch.no_grad()
def validate(fabric, model, dataloader):
    model.eval()
    total_correct = 0
    total_masked = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        logits = model(input_ids)
        preds = torch.argmax(logits, dim=-1)
        
        mask = labels != -100
        correct = (preds == labels) & mask
        
        total_correct += correct.sum().item()
        total_masked += mask.sum().item()
        
    model.train()
    
    # Sync across devices
    total_correct = fabric.all_reduce(total_correct, reduce_op="sum")
    total_masked = fabric.all_reduce(total_masked, reduce_op="sum")
    
    return total_correct / total_masked if total_masked > 0 else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--wandb_dir", type=str, default="./wandb")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="mqar_exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=200) 
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    
    # Model overrides
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)

    args = parser.parse_args()
    main(args)
