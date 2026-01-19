"""
Training script for Vanilla RNN image captioning model.
Part of ablation study comparing: Vanilla RNN, LSTM, Attention LSTM, Transformer.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import json
import time
import torch
import matplotlib.pyplot as plt

from a5_helper import load_coco_captions, decode_captions
from models.Vanilla_RNN import VanillaRNNCaptioner


def load_dataset(dataset_path="./datasets/coco.pt"):
    """Load dataset (COCO or Flickr)."""
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        return load_coco_captions(dataset_path)
    elif os.path.exists("./datasets/flickr.pt"):
        print("Loading Flickr dataset...")
        return load_coco_captions("./datasets/flickr.pt")
    else:
        raise FileNotFoundError(
            "No dataset found! Run kaggle_setup.py first to prepare Flickr data."
        )


def load_config(config_path: str = "configs/Vanilla_RNN.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(model, optimizer, images, captions, batch_size, device, gradient_clip=None):
    """Train for one epoch with data augmentation."""
    model.train()
    total_loss = 0.0
    num_batches = len(images) // batch_size
    
    # Shuffle data
    perm = torch.randperm(len(images))
    images, captions = images[perm], captions[perm]
    
    for i in range(num_batches):
        batch_img = images[i*batch_size:(i+1)*batch_size].to(device)
        batch_cap = captions[i*batch_size:(i+1)*batch_size].to(device)
        
        # Simple data augmentation: random horizontal flip
        if torch.rand(1).item() > 0.5:
            batch_img = torch.flip(batch_img, dims=[3])  # flip horizontally
        
        optimizer.zero_grad()
        loss = model(batch_img, batch_cap)
        loss.backward()
        
        # Gradient clipping
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / num_batches


def evaluate(model, images, captions, batch_size, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = len(images) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_img = images[i*batch_size:(i+1)*batch_size].to(device)
            batch_cap = captions[i*batch_size:(i+1)*batch_size].to(device)
            loss = model(batch_img, batch_cap)
            total_loss += loss.item()
    
    return total_loss / max(num_batches, 1)


def train(config, data, device):
    """Train Vanilla RNN model."""
    word_to_idx = data["vocab"]["token_to_idx"]
    
    model = VanillaRNNCaptioner(
        word_to_idx=word_to_idx,
        wordvec_dim=config["model"]["wordvec_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        image_encoder_pretrained=config["model"]["image_encoder_pretrained"],
        ignore_index=word_to_idx.get("<NULL>"),
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    optimizer = torch.optim.AdamW(  # استخدم AdamW بدل Adam
        model.parameters(), 
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler بدل StepLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config["training"].get("early_stopping_patience", 10)
    gradient_clip = config["training"].get("gradient_clip", None)
    
    for epoch in range(config["training"]["num_epochs"]):
        start = time.time()
        
        train_loss = train_epoch(
            model, optimizer,
            data["train_images"], data["train_captions"],
            config["training"]["batch_size"], device,
            gradient_clip=gradient_clip
        )
        val_loss = evaluate(
            model,
            data["val_images"], data["val_captions"],
            config["training"]["batch_size"], device
        )
        
        epoch_time = time.time() - start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_times"].append(epoch_time)
        
        scheduler.step()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            status = "✓ Best"
        else:
            patience_counter += 1
            status = f"({patience_counter}/{patience})"
        
        # Print with learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
              f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
              f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s {status}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
            # Restore best model
            model.load_state_dict(best_model_state)
            break
    
    return model, history


def save_results(config, history, model):
    """Save training results."""
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    # Save history
    results = {
        "model": "Vanilla_RNN",
        "num_params": model.count_parameters(),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": min(history["val_loss"]),
        "total_time": sum(history["epoch_times"]),
        "train_loss_history": history["train_loss"],
        "val_loss_history": history["val_loss"],
    }
    
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save model
    if config["output"]["save_model"]:
        torch.save(model.state_dict(), os.path.join(results_dir, "model.pt"))
    
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Vanilla RNN Training")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(["Train", "Val"], [history["train_loss"][-1], history["val_loss"][-1]])
    plt.ylabel("Final Loss")
    plt.title("Final Losses")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
    plt.close()
    
    print(f"\nResults saved to {results_dir}/")


def main():
    print("="*50)
    print("Training Vanilla RNN for Ablation Study")
    print("="*50)
    
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\nLoading dataset...")
    data = load_dataset()
    
    print("\nTraining...")
    model, history = train(config, data, device)
    
    save_results(config, history, model)
    
    print("\n" + "="*50)
    print("Vanilla RNN Training Complete!")
    print(f"Best Val Loss: {min(history['val_loss']):.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
