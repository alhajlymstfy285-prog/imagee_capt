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


def train_epoch(model, optimizer, images, captions, batch_size, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(images) // batch_size
    
    perm = torch.randperm(len(images))
    images, captions = images[perm], captions[perm]
    
    for i in range(num_batches):
        batch_img = images[i*batch_size:(i+1)*batch_size].to(device)
        batch_cap = captions[i*batch_size:(i+1)*batch_size].to(device)
        
        optimizer.zero_grad()
        loss = model(batch_img, batch_cap)
        loss.backward()
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=config["training"]["lr_decay"]
    )
    
    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    
    for epoch in range(config["training"]["num_epochs"]):
        start = time.time()
        
        train_loss = train_epoch(
            model, optimizer,
            data["train_images"], data["train_captions"],
            config["training"]["batch_size"], device
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
        
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
              f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, Time: {epoch_time:.1f}s")
    
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
