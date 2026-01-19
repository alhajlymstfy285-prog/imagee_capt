"""
Training script for LSTM model
part of ablation study comparing LSTM and Vanilla RNN and lstm with attention and Transformers
"""
import os
import sys
from time import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.LSTM import LSTMCaptioner
from a5_helper import decode_captions
from metrics import evaluate_captions
from flickr_dataset import create_flickr_dataloaders
from training.Vanilla_RNN import evaluate_metrics_dataloader

def load_dataset_efficient(config=None):
    """Load dataset using DataLoader (memory efficient)."""
    images_path = "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images"
    captions_file = "/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv"
    
    # Alternative paths
    if not os.path.exists(images_path):
        images_path = "/kaggle/input/flickr-image-dataset/flickr30k_images"
    
    # Get sample settings from config
    use_sample = False
    sample_size = None
    if config and "data" in config:
        use_sample = config["data"].get("use_sample", False)
        sample_size = config["data"].get("sample_size", 1000)
    
    if use_sample:
        print(f"Using SAMPLE mode: {sample_size} images only (for testing)")
    else:
        print("Using FULL dataset: ~30,000 images")
    
    print("Creating DataLoaders (memory efficient)...")
    train_loader, val_loader, vocab = create_flickr_dataloaders(
        images_path,
        captions_file,
        batch_size=32,
        num_workers=2,
        max_samples=sample_size if use_sample else None
    )
    
    return train_loader, val_loader, vocab

def load_config(config_path: str = "configs/lstm_config.yaml") -> dict:
    """Load configuration from a YAML file."""
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

def evaluate_metrics(model, images, captions, idx_to_word, device, num_samples=100):
    """
    Evaluate using BLEU, METEOR, CIDEr metrics.
    
    Args:
        model: trained model
        images: validation images
        captions: ground truth captions
        idx_to_word: vocabulary mapping
        device: cuda or cpu
        num_samples: number of samples to evaluate (for speed)
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    # Sample subset for faster evaluation
    num_samples = min(num_samples, len(images))
    sample_indices = torch.randperm(len(images))[:num_samples]
    
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for idx in sample_indices:
            img = images[idx:idx+1].to(device)
            gt_caption = captions[idx]
            
            # Generate caption
            generated = model.sample(img, max_length=20)
            
            # Decode - convert tensors to ints
            def decode_caption(caption_tensor, vocab):
                """Decode caption tensor to text"""
                words = []
                for token_id in caption_tensor:
                    # Convert tensor to int
                    if isinstance(token_id, torch.Tensor):
                        token_id = token_id.item()
                    
                    if token_id in vocab:
                        word = vocab[token_id]
                        if word not in ['<NULL>', '<START>', '<END>']:
                            words.append(word)
                        if word == '<END>':
                            break
                return ' '.join(words)
            
            gt_text = decode_caption(gt_caption, idx_to_word)
            gen_text = decode_caption(generated[0], idx_to_word)
            
            references.append([gt_text])
            hypotheses.append(gen_text)
    
    # Compute metrics
    metrics = evaluate_captions(references, hypotheses)
    
    return metrics

def train_with_dataloader(config, train_loader, val_loader, vocab, device):
    """Train using DataLoader (memory efficient)."""
    word_to_idx = vocab["token_to_idx"]
    
    # Get GloVe settings
    glove_path = None
    freeze_embeddings = False
    if config.get("embeddings", {}).get("use_glove", False):
        glove_path = config["embeddings"].get("glove_path")
        freeze_embeddings = config["embeddings"].get("freeze", False)
        
        if glove_path and os.path.exists(glove_path):
            print(f"Using GloVe embeddings from: {glove_path}")
            print(f"Embeddings frozen: {freeze_embeddings}")
        else:
            print(f"Warning: GloVe file not found, using random initialization")
            glove_path = None
    
    model = LSTMCaptioner(
        word_to_idx=word_to_idx,
        wordvec_dim=config["model"]["wordvec_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        image_encoder_pretrained=config["model"].get("image_encoder_pretrained", True),
        ignore_index=word_to_idx.get("<NULL>"),
        backbone=config["model"].get("backbone", "resnet50"),
        glove_path=glove_path,
        freeze_embeddings=freeze_embeddings,
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config["training"].get("early_stopping_patience", 10)
    gradient_clip = config["training"].get("gradient_clip", None)
    best_metrics = None
    
    for epoch in range(config["training"]["num_epochs"]):
        start = time()
        
        # Train
        model.train()
        train_loss = 0.0
        for batch_images, batch_captions in train_loader:
            batch_images = batch_images.to(device)
            batch_captions = batch_captions.to(device)
            
            optimizer.zero_grad()
            loss = model(batch_images, batch_captions)
            loss.backward()
            
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_images, batch_captions in val_loader:
                batch_images = batch_images.to(device)
                batch_captions = batch_captions.to(device)
                loss = model(batch_images, batch_captions)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        epoch_time = time() - start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_times"].append(epoch_time)
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            status = "✓ Best"
        else:
            patience_counter += 1
            status = f"({patience_counter}/{patience})"
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
              f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
              f"LR: {current_lr:.6f}, Time: {epoch_time:.1f}s {status}")
        
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
            model.load_state_dict(best_model_state)
            break
    
    # Compute final metrics
    print("\nComputing final metrics...")
    idx_to_word = vocab["idx_to_token"]
    best_metrics = evaluate_metrics_dataloader(
        model, val_loader, idx_to_word, device, num_samples=200
    )
    
    history["metrics"] = best_metrics
    
    return model, history


def save_results(config, history, model):
    """Save training results."""
    import json
    import matplotlib.pyplot as plt
    
    results_dir = config.get("output", {}).get("results_dir", "results/LSTM")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save history
    results = {
        "model": "LSTM",
        "num_params": model.count_parameters(),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": min(history["val_loss"]),
        "total_time": sum(history["epoch_times"]),
        "train_loss_history": history["train_loss"],
        "val_loss_history": history["val_loss"],
        "metrics": history.get("metrics", {}),
    }
    
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save model
    if config.get("output", {}).get("save_model", True):
        torch.save(model.state_dict(), os.path.join(results_dir, "model.pt"))
    
    # Plot training curves
    if config.get("output", {}).get("save_plots", True):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("LSTM Training")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(["Train", "Val"], [history["train_loss"][-1], history["val_loss"][-1]])
        plt.ylabel("Final Loss")
        plt.title("Final Losses")
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "training_curves.png"), dpi=150)
        plt.close()
    
    print(f"\n✅ Results saved to {results_dir}/")


def main():
    print("="*50)
    print("Training LSTM for Image Captioning")
    print("="*50)
    
    config = load_config("configs/LSTM.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\nLoading dataset (memory efficient)...")
    train_loader, val_loader, vocab = load_dataset_efficient(config)
    
    print("\nTraining...")
    model, history = train_with_dataloader(config, train_loader, val_loader, vocab, device)
    
    save_results(config, history, model)
    
    print("\n" + "="*50)
    print("LSTM Training Complete!")
    print(f"Best Val Loss: {min(history['val_loss']):.4f}")
    if "metrics" in history and history["metrics"]:
        print("\nFinal Metrics:")
        for name, value in history["metrics"].items():
            print(f"  {name}: {value:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
