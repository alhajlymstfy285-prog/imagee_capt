"""
LSTM Training Script with GloVe Support
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rnn_lstm_captioning import CaptioningRNN
from a5_helper import load_coco_captions, train_captioner


def load_config(config_path='configs/LSTM.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Load configuration
    config = load_config()
    
    print("=" * 60)
    print("LSTM Image Captioning Training")
    print("=" * 60)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading COCO dataset...")
    data = load_coco_captions(max_train=None)
    
    # Extract vocabulary
    word_to_idx = data['word_to_idx']
    vocab_size = len(word_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # GloVe configuration
    glove_path = None
    freeze_embeddings = False
    
    if config['embeddings']['use_glove']:
        glove_path = config['embeddings']['glove_path']
        freeze_embeddings = config['embeddings']['freeze']
        
        if os.path.exists(glove_path):
            print(f"\nUsing GloVe embeddings from: {glove_path}")
            print(f"Embeddings frozen: {freeze_embeddings}")
        else:
            print(f"\nWarning: GloVe file not found at {glove_path}")
            print("Falling back to random initialization")
            glove_path = None
    
    # Create model
    print("\nInitializing LSTM model...")
    model = CaptioningRNN(
        word_to_idx=word_to_idx,
        wordvec_dim=config['model']['wordvec_dim'],
        hidden_dim=config['model']['hidden_dim'],
        cell_type=config['model']['cell_type'],
        image_encoder_pretrained=True,
        ignore_index=word_to_idx['<NULL>'],
        glove_path=glove_path,
        freeze_embeddings=freeze_embeddings,
        backbone=config['model'].get('backbone', 'resnet50')
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['scheduler']['step_size'],
        gamma=config['scheduler']['gamma']
    )
    
    # Training configuration
    train_config = {
        'num_epochs': config['training']['num_epochs'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'print_every': 100,
        'verbose': True,
        'device': device
    }
    
    # Create results directory
    results_dir = f"results/{config['model']['cell_type'].upper()}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Epochs: {train_config['num_epochs']}")
    print(f"Batch size: {train_config['batch_size']}")
    print(f"Learning rate: {train_config['learning_rate']}")
    print(f"Results directory: {results_dir}")
    print("=" * 60 + "\n")
    
    # Train model
    train_captioner(
        model=model,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        **train_config
    )
    
    # Save final model
    model_path = os.path.join(results_dir, 'model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'word_to_idx': word_to_idx
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
