"""
PyTorch Dataset for Flickr30k - loads images on-the-fly to save memory
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd


class FlickrDataset(Dataset):
    """
    Flickr30k Dataset that loads images on-the-fly.
    Memory efficient - doesn't load all images at once.
    """
    
    def __init__(
        self,
        images_path,
        captions_file,
        vocab,
        max_length=32,
        transform=None,
        split='train',
        train_ratio=0.8
    ):
        """
        Args:
            images_path: Path to images folder
            captions_file: Path to captions CSV
            vocab: Dictionary with 'token_to_idx' and 'idx_to_token'
            max_length: Maximum caption length
            transform: Image transforms
            split: 'train' or 'val'
            train_ratio: Ratio for train/val split
        """
        self.images_path = images_path
        self.vocab = vocab
        self.token_to_idx = vocab['token_to_idx']
        self.max_length = max_length
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        # Load captions
        df = pd.read_csv(captions_file, delimiter='|', encoding='utf-8')
        df.columns = df.columns.str.strip()
        
        # Detect column names
        if 'image_name' in df.columns:
            img_col = 'image_name'
            cap_col = 'comment' if 'comment' in df.columns else ' comment'
        else:
            img_col = df.columns[0]
            cap_col = df.columns[2] if len(df.columns) > 2 else df.columns[1]
        
        # Get unique images with their first caption
        self.data = []
        for img_name in df[img_col].unique():
            img_path = os.path.join(images_path, img_name)
            
            # Check if image exists
            if not os.path.exists(img_path):
                # Try alternative path
                img_path = os.path.join(images_path, "flickr30k_images", img_name)
                if not os.path.exists(img_path):
                    continue
            
            # Get first caption
            caption = df[df[img_col] == img_name][cap_col].iloc[0]
            caption = str(caption).strip()
            
            if not caption or caption == 'nan':
                continue
            
            # Tokenize
            tokens = caption.lower().replace('.', '').replace(',', '').split()
            
            # Skip very short or long captions
            if len(tokens) < 3 or len(tokens) > 30:
                continue
            
            self.data.append((img_path, tokens))
        
        # Train/val split
        split_idx = int(train_ratio * len(self.data))
        if split == 'train':
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
        
        print(f"{split.capitalize()} dataset: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (3, 224, 224)
            caption: Tensor of shape (max_length,) with token indices
        """
        img_path, tokens = self.data[idx]
        
        # Load and transform image
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            # Return dummy data if image fails to load
            img = torch.zeros(3, 224, 224)
            tokens = ['a', 'photo']
        
        # Convert caption to indices
        caption_indices = [self.token_to_idx.get('<START>', 1)]
        for token in tokens:
            if token in self.token_to_idx:
                caption_indices.append(self.token_to_idx[token])
        caption_indices.append(self.token_to_idx.get('<END>', 2))
        
        # Pad to max_length
        caption = torch.zeros(self.max_length, dtype=torch.long)
        caption[:len(caption_indices)] = torch.tensor(caption_indices[:self.max_length])
        
        return img, caption


def create_flickr_dataloaders(
    images_path,
    captions_file,
    batch_size=32,
    num_workers=2,
    train_ratio=0.8
):
    """
    Create train and val dataloaders for Flickr30k.
    
    Returns:
        train_loader, val_loader, vocab
    """
    # Build vocabulary first
    print("Building vocabulary...")
    df = pd.read_csv(captions_file, delimiter='|', encoding='utf-8')
    df.columns = df.columns.str.strip()
    
    # Detect columns
    if 'image_name' in df.columns:
        cap_col = 'comment' if 'comment' in df.columns else ' comment'
    else:
        cap_col = df.columns[2] if len(df.columns) > 2 else df.columns[1]
    
    # Build vocab
    vocab = {"<NULL>": 0, "<START>": 1, "<END>": 2}
    idx = 3
    
    for caption in df[cap_col]:
        caption = str(caption).strip()
        if caption and caption != 'nan':
            tokens = caption.lower().replace('.', '').replace(',', '').split()
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
    
    vocab_dict = {
        'token_to_idx': vocab,
        'idx_to_token': {v: k for k, v in vocab.items()}
    }
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = FlickrDataset(
        images_path, captions_file, vocab_dict,
        split='train', train_ratio=train_ratio
    )
    
    val_dataset = FlickrDataset(
        images_path, captions_file, vocab_dict,
        split='val', train_ratio=train_ratio
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, vocab_dict


if __name__ == "__main__":
    # Test
    images_path = "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images"
    captions_file = "/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv"
    
    train_loader, val_loader, vocab = create_flickr_dataloaders(
        images_path, captions_file, batch_size=32
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    images, captions = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Captions: {captions.shape}")
