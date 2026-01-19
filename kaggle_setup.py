"""
سكريبت لتحضير البيانات من Flickr dataset على Kaggle
Run this in Kaggle notebook after adding Flickr dataset
"""

import os
import json
import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms


def prepare_flickr_data(
    flickr_images_path="/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images",
    captions_file="/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv",
    output_path="./datasets/flickr.pt",
    max_samples=5000
):
    """
    تحويل Flickr dataset إلى الصيغة المطلوبة
    
    Args:
        flickr_images_path: مسار مجلد الصور
        captions_file: مسار ملف التعليقات
        output_path: مسار حفظ الملف المعالج
        max_samples: أقصى عدد من الصور (للتدريب السريع)
    """
    
    print("="*60)
    print("Loading Flickr dataset...")
    print("="*60)
    
    # Check if paths exist
    if not os.path.exists(flickr_images_path):
        # Try alternative path
        flickr_images_path = "/kaggle/input/flickr-image-dataset/flickr30k_images"
        if not os.path.exists(flickr_images_path):
            raise FileNotFoundError(f"Images path not found: {flickr_images_path}")
    
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found: {captions_file}")
    
    print(f"✓ Images path: {flickr_images_path}")
    print(f"✓ Captions file: {captions_file}")
    
    # تحميل التعليقات
    import pandas as pd
    
    try:
        # Try reading with pipe delimiter
        df = pd.read_csv(captions_file, delimiter='|', encoding='utf-8')
    except:
        try:
            # Try reading with comma delimiter
            df = pd.read_csv(captions_file, encoding='utf-8')
        except Exception as e:
            print(f"Error reading CSV: {e}")
            raise
    
    # Clean column names
    df.columns = df.columns.str.strip()
    print(f"✓ Loaded {len(df)} captions")
    print(f"  Columns: {list(df.columns)}")
    
    # Detect column names (handle different formats)
    if 'image_name' in df.columns:
        img_col = 'image_name'
        cap_col = 'comment' if 'comment' in df.columns else ' comment'
    elif 'image' in df.columns:
        img_col = 'image'
        cap_col = 'caption'
    else:
        img_col = df.columns[0]
        cap_col = df.columns[1]
    
    print(f"  Using columns: image='{img_col}', caption='{cap_col}'")
    
    # تحويل الصور
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    
    # بناء المفردات
    vocab = {"<NULL>": 0, "<START>": 1, "<END>": 2}
    idx = 3
    
    images_list = []
    captions_list = []
    
    # معالجة البيانات
    image_names = df[img_col].unique()[:max_samples]
    print(f"\nProcessing {len(image_names)} images...")
    
    processed = 0
    skipped = 0
    
    for i, img_name in enumerate(image_names):
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{len(image_names)} images...")
        
        # Try different path combinations
        img_path = os.path.join(flickr_images_path, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(flickr_images_path, "flickr30k_images", img_name)
        
        if not os.path.exists(img_path):
            skipped += 1
            continue
            
        # تحميل الصورة
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            
            # الحصول على التعليق (أول تعليق للصورة)
            caption_row = df[df[img_col] == img_name].iloc[0]
            caption = str(caption_row[cap_col]).strip()
            
            # Skip if caption is empty or NaN
            if not caption or caption == 'nan':
                skipped += 1
                continue
            
            # تحويل التعليق إلى tokens (lowercase and split)
            tokens = caption.lower().replace('.', '').replace(',', '').split()
            
            # Skip very short or very long captions
            if len(tokens) < 3 or len(tokens) > 30:
                skipped += 1
                continue
            
            # إضافة كلمات جديدة للمفردات
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
            
            # تحويل التعليق إلى أرقام
            caption_indices = [vocab["<START>"]]
            caption_indices.extend([vocab[token] for token in tokens])
            caption_indices.append(vocab["<END>"])
            
            images_list.append(img_tensor)
            captions_list.append(torch.tensor(caption_indices))
            processed += 1
            
        except Exception as e:
            skipped += 1
            if skipped < 5:  # Only print first few errors
                print(f"  Warning: Error processing {img_name}: {e}")
            continue
    
    print(f"\n✓ Successfully processed: {processed} images")
    print(f"✗ Skipped: {skipped} images")
    
    if len(images_list) == 0:
        raise ValueError("No images were successfully processed!")
    
    # تقسيم البيانات (80% train, 20% val)
    split_idx = int(0.8 * len(images_list))
    print(f"\nSplitting data: {split_idx} train, {len(images_list)-split_idx} val")
    
    # Pad captions to same length
    max_len = max(len(cap) for cap in captions_list)
    print(f"Max caption length: {max_len} tokens")
    
    captions_padded = []
    for cap in captions_list:
        padded = torch.zeros(max_len, dtype=torch.long)
        padded[:len(cap)] = cap
        captions_padded.append(padded)
    
    # تجميع البيانات
    data_dict = {
        "train_images": torch.stack(images_list[:split_idx]),
        "val_images": torch.stack(images_list[split_idx:]),
        "train_captions": torch.stack(captions_padded[:split_idx]),
        "val_captions": torch.stack(captions_padded[split_idx:]),
        "vocab": {
            "token_to_idx": vocab,
            "idx_to_token": {v: k for k, v in vocab.items()}
        }
    }
    
    # حفظ البيانات
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data_dict, output_path)
    
    print("\n" + "="*60)
    print("✓ Dataset saved successfully!")
    print("="*60)
    print(f"  Output: {output_path}")
    print(f"  Train images: {data_dict['train_images'].shape}")
    print(f"  Train captions: {data_dict['train_captions'].shape}")
    print(f"  Val images: {data_dict['val_images'].shape}")
    print(f"  Val captions: {data_dict['val_captions'].shape}")
    print(f"  Vocabulary size: {len(vocab)} words")
    print("="*60)
    
    return data_dict


if __name__ == "__main__":
    # استخدم هذا في Kaggle notebook
    try:
        data = prepare_flickr_data()
        print("\n✓ Setup complete! You can now run training scripts.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease check:")
        print("  1. Flickr dataset is added to Kaggle notebook")
        print("  2. Dataset path is correct")
        print("  3. All required packages are installed")
        raise
