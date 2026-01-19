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
    flickr_path="/kaggle/input/flickr-image-dataset/flickr30k_images",
    captions_file="/kaggle/input/flickr-image-dataset/results.csv",
    output_path="./datasets/flickr.pt",
    max_samples=10000
):
    """
    تحويل Flickr dataset إلى الصيغة المطلوبة
    
    Args:
        flickr_path: مسار مجلد الصور
        captions_file: مسار ملف التعليقات
        output_path: مسار حفظ الملف المعالج
        max_samples: أقصى عدد من الصور (للتدريب السريع)
    """
    
    print("Loading Flickr dataset...")
    
    # تحميل التعليقات
    import pandas as pd
    df = pd.read_csv(captions_file, delimiter='|')
    df.columns = df.columns.str.strip()
    
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
    image_names = df['image_name'].unique()[:max_samples]
    
    for img_name in image_names:
        img_path = os.path.join(flickr_path, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        # تحميل الصورة
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            
            # الحصول على التعليق
            caption = df[df['image_name'] == img_name][' comment'].iloc[0].strip()
            
            # تحويل التعليق إلى tokens
            tokens = caption.lower().split()
            
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
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    # تقسيم البيانات
    split_idx = int(0.8 * len(images_list))
    
    # Pad captions to same length
    max_len = max(len(cap) for cap in captions_list)
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
    
    print(f"✓ Dataset saved to {output_path}")
    print(f"  Train images: {data_dict['train_images'].shape}")
    print(f"  Val images: {data_dict['val_images'].shape}")
    print(f"  Vocabulary size: {len(vocab)}")
    
    return data_dict


if __name__ == "__main__":
    # استخدم هذا في Kaggle notebook
    data = prepare_flickr_data()
