"""
سكريبت بسيط للتحقق من مسارات Kaggle dataset
Run this first to verify dataset is properly added
"""

import os
import sys

print("="*60)
print("Checking Kaggle Dataset Paths")
print("="*60)

# Check if we're on Kaggle
if os.path.exists('/kaggle/input'):
    print("✓ Running on Kaggle")
else:
    print("✗ Not running on Kaggle (or /kaggle/input not found)")
    sys.exit(1)

# List all datasets
print("\n📁 Available datasets:")
try:
    datasets = os.listdir('/kaggle/input')
    for ds in datasets:
        print(f"  - {ds}")
except Exception as e:
    print(f"  Error: {e}")
    sys.exit(1)

# Check for Flickr dataset
flickr_paths = [
    '/kaggle/input/flickr-image-dataset',
    '/kaggle/input/flickr30k-dataset',
    '/kaggle/input/flickr8k-dataset',
]

flickr_found = None
for path in flickr_paths:
    if os.path.exists(path):
        flickr_found = path
        break

if flickr_found:
    print(f"\n✓ Flickr dataset found at: {flickr_found}")
    
    # List contents
    print("\n📂 Dataset contents:")
    try:
        contents = os.listdir(flickr_found)
        for item in contents:
            item_path = os.path.join(flickr_found, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # List subdirectory
                try:
                    subitems = os.listdir(item_path)[:5]  # First 5 items
                    for subitem in subitems:
                        print(f"      - {subitem}")
                    if len(os.listdir(item_path)) > 5:
                        print(f"      ... and {len(os.listdir(item_path))-5} more")
                except:
                    pass
            else:
                size = os.path.getsize(item_path) / (1024*1024)  # MB
                print(f"  📄 {item} ({size:.1f} MB)")
    except Exception as e:
        print(f"  Error listing contents: {e}")
    
    # Look for images folder
    print("\n🔍 Looking for images folder...")
    possible_image_paths = [
        os.path.join(flickr_found, 'flickr30k_images'),
        os.path.join(flickr_found, 'flickr30k_images', 'flickr30k_images'),
        os.path.join(flickr_found, 'Images'),
        os.path.join(flickr_found, 'images'),
    ]
    
    images_path = None
    for path in possible_image_paths:
        if os.path.exists(path) and os.path.isdir(path):
            images_path = path
            num_images = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"  ✓ Found images at: {path}")
            print(f"    Number of images: {num_images}")
            break
    
    if not images_path:
        print("  ✗ Images folder not found!")
    
    # Look for captions file
    print("\n🔍 Looking for captions file...")
    possible_caption_files = [
        os.path.join(flickr_found, 'results.csv'),
        os.path.join(flickr_found, 'flickr30k_images', 'results.csv'),
        os.path.join(flickr_found, 'captions.txt'),
        os.path.join(flickr_found, 'captions.csv'),
    ]
    
    captions_file = None
    for path in possible_caption_files:
        if os.path.exists(path):
            captions_file = path
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"  ✓ Found captions at: {path}")
            print(f"    File size: {size:.1f} MB")
            
            # Try to read first few lines
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = [f.readline().strip() for _ in range(3)]
                print(f"    First lines:")
                for line in lines:
                    print(f"      {line[:80]}...")
            except:
                pass
            break
    
    if not captions_file:
        print("  ✗ Captions file not found!")
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    if images_path and captions_file:
        print("✓ Dataset is ready!")
        print(f"\nUse these paths in kaggle_setup.py:")
        print(f"  flickr_images_path = '{images_path}'")
        print(f"  captions_file = '{captions_file}'")
    else:
        print("✗ Dataset is incomplete!")
        if not images_path:
            print("  - Images folder not found")
        if not captions_file:
            print("  - Captions file not found")
    print("="*60)
    
else:
    print("\n✗ Flickr dataset not found!")
    print("\nPlease add Flickr dataset to your Kaggle notebook:")
    print("  1. Click 'Add Data' in the right panel")
    print("  2. Search for 'flickr' or 'flickr30k'")
    print("  3. Add the dataset")
    print("  4. Run this script again")
