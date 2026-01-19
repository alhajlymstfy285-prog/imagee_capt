"""
Script to download and extract GloVe embeddings on Kaggle
"""
import os
import urllib.request
import zipfile

def download_glove():
    """Download GloVe 6B embeddings (300d)"""
    
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = "glove.6B.zip"
    
    print("Downloading GloVe embeddings...")
    print(f"URL: {url}")
    print("Size: ~822 MB - this will take a few minutes...")
    
    # Download
    urllib.request.urlretrieve(url, zip_path)
    print("✓ Download complete!")
    
    # Extract
    print("\nExtracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    
    print("✓ Extraction complete!")
    
    # List extracted files
    print("\nExtracted files:")
    for f in ['glove.6B.50d.txt', 'glove.6B.100d.txt', 
              'glove.6B.200d.txt', 'glove.6B.300d.txt']:
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  ✓ {f} ({size_mb:.1f} MB)")
    
    # Clean up zip
    os.remove(zip_path)
    print(f"\n✓ Removed {zip_path}")
    
    print("\n" + "="*50)
    print("GloVe embeddings ready to use!")
    print("File: glove.6B.300d.txt")
    print("="*50)

if __name__ == "__main__":
    download_glove()
