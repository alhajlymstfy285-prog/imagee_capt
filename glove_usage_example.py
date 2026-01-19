"""
Example: Using GloVe Embeddings with Image Captioning Models

This script demonstrates how to use pre-trained GloVe embeddings
instead of training embeddings from scratch.
"""

import torch
from rnn_lstm_captioning import CaptioningRNN, load_glove_embeddings

# Example usage with GloVe embeddings

# 1. Download GloVe embeddings from: https://nlp.stanford.edu/projects/glove/
#    Common options:
#    - glove.6B.50d.txt   (50 dimensions)
#    - glove.6B.100d.txt  (100 dimensions)
#    - glove.6B.200d.txt  (200 dimensions)
#    - glove.6B.300d.txt  (300 dimensions)

# 2. Load your vocabulary (example)
word_to_idx = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    'a': 3,
    'dog': 4,
    'cat': 5,
    # ... rest of vocabulary
}

# 3. Create model with GloVe embeddings
model = CaptioningRNN(
    word_to_idx=word_to_idx,
    wordvec_dim=300,  # Must match GloVe dimension
    hidden_dim=512,
    cell_type='lstm',
    glove_path='glove.6B.300d.txt',  # Path to GloVe file
    freeze_embeddings=False  # Set to True to freeze embeddings during training
)

# 4. Alternative: Load GloVe separately and pass to model
glove_embeddings = load_glove_embeddings(
    glove_path='glove.6B.300d.txt',
    word_to_idx=word_to_idx,
    embed_dim=300
)

# Then create model without glove_path but with pretrained embeddings
from rnn_lstm_captioning import WordEmbedding
word_embedd = WordEmbedding(
    vocab_size=len(word_to_idx),
    embed_size=300,
    pretrained_embeddings=glove_embeddings,
    freeze=False
)

print("GloVe embeddings loaded successfully!")
print(f"Embedding shape: {model.word_embedd.W_embed.shape}")
print(f"Embeddings frozen: {not model.word_embedd.W_embed.requires_grad}")
