import math
from typing import Optional, Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction


def hello_rnn_lstm_captioning():
    print("Hello from rnn_lstm_captioning.py!")


class ImageEncoder(nn.Module):
    """
    CNN encoder using ResNet50 for extracting image features.
    Outputs spatial grid features for attention-based models.
    """

    def __init__(self, pretrained: bool = True, verbose: bool = True, backbone: str = 'resnet50'):
        """
        Args:
            pretrained: Whether to use ImageNet pretrained weights
            verbose: Whether to print output shapes
            backbone: CNN architecture ('resnet50', 'resnet101', 'regnet_x_400mf')
        """
        super().__init__()
        
        # Select backbone architecture
        if backbone == 'resnet50':
            self.cnn = torchvision.models.resnet50(pretrained=pretrained)
            feature_node = 'layer4'
        elif backbone == 'resnet101':
            self.cnn = torchvision.models.resnet101(pretrained=pretrained)
            feature_node = 'layer4'
        elif backbone == 'regnet_x_400mf':
            self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)
            feature_node = 'trunk_output.block4'
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.backbone_name = backbone
        
        # Extract spatial features before global pooling
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={feature_node: "c5"}
        )

        # Infer output channels
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))["c5"]
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print(f"Using {backbone} backbone")
            print(f"Input shape: (2, 3, 224, 224)")
            print(f"Output c5 features shape: {dummy_out.shape}")
            print(f"Output channels: {self._out_channels}")

        # ImageNet normalization
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        """Number of output channels in extracted features."""
        return self._out_channels

    def forward(self, images: torch.Tensor):
        # Convert uint8 to float if needed
        if images.dtype == torch.uint8:
            images = images.to(dtype=torch.float32)
            images /= 255.0

        # Normalize with ImageNet stats
        images = self.normalize(images)

        # Extract features
        features = self.backbone(images)["c5"]
        return features


##############################################################################
# Recurrent Neural Network                                                   #
##############################################################################
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Forward pass for a single timestep of a vanilla RNN.
    """
    next_h = torch.tanh(x @ Wx + prev_h @ Wh + b)
    cache = (x, prev_h, Wx, Wh, b)
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    """
    x, prev_h, Wx, Wh, b = cache
    out = torch.tanh(x @ Wx + prev_h @ Wh + b)
    dtanh = (1 - out**2) * dnext_h
    dx = dtanh @ Wx.T
    dprev_h = dtanh @ Wh.T
    dWx = x.T @ dtanh
    dWh = prev_h.T @ dtanh
    db = dtanh.sum(dim=0)
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for a vanilla RNN over an entire sequence.
    """
    N, T, D = x.shape
    h_list = []
    cache = []
    prev_h = h0

    for t in range(T):
        next_h, cache_t = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h_list.append(next_h)
        cache.append(cache_t)
        prev_h = next_h

    h = torch.stack(h_list, dim=1)
    return h, cache

def rnn_backward(dh, cache):
    """
    Backward pass for a vanilla RNN over an entire sequence.
    """
    N, T, H = dh.shape
    x, prev_h, Wx, Wh, b = cache[0]
    D = x.shape[1]

    dx = torch.zeros(N, T, D, dtype=dh.dtype, device=dh.device)
    dWx = torch.zeros(D, H, dtype=dh.dtype, device=dh.device)
    dWh = torch.zeros(H, H, dtype=dh.dtype, device=dh.device)
    db = torch.zeros(H, dtype=dh.dtype, device=dh.device)
    dprev_h = torch.zeros(N, H, dtype=dh.dtype, device=dh.device)

    for t in range(T - 1, -1, -1):
        dx_t, dprev_h_t, dWx_t, dWh_t, db_t = rnn_step_backward(
            dh[:, t, :] + dprev_h, cache[t]
        )
        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
        dprev_h = dprev_h_t

    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db

class RNN(nn.Module):
    """
    Single-layer vanilla RNN module.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h

def load_glove_embeddings(glove_path: str, word_to_idx: dict, embed_dim: int = 300):
    """
    Load GloVe embeddings and create embedding matrix for vocabulary.
    
    Args:
        glove_path: Path to GloVe file (e.g., 'glove.6B.300d.txt')
        word_to_idx: Dictionary mapping words to indices
        embed_dim: Dimension of GloVe embeddings
    
    Returns:
        embedding_matrix: Tensor of shape (vocab_size, embed_dim)
    """
    vocab_size = len(word_to_idx)
    embedding_matrix = torch.randn(vocab_size, embed_dim) * 0.01
    
    found_words = 0
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in word_to_idx:
                    idx = word_to_idx[word]
                    vector = torch.tensor([float(val) for val in values[1:]])
                    embedding_matrix[idx] = vector
                    found_words += 1
        
        print(f"Loaded GloVe embeddings: {found_words}/{vocab_size} words found")
    except FileNotFoundError:
        print(f"GloVe file not found at {glove_path}, using random initialization")
    
    return embedding_matrix


class WordEmbedding(nn.Module):
    """
    Word embedding layer with optional GloVe initialization.
    """

    def __init__(self, vocab_size: int, embed_size: int, pretrained_embeddings: Optional[torch.Tensor] = None, freeze: bool = False):
        super().__init__()
        
        if pretrained_embeddings is not None:
            self.W_embed = nn.Parameter(pretrained_embeddings.clone(), requires_grad=not freeze)
        else:
            self.W_embed = nn.Parameter(
                torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
            )

    def forward(self, x):
        return self.W_embed[x]


def temporal_softmax_loss(x, y, ignore_index=None):
    """
    Temporal softmax loss for sequence models.
    """
    N, T, V = x.shape
    loss = torch.nn.functional.cross_entropy(
        x.reshape(N * T, V), 
        y.reshape(N * T), 
        ignore_index=ignore_index, 
        reduction='sum',
        label_smoothing=0.1
    ) / N
    return loss


class CaptioningRNN(nn.Module):
    """
    Image captioning model using RNN/LSTM/Attention.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim: int = 512,
        wordvec_dim: int = 128,
        hidden_dim: int = 128,
        cell_type: str = "rnn",
        image_encoder_pretrained: bool = True,
        ignore_index: Optional[int] = None,
        glove_path: Optional[str] = None,
        freeze_embeddings: bool = False,
        backbone: str = 'resnet50',
    ):
        super().__init__()
        if cell_type not in {"rnn", "lstm", "attn"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        self.ignore_index = ignore_index

        self.image_encoder = ImageEncoder(
            pretrained=image_encoder_pretrained,
            backbone=backbone
        )
        
        # Load GloVe embeddings if path provided
        pretrained_embeddings = None
        if glove_path is not None:
            pretrained_embeddings = load_glove_embeddings(glove_path, word_to_idx, wordvec_dim)
        
        self.word_embedd = WordEmbedding(vocab_size, wordvec_dim, pretrained_embeddings, freeze_embeddings)
        self.feature_projection = nn.Linear(self.image_encoder.out_channels, hidden_dim)

        if cell_type == "rnn":
            self.rnn = RNN(wordvec_dim, hidden_dim)
        elif cell_type == "lstm":
            self.rnn = LSTM(wordvec_dim, hidden_dim)
        elif cell_type == "attn":
            self.rnn = AttentionLSTM(wordvec_dim, hidden_dim, self.image_encoder.out_channels)

        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, images, captions):
        """
        Compute training loss for the captioning model.
        """
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        features = self.image_encoder(images)
        
        if self.cell_type in ['lstm', 'rnn']:
            features_pooled = features.mean(dim=[2, 3])
            h0 = self.feature_projection(features_pooled)
            word_embedd = self.dropout(self.word_embedd(captions_in))
            h = self.rnn(word_embedd, h0)
            h = self.dropout(h)
            h = torch.nn.functional.layer_norm(h, h.shape[1:])
        elif self.cell_type == 'attn':
            word_embedd = self.dropout(self.word_embedd(captions_in))
            h = self.rnn(word_embedd, features)
            h = self.dropout(h)

        scores = self.output_projection(h)
        loss = temporal_softmax_loss(scores, captions_out, ignore_index=self._null)
        return loss

    def sample(self, images, max_length=15):
        """
        Generate captions for input images using greedy sampling.
        """
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == "attn":
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()

        features = self.image_encoder(images)
        
        if self.cell_type in ['rnn', 'lstm']:
            features_pooled = features.mean(dim=[2, 3])
            h = self.feature_projection(features_pooled)
            
            if self.cell_type == 'lstm':
                c = torch.zeros_like(h)
            
            current_word = torch.full((N,), self._start, dtype=torch.long, device=images.device)
            
            for t in range(max_length):
                word_embed = self.word_embedd(current_word)
                
                if self.cell_type == 'rnn':
                    h = self.rnn.step_forward(word_embed, h)
                elif self.cell_type == 'lstm':
                    h, c = self.rnn.step_forward(word_embed, h, c)
                
                scores = self.output_projection(h)
                current_word = scores.argmax(dim=1)
                captions[:, t] = current_word
        
        elif self.cell_type == 'attn':
            N, C, H_feat, W_feat = features.shape
            A = self.feature_projection(features.permute(0, 2, 3, 1))
            A = A.permute(0, 3, 1, 2)
            
            h = A.mean(dim=(2, 3))
            c = torch.zeros_like(h)
            
            current_word = torch.full((N,), self._start, dtype=torch.long, device=images.device)
            
            for t in range(max_length):
                word_embed = self.word_embedd(current_word)
                h, c, attn_weights = self.rnn.step_forward(word_embed, h, c, A)
                attn_weights_all[:, t] = attn_weights
                
                scores = self.output_projection(h)
                current_word = scores.argmax(dim=1)
                captions[:, t] = current_word
        
        if self.cell_type == "attn":
            return captions, attn_weights_all.cpu()
        else:
            return captions


class LSTM(nn.Module):
    """Single-layer, uni-directional LSTM module."""

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
            Wx: Weights for input-to-hidden connections, of shape (D, 4H)
            Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
            b: Biases, of shape (4H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep of an LSTM.
        """
        a = x @ self.Wx + prev_h @ self.Wh + self.b
        
        H = prev_h.shape[1]
        i = torch.sigmoid(a[:, :H])
        f = torch.sigmoid(a[:, H:2*H])
        o = torch.sigmoid(a[:, 2*H:3*H])
        g = torch.tanh(a[:, 3*H:])
        
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for an LSTM over an entire sequence.
        """
        c0 = torch.zeros_like(h0)
        N, T, D = x.shape
        h_list = []
        prev_h = h0
        prev_c = c0

        for t in range(T):
            next_h, next_c = self.step_forward(x[:, t, :], prev_h, prev_c)
            h_list.append(next_h)
            prev_h = next_h
            prev_c = next_c

        hn = torch.stack(h_list, dim=1)
        return hn


def dot_product_attention(prev_h, A):
    """
    Scaled dot-product attention layer.
    """
    N, H, D_a, _ = A.shape
    
    A_flat = A.reshape(N, H, D_a * D_a)
    prev_h_expanded = prev_h.unsqueeze(1)
    scores = (prev_h_expanded @ A_flat).squeeze(1)
    scores = scores / torch.sqrt(torch.tensor(H, dtype=scores.dtype))
    attn_weights_flat = torch.softmax(scores, dim=1)
    attn_weights = attn_weights_flat.reshape(N, D_a, D_a)
    attn = (A_flat @ attn_weights_flat.unsqueeze(2)).squeeze(2)
    return attn, attn_weights


class AttentionLSTM(nn.Module):
    """
    This is our single-layer, uni-directional Attention module.

    Args:
        input_dim: Input size, denoted as D before
        hidden_dim: Hidden size, denoted as H before
        attn_dim: Attention feature dimension (CNN output channels)
    """

    def __init__(self, input_dim: int, hidden_dim: int, attn_dim: int = 1280):
        super().__init__()
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.Wattn = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))
        self.attn_projection = nn.Conv2d(attn_dim, hidden_dim, kernel_size=1)

    def step_forward(
        self,
        x: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep of attention LSTM.
        """
        a = x @ self.Wx + prev_h @ self.Wh + attn @ self.Wattn + self.b
        
        H = prev_h.shape[1]
        i = torch.sigmoid(a[:, :H])
        f = torch.sigmoid(a[:, H:2*H])
        o = torch.sigmoid(a[:, 2*H:3*H])
        g = torch.tanh(a[:, 3*H:])
        
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        Forward pass for attention LSTM over an entire sequence.
        """
        A_proj = self.attn_projection(A)
        h0 = A_proj.mean(dim=(2, 3))
        c0 = h0

        N, T, D = x.shape
        h_list = []
        prev_h = h0
        prev_c = c0

        for t in range(T):
            attn, _ = dot_product_attention(prev_h, A_proj)
            next_h, next_c = self.step_forward(x[:, t, :], prev_h, prev_c, attn)
            h_list.append(next_h)
            prev_h = next_h
            prev_c = next_c

        hn = torch.stack(h_list, dim=1)
        return hn
