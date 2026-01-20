"""
Vanilla RNN model for image captioning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from rnn_lstm_captioning import CaptioningRNN


class VanillaRNNCaptioner(nn.Module):
    """Vanilla RNN captioning model wrapper."""
    
    def __init__(
        self,
        word_to_idx: dict,
        wordvec_dim: int = 128,
        hidden_dim: int = 128,
        image_encoder_pretrained: bool = True,
        ignore_index: int = None,
        backbone: str = 'resnet50',
        glove_path: str = None,
        freeze_embeddings: bool = False,
        dropout: float = 0.3,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        
        self.model = CaptioningRNN(
            word_to_idx=word_to_idx,
            wordvec_dim=wordvec_dim,
            hidden_dim=hidden_dim,
            cell_type="rnn",
            image_encoder_pretrained=image_encoder_pretrained,
            ignore_index=ignore_index,
            backbone=backbone,
            glove_path=glove_path,
            freeze_embeddings=freeze_embeddings,
            dropout=dropout,
            label_smoothing=label_smoothing,
        )
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        return self.model(images, captions, teacher_forcing_ratio=teacher_forcing_ratio)
    
    def sample(self, images: torch.Tensor, max_length: int = 15, beam_size: int = 1, length_penalty: float = 0.7) -> torch.Tensor:
        return self.model.sample(images, max_length, beam_size=beam_size, length_penalty=length_penalty)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
