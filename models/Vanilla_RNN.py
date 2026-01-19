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
    ):
        super().__init__()
        
        self.model = CaptioningRNN(
            word_to_idx=word_to_idx,
            wordvec_dim=wordvec_dim,
            hidden_dim=hidden_dim,
            cell_type="rnn",
            image_encoder_pretrained=image_encoder_pretrained,
            ignore_index=ignore_index,
        )
    
    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        return self.model(images, captions)
    
    def sample(self, images: torch.Tensor, max_length: int = 15) -> torch.Tensor:
        return self.model.sample(images, max_length)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
