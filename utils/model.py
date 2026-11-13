import torch
import torch.nn as nn

from .rules import triangular_activation


class SCORENet(nn.Module):
    def __init__(
        self,
        n_feature: int,
        n_hidden: int,
        n_classes: int,
        dropout_rate: float = 0.25,
        device: str = "cpu",
        output_bias = False
    ):
        super(SCORENet, self).__init__()

        self.N = n_feature
        self.H = n_hidden
        self.C = n_classes
        self.device = device
        
        # Linear projection from input features to hidden rule units.
        self.feature_linear = nn.Linear(self.N, self.H, bias=False)
        
        # One class-independent center and width per rule.
        self.base_feature_center = nn.Parameter(torch.zeros(self.H))
        self.base_feature_width = nn.Parameter(torch.rand(self.H))
        
        # Output layer: maps rule activations to class logits.
        self.output_linear = nn.Linear(self.H, self.C, bias=output_bias)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (B, N)
        
        h = self.feature_linear(x)  # (B, H)

        # Widths are kept positive.
        width = torch.abs(self.base_feature_width)  # (H,)

        center = self.base_feature_center  # broadcast over the batch

        activation = triangular_activation(h, center, width)  # (B, H)

        activation = self.dropout(activation)

        logits = self.output_linear(activation)  # (B, C)
        
        return logits