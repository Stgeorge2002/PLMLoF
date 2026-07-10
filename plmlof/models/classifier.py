"""Classification and regression heads for PLMLoF."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification.

    Down-weights easy examples so the model focuses on hard ones (e.g. WT
    variants near the LoF/GoF boundary).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class ClassifierHead(nn.Module):
    """MLP classification head with residual connections.

    Takes concatenated comparison features + nucleotide features
    and outputs class logits.
    """

    def __init__(
        self,
        input_size: int,
        hidden_dims: list[int] | None = None,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_size: Size of input feature vector (comparison_size + num_nuc_features).
            hidden_dims: Hidden layer dimensions. Default: [256, 64].
            num_classes: Number of output classes (3: LoF, WT, GoF).
            dropout: Dropout probability.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 64]

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Input projection to match first hidden dim
        self.input_proj = nn.Linear(input_size, hidden_dims[0])
        
        # Hidden layers with residual connections
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layer_norms.append(nn.LayerNorm(hidden_dims[i + 1]))
        
        # Output layer
        self.output = nn.Linear(hidden_dims[-1], num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)  # Small gain for residual
            nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features [batch, input_size].

        Returns:
            Logits [batch, num_classes].
        """
        # Initial projection
        x = self.input_proj(features)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers with residual connections
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            residual = x
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            
            # Residual connection if dimensions match
            if x.shape == residual.shape:
                x = x + residual
            
            # Decrease dropout for later layers
            drop_rate = self.dropout * (1 - (i + 1) * 0.3)
            x = F.dropout(x, p=max(drop_rate, 0.1), training=self.training)
        
        # Output layer
        return self.output(x)


class RegressionHead(nn.Module):
    """MLP regression head for predicting continuous DMS fitness z-scores.

    Shares the same input features as ClassifierHead but outputs a scalar.
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features [batch, input_size].

        Returns:
            Predicted z-scores [batch].
        """
        return self.mlp(features).squeeze(-1).clamp(-10.0, 10.0)
