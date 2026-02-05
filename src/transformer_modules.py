"""
Lightweight Transformer Modules for BEV Refinement
Deformable attention for efficient spatial modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionEmbeddingSine(nn.Module):
    """2D positional encoding for BEV grid"""
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            pos: (B, 2*num_pos_feats, H, W)
        """
        B, C, H, W = x.shape
        
        # Create coordinate grids
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device)
        
        if self.normalize:
            y_embed = y_embed / (H - 1) * self.scale
            x_embed = x_embed / (W - 1) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        # Compute positional encodings for each dimension
        pos_x = x_embed[:, None] / dim_t  # (W, num_pos_feats)
        pos_y = y_embed[:, None] / dim_t  # (H, num_pos_feats)
        
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # (W, num_pos_feats)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)  # (H, num_pos_feats)
        
        # Create 2D position embeddings
        pos_y = pos_y.unsqueeze(1).repeat(1, W, 1)  # (H, W, num_pos_feats)
        pos_x = pos_x.unsqueeze(0).repeat(H, 1, 1)  # (H, W, num_pos_feats)
        
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)  # (2*num_pos_feats, H, W)
        pos = pos.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2*num_pos_feats, H, W)
        
        return pos


class DeformableAttention(nn.Module):
    """Simplified Deformable Attention for BEV features
    
    Instead of attending to all 40k positions, sample K=8 reference points
    """
    def __init__(self, d_model=256, n_heads=8, n_points=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        
        # Learnable offset prediction
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        
        # Attention weights for each sampling point
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        
        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        
        # Initialize offsets to form a regular grid
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        
        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1
            
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)
    
    def forward(self, query, value, reference_points):
        """
        Args:
            query: (B, N, C) where N = H*W
            value: (B, N, C)
            reference_points: (B, N, 2) normalized coordinates in [0, 1]
            
        Returns:
            output: (B, N, C)
        """
        B, N, C = query.shape
        H = W = int(math.sqrt(N))
        
        # Predict sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(B, N, self.n_heads, self.n_points, 2)
        
        # Predict attention weights
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(B, N, self.n_heads, self.n_points)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Get sampling locations
        sampling_locations = reference_points[:, :, None, None, :] + sampling_offsets / H
        sampling_locations = sampling_locations.clamp(0, 1)
        
        # Project value
        value = self.value_proj(value).view(B, H, W, self.n_heads, C // self.n_heads)
        value = value.permute(0, 3, 1, 2, 4)  # (B, n_heads, H, W, C//n_heads)
        
        # Sample features at predicted locations
        # Simplified: use grid_sample
        sampled_features = []
        for head in range(self.n_heads):
            head_locs = sampling_locations[:, :, head, :, :]  # (B, N, n_points, 2)
            head_locs = head_locs.view(B, H, W, self.n_points, 2)
            
            # Convert to grid_sample format [-1, 1]
            grid = head_locs * 2.0 - 1.0
            grid = grid.view(B, H * W * self.n_points, 1, 2)
            
            # Sample
            head_value = value[:, head:head+1, :, :, :]  # (B, 1, H, W, C//n_heads)
            head_value = head_value.squeeze(1).permute(0, 3, 1, 2)  # (B, C//n_heads, H, W)
            
            sampled = F.grid_sample(head_value, grid, mode='bilinear', align_corners=False)
            sampled = sampled.squeeze(-1).view(B, C // self.n_heads, H, W, self.n_points)
            sampled = sampled.permute(0, 2, 3, 4, 1).contiguous()  # (B, H, W, n_points, C//n_heads)
            sampled_features.append(sampled)
        
        sampled_features = torch.stack(sampled_features, dim=3)  # (B, H, W, n_heads, n_points, C//n_heads)
        sampled_features = sampled_features.view(B, N, self.n_heads, self.n_points, C // self.n_heads)
        
        # Apply attention weights
        output = (sampled_features * attention_weights.unsqueeze(-1)).sum(dim=3)
        output = output.view(B, N, C)
        
        # Output projection
        output = self.output_proj(output)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with deformable attention"""
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # Deformable self-attention
        self.self_attn = DeformableAttention(d_model, n_heads, n_points=8)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
    
    def forward(self, src, pos, reference_points):
        """
        Args:
            src: (B, N, C)
            pos: (B, C, H, W) positional encoding
            reference_points: (B, N, 2) normalized grid coordinates
        """
        B, N, C = src.shape
        H = W = int(math.sqrt(N))
        
        # Add positional encoding
        pos_flat = pos.flatten(2).permute(0, 2, 1)  # (B, N, C)
        q = k = src + pos_flat
        
        # Self-attention with residual
        src2 = self.self_attn(q, src, reference_points)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward with residual
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class LightweightBEVTransformer(nn.Module):
    """Lightweight Transformer for BEV feature refinement
    
    Single layer transformer with deformable attention
    """
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Positional encoding
        self.pos_encoder = PositionEmbeddingSine(d_model // 2, normalize=True)
        
        # Single transformer layer (lightweight!)
        self.encoder = TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
    
    def forward(self, x):
        """
        Args:
            x: BEV features (B, C, H, W)
            
        Returns:
            refined_x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Generate positional encoding
        pos = self.pos_encoder(x)  # (B, C, H, W)
        
        # Flatten spatial dimensions
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        
        # Create reference points (normalized grid)
        y_grid = torch.linspace(0, 1, H, device=x.device)
        x_grid = torch.linspace(0, 1, W, device=x.device)
        grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing='ij')
        reference_points = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        reference_points = reference_points.view(-1, 2).unsqueeze(0).repeat(B, 1, 1)  # (B, H*W, 2)
        
        # Apply transformer
        refined = self.encoder(x_flat, pos, reference_points)
        
        # Reshape back to spatial
        refined = refined.permute(0, 2, 1).view(B, C, H, W)
        
        return refined


def test_transformer():
    """Test transformer modules"""
    model = LightweightBEVTransformer(d_model=256, n_heads=8)
    
    # Test forward
    x = torch.randn(2, 256, 200, 200)
    output = model(x)
    
    print("Lightweight Transformer Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")


if __name__ == '__main__':
    test_transformer()
