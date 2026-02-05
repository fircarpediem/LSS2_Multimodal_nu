"""
VoVNetV2 Backbone using timm library (CORRECT implementation)
Much simpler and guaranteed to work with pretrained weights
"""

import torch
import torch.nn as nn


class VoVNetV2(nn.Module):
    """VoVNet-V2 backbone using timm library
    
    This uses the official timm implementation which is:
    - Correct architecture (matches papers)
    - Has pretrained weights
    - Well-tested and maintained
    
    Args:
        model_name (str): 'vovnet39' or 'vovnet57'
        pretrained (bool): Load ImageNet pretrained weights
    """
    
    def __init__(self, model_name='vovnet39', pretrained=False):
        super(VoVNetV2, self).__init__()
        
        try:
            import timm
        except ImportError:
            raise ImportError("timm library required. Install with: pip install timm")
        
        # Map our names to timm names
        timm_model_map = {
            'vovnet39': 'ese_vovnet39b',
            'vovnet57': 'ese_vovnet57b',
        }
        
        if model_name not in timm_model_map:
            raise ValueError(f"model_name must be one of {list(timm_model_map.keys())}")
        
        self.model_name = model_name
        timm_name = timm_model_map[model_name]
        
        # Create backbone using timm (features_only=True to get intermediate features)
        print(f"Creating {model_name} backbone using timm ({timm_name})...")
        if pretrained:
            print(f"Loading pretrained ImageNet weights...")
        
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3, 4),  # Get C3 and C4 features (stage 3 and 4)
        )
        
        # VoVNet output channels are fixed for all variants
        # Stage 3 (C3): 768 channels, Stage 4 (C4): 1024 channels
        self.c3_channels = 768
        self.c4_channels = 1024
        
        print(f"✓ Backbone created successfully")
        print(f"  C3 channels: {self.c3_channels}")
        print(f"  C4 channels: {self.c4_channels}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total_params/1e6:.2f}M")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B*N, 3, H, W)
            
        Returns:
            Dictionary with multi-scale features:
                'c3': (B*N, C3, H/16, W/16)
                'c4': (B*N, C4, H/16, W/16)
        """
        # Get features from timm backbone
        features = self.backbone(x)
        
        # features is a list: [c3, c4]
        c3 = features[0]
        c4 = features[1]
        
        return {
            'c3': c3,
            'c4': c4
        }


def test_vovnet():
    """Test VoVNetV2 backbone"""
    print("=" * 70)
    print("Testing VoVNetV2 with timm backend")
    print("=" * 70)
    
    for model_name in ['vovnet39', 'vovnet57']:
        print(f"\n{'='*70}")
        print(f"Testing {model_name.upper()}")
        print(f"{'='*70}")
        
        # Test with pretrained
        model = VoVNetV2(model_name=model_name, pretrained=True)
        
        # Test forward pass
        x = torch.randn(2, 3, 128, 352)
        outputs = model(x)
        
        print(f"\nForward pass:")
        print(f"  Input: {x.shape}")
        print(f"  C3 output: {outputs['c3'].shape}")
        print(f"  C4 output: {outputs['c4'].shape}")
        
        # Expected parameter counts
        expected_params = {
            'vovnet39': 22.6,  # Approximately
            'vovnet57': 36.6,  # Approximately
        }
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nParameters:")
        print(f"  Total: {total_params/1e6:.2f}M")
        print(f"  Expected: ~{expected_params[model_name]}M")
        
        # Check if close to expected
        diff_pct = abs(total_params/1e6 - expected_params[model_name]) / expected_params[model_name] * 100
        if diff_pct < 10:
            print(f"  ✓ Parameter count matches! (within {diff_pct:.1f}%)")
        else:
            print(f"  ⚠ Parameter count mismatch: {diff_pct:.1f}% difference")
        
        del model
    
    print("\n" + "=" * 70)
    print("✓ All tests completed!")
    print("=" * 70)


if __name__ == '__main__':
    test_vovnet()
