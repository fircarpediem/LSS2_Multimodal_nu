"""
VoVNet-V2-99 Backbone Implementation
Memory-efficient backbone for BEV perception
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class eSEModule(nn.Module):
    """Effective Squeeze-and-Excitation - Lightweight channel attention"""
    def __init__(self, channel):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0)
        self.hsigmoid = nn.Hardsigmoid(inplace=True)
        
    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class OSA_module(nn.Module):
    """One-Shot Aggregation Module - Core of VoVNet"""
    def __init__(self, in_ch, out_ch, layer_num=5, identity=False):
        super(OSA_module, self).__init__()
        self.identity = identity
        self.layers = nn.ModuleList()
        
        # Progressive feature extraction
        in_channel = in_ch
        for i in range(layer_num):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            in_channel = out_ch
        
        # One-shot aggregation
        concat_ch = in_ch + out_ch * layer_num
        self.concat_conv = nn.Sequential(
            nn.Conv2d(concat_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # Effective SE attention
        self.ese = eSEModule(out_ch)
        
    def forward(self, x):
        identity_feat = x
        output = [x]
        
        # Progressive feature extraction
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        
        # One-shot aggregation (concat only once!)
        x = torch.cat(output, dim=1)
        x = self.concat_conv(x)
        x = self.ese(x)
        
        if self.identity:
            x = x + identity_feat
            
        return x


class VoVNetV2(nn.Module):
    """VoVNet-V2 family for BEV perception
    Improved version with Effective Squeeze-Excitation (eSE) attention
    
    Args:
        model_name (str): 'vovnet39', 'vovnet57', or 'vovnet99'
        pretrained (bool): Load ImageNet pretrained weights
    """
    
    # Configuration for different VoVNet variants
    CONFIGS = {
        'vovnet39': {
            'stem_ch': 128,
            'stage_configs': [
                (1, 256),   # Stage 1: 1 block, 256 channels
                (1, 512),   # Stage 2: 1 block, 512 channels
                (1, 768),   # Stage 3: 1 block, 768 channels
                (1, 1024),  # Stage 4: 1 block, 1024 channels
            ],
            'params': '22M',
        },
        'vovnet57': {
            'stem_ch': 128,
            'stage_configs': [
                (1, 256),   # Stage 1
                (1, 512),   # Stage 2
                (2, 768),   # Stage 3: 2 blocks
                (2, 1024),  # Stage 4: 2 blocks
            ],
            'params': '36M',
        },
        'vovnet99': {
            'stem_ch': 128,
            'stage_configs': [
                (1, 256),   # Stage 1
                (1, 512),   # Stage 2
                (2, 768),   # Stage 3
                (2, 1024),  # Stage 4
            ],
            'params': '54M',
        },
    }
    
    def __init__(self, model_name='vovnet57', pretrained=False):
        super(VoVNetV2, self).__init__()
        
        if model_name not in self.CONFIGS:
            raise ValueError(f"model_name must be one of {list(self.CONFIGS.keys())}")
        
        config = self.CONFIGS[model_name]
        stem_ch = config['stem_ch']
        stage_configs = config['stage_configs']
        
        self.model_name = model_name
        
        # Stem: 3 conv layers
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_ch, stem_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_ch, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Max pooling
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Build stages
        self.stage1 = self._make_stage(256, *stage_configs[0])
        self.stage2 = self._make_stage(256, *stage_configs[1])
        self.stage3 = self._make_stage(512, *stage_configs[2])
        self.stage4 = self._make_stage(768, *stage_configs[3])
        
        # Store output channels for reference
        self.c3_channels = stage_configs[2][1]  # 768
        self.c4_channels = stage_configs[3][1]  # 1024
        
        # Initialize weights
        self._initialize_weights()
        
        if pretrained:
            self._load_pretrained_weights()
    
    def _make_stage(self, in_ch, num_blocks, out_ch):
        """Create a stage with multiple OSA blocks"""
        layers = []
        for i in range(num_blocks):
            layers.append(
                OSA_module(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    layer_num=5,
                    identity=(i > 0)
                )
            )
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights with proper initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_weights(self):
        """Load pretrained weights from timm library
        
        Strategy: Load only matching backbone layers, ignore classifier/head
        """
        try:
            import timm
            
            # Map our model names to timm model names
            timm_model_map = {
                'vovnet39': 'ese_vovnet39b',  # ese_vovnet39b.ra_in1k
                'vovnet57': 'ese_vovnet57b',  # ese_vovnet57b.ra4_e3600_r256_in1k (higher resolution trained)
                'vovnet99': None,  # Not available in timm
            }
            
            if self.model_name not in timm_model_map:
                print(f"Warning: Unknown model {self.model_name}")
                return
            
            timm_name = timm_model_map[self.model_name]
            
            if timm_name is None:
                print(f"Warning: {self.model_name} pretrained weights not available in timm")
                print("Continuing with random initialization...")
                return
            
            print(f"Loading pretrained weights: {timm_name} from timm...")
            
            # Load timm model with pretrained weights (features_only to get just backbone)
            try:
                # Load full model to get all weights
                timm_model = timm.create_model(timm_name, pretrained=True)
                timm_state = timm_model.state_dict()
                
                # Our model state
                our_state = self.state_dict()
                
                # Smart mapping: Only load weights that match our architecture
                matched_state = {}
                loaded_count = 0
                
                for our_key, our_param in our_state.items():
                    # Try direct match first
                    if our_key in timm_state:
                        if our_param.shape == timm_state[our_key].shape:
                            matched_state[our_key] = timm_state[our_key]
                            loaded_count += 1
                            continue
                    
                    # Try without module prefix (some models use 'module.')
                    alt_key = our_key.replace('module.', '')
                    if alt_key in timm_state:
                        if our_param.shape == timm_state[alt_key].shape:
                            matched_state[our_key] = timm_state[alt_key]
                            loaded_count += 1
                            continue
                
                # Load matched weights only
                if matched_state:
                    self.load_state_dict(matched_state, strict=False)
                    
                    print(f"✓ Loaded pretrained weights for {self.model_name}")
                    print(f"  Matched layers: {loaded_count}/{len(our_state)}")
                    print(f"  Coverage: {loaded_count/len(our_state)*100:.1f}%")
                    
                    # Verify total parameters didn't change
                    total_params = sum(p.numel() for p in self.parameters())
                    print(f"  Model parameters: {total_params/1e6:.2f}M (should be ~{self.CONFIGS[self.model_name]['params']})")
                else:
                    print(f"Warning: No matching weights found between timm and our model!")
                    print("This likely means architecture mismatch. Using random initialization...")
                    
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                print("Available VoVNet models in timm:")
                try:
                    vovnet_models = timm.list_models('*vovnet*', pretrained=True)
                    for m in vovnet_models:
                        print(f"  - {m}")
                except:
                    pass
                print("Continuing with random initialization...")
                
        except ImportError:
            print("Warning: timm library not installed. Install with: pip install timm")
            print("Continuing with random initialization...")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B*N, 3, H, W)
            
        Returns:
            Dictionary with multi-scale features:
                'c3': (B*N, 768, H/16, W/16)
                'c4': (B*N, 1024, H/16, W/16)
        """
        # Input: (B*N, 3, 128, 352)
        x = self.stem(x)           # (B*N, 256, 64, 176) - stride 2
        x = self.pool(x)           # (B*N, 256, 32, 88)  - stride 4
        
        x = self.stage1(x)         # (B*N, 256, 32, 88)
        x = self.pool(x)           # (B*N, 256, 16, 44)  - stride 8
        
        x = self.stage2(x)         # (B*N, 512, 16, 44)
        x = self.pool(x)           # (B*N, 512, 8, 22)   - stride 16
        
        c3 = self.stage3(x)        # (B*N, 768, 8, 22)
        c4 = self.stage4(c3)       # (B*N, 1024, 8, 22)
        
        return {
            'c3': c3,
            'c4': c4
        }


def test_vovnet():
    """Test VoVNetV2 backbone"""
    print("=" * 60)
    
    for model_name in ['vovnet39', 'vovnet57', 'vovnet99']:
        print(f"\nTesting VoVNetV2-{model_name[6:].upper()}:")
        print("-" * 60)
        
        model = VoVNetV2(model_name=model_name, pretrained=False)
        
        # Test forward pass
        x = torch.randn(4, 3, 128, 352)  # Batch of 4 images
        outputs = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"C3 shape: {outputs['c3'].shape}")
        print(f"C4 shape: {outputs['c4'].shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    print("=" * 60)


if __name__ == '__main__':
    test_vovnet()
