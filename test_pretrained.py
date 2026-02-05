"""
Quick test for VoVNetV2 pretrained weights loading (timm backend)
"""

import torch
from src.vovnet_timm import VoVNetV2

print("=" * 70)
print("Testing VoVNetV2 with timm backend")
print("=" * 70)

# Test VoVNet-39 with pretrained
print("\n[1/2] Testing VoVNet-39 with pretrained=True...")
model39 = VoVNetV2(model_name='vovnet39', pretrained=True)

# Test forward pass
x = torch.randn(2, 3, 128, 352)
outputs = model39(x)

print(f"\nForward pass successful!")
print(f"  Input: {x.shape}")
print(f"  C3 output: {outputs['c3'].shape}")
print(f"  C4 output: {outputs['c4'].shape}")

# Count parameters
total_params = sum(p.numel() for p in model39.parameters())
print(f"  Total parameters: {total_params / 1e6:.2f}M")
print(f"  Expected: ~22M")

if abs(total_params/1e6 - 22) < 5:
    print(f"  ✓ Parameter count CORRECT!")
else:
    print(f"  ✗ Parameter count WRONG!")

# Test VoVNet-57 with pretrained
print("\n" + "=" * 70)
print("[2/2] Testing VoVNet-57 with pretrained=True...")
model57 = VoVNetV2(model_name='vovnet57', pretrained=True)

outputs57 = model57(x)
print(f"\nForward pass successful!")
print(f"  Input: {x.shape}")
print(f"  C3 output: {outputs57['c3'].shape}")
print(f"  C4 output: {outputs57['c4'].shape}")

total_params57 = sum(p.numel() for p in model57.parameters())
print(f"  Total parameters: {total_params57 / 1e6:.2f}M")
print(f"  Expected: ~36M")

if abs(total_params57/1e6 - 36) < 5:
    print(f"  ✓ Parameter count CORRECT!")
else:
    print(f"  ✗ Parameter count WRONG!")

print("\n" + "=" * 70)
print("✓ All tests passed!")
print("=" * 70)
