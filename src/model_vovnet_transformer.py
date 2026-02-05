"""
VoVNet-99 + LSS v2 + Lightweight Transformer
Main model implementation for multi-modal BEV perception
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vovnet_timm import VoVNetV2  # Use timm backend (correct + pretrained)
    from .transformer_modules import LightweightBEVTransformer
    from .tools import gen_dx_bx, QuickCumsum
    from .modules import BevPost, Predictor
except ImportError:
    from vovnet_timm import VoVNetV2
    from transformer_modules import LightweightBEVTransformer
    from tools import gen_dx_bx, QuickCumsum
    from modules import BevPost, Predictor


class MultiScaleDepthNet(nn.Module):
    """Multi-scale depth prediction (LSS v2 improvement)"""
    def __init__(self, c3_channels=768, c4_channels=1024, depth_channels=41):
        super().__init__()
        
        # Depth prediction from C3
        self.depth_c3 = nn.Sequential(
            nn.Conv2d(c3_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, depth_channels, 1)
        )
        
        # Depth prediction from C4
        self.depth_c4 = nn.Sequential(
            nn.Conv2d(c4_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, depth_channels, 1)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(depth_channels * 2, depth_channels, 1),
            nn.BatchNorm2d(depth_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, c3, c4):
        """
        Args:
            c3: (B*N, 768, H, W) - 1/16 resolution
            c4: (B*N, 1024, H/2, W/2) - 1/32 resolution
        Returns:
            depth: (B*N, D, H, W) - depth probability distribution
        """
        d3 = self.depth_c3(c3)  # (B*N, D, H, W)
        d4 = self.depth_c4(c4)  # (B*N, D, H/2, W/2)
        
        # Upsample d4 to match d3's spatial dimensions
        d4_upsampled = F.interpolate(d4, size=d3.shape[2:], mode='bilinear', align_corners=False)
        
        # Fuse multi-scale
        depth_fused = self.fusion(torch.cat([d3, d4_upsampled], dim=1))
        
        # Softmax to get probability distribution
        depth_prob = F.softmax(depth_fused, dim=1)
        
        return depth_prob


class CamEncodeV2(nn.Module):
    """Camera encoding with depth (LSS v2)"""
    def __init__(self, D, C_in, C_out):
        super().__init__()
        self.D = D
        self.C_out = C_out
        
        # Feature projection
        self.feat_proj = nn.Conv2d(C_in, C_out, 1)
    
    def forward(self, features, depth):
        """
        Args:
            features: (B*N, C_in, H, W)
            depth: (B*N, D, H, W) - depth distribution
        Returns:
            cam_feats: (B*N, C_out, D, H, W)
        """
        B_N, C, H, W = features.shape
        
        # Project features
        feat = self.feat_proj(features)  # (B*N, C_out, H, W)
        
        # Depth-weighted features
        # depth: (B*N, D, H, W) -> (B*N, 1, D, H, W)
        # feat:  (B*N, C, H, W) -> (B*N, C, 1, H, W)
        depth_expanded = depth.unsqueeze(1)  # (B*N, 1, D, H, W)
        feat_expanded = feat.unsqueeze(2)    # (B*N, C, 1, H, W)
        
        # Multiply: (B*N, C, D, H, W)
        cam_feats = feat_expanded * depth_expanded
        
        return cam_feats


class BEVEncoderTransformer(nn.Module):
    """BEV encoder with transformer refinement"""
    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        
        # Compress BEV features
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Lightweight transformer
        self.transformer = LightweightBEVTransformer(
            d_model=256, 
            n_heads=8, 
            dim_feedforward=1024,
            dropout=0.1
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C_in, H, W) raw BEV features
        Returns:
            seg: (B, out_C, H, W) segmentation
            features: (B, 256, H, W) refined features for other tasks
        """
        # Compress
        x = self.compress(x)  # (B, 256, H, W)
        
        # Transformer refinement
        refined = self.transformer(x)  # (B, 256, H, W)
        
        # Segmentation
        seg = self.seg_head(refined)
        
        return seg, refined


class VoVNetBEVTransformer(nn.Module):
    """
    VoVNet-V2 + LSS v2 + Lightweight Transformer
    For multi-modal BEV perception (BEV seg + Action + Description)
    
    Args:
        vovnet_type (str): 'vovnet39' (default), 'vovnet57', or 'vovnet99'
    """
    def __init__(self, bsize, grid_conf, data_aug_conf, outC=4, vovnet_type='vovnet57', pretrained=True):
        super().__init__()
        self.bsize = bsize
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.vovnet_type = vovnet_type
        
        # Grid parameters
        dx, bx, nx = gen_dx_bx(
            grid_conf['xbound'],
            grid_conf['ybound'],
            grid_conf['zbound']
        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        
        # LSS parameters
        self.downsample = 16
        self.D = 41  # Depth bins
        self.C = 128  # Feature channels
        
        # Create frustum
        self.frustum = self.create_frustum()
        
        # VoVNet-V2 backbone (configurable)
        self.backbone = VoVNetV2(model_name=vovnet_type, pretrained=pretrained)
        
        # Multi-scale depth prediction (LSS v2)
        # All VoVNet variants output 768 (C3) and 1024 (C4) channels
        self.depth_net = MultiScaleDepthNet(
            c3_channels=self.backbone.c3_channels,
            c4_channels=self.backbone.c4_channels,
            depth_channels=self.D
        )
        
        # Camera encoding (LSS v2)
        # Use C3 features since depth is computed at C3 resolution
        self.cam_encode = CamEncodeV2(
            D=self.D,
            C_in=self.backbone.c3_channels,  # Use C3 (768) instead of C4
            C_out=self.C
        )
        
        # BEV encoder with transformer
        self.bev_encoder = BEVEncoderTransformer(
            in_channels=self.C * self.nx[2].item(),  # C * Z
            out_channels=outC
        )
        
        # BEV post-processing for action/desc
        self.bev_post = BevPost(in_channels=256, out_channels=8)
        
        # Action & Description heads
        # BevPost output: (B, 8, 8, 22) after crop and pooling
        self.action_head = nn.Sequential(
            nn.Linear(8 * 8 * 22, 256),  # 1408 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        
        self.desc_head = nn.Sequential(
            nn.Linear(8 * 8 * 22, 256),  # 1408 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 8)
        )
        
        self.use_quickcumsum = True
    
    def create_frustum(self):
        """Create frustum grid in image plane"""
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)
    
    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Transform frustum to ego coordinates"""
        B, N, _ = trans.shape
        
        # Undo post-transformation
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        
        # Camera to ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        
        return points
    
    def voxel_pooling(self, geom_feats, x):
        """Voxel pooling with QuickCumsum (optimized)"""
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        
        # Flatten
        x = x.reshape(Nprime, C)
        
        # Quantize to voxel grid
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        
        # Filter out-of-bounds
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # Sort by voxel
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        
        # Cumsum trick
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
        
        # Create dense grid
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        
        # Collapse Z
        final = torch.cat(final.unbind(dim=2), 1)
        
        return final
    
    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans):
        """
        Args:
            imgs: (B*N, 3, H, W) - N cameras
            rots, trans, intrins: Camera parameters
            post_rots, post_trans: Data augmentation parameters
            
        Returns:
            bev_seg: (B, 4, 200, 200) - BEV segmentation
            action: (B, 4) - Action prediction
            desc: (B, 8) - Description prediction
        """
        B = self.bsize
        N = imgs.shape[0] // B
        
        # Extract multi-scale features with VoVNet-99
        feat_dict = self.backbone(imgs)
        c3 = feat_dict['c3']  # (B*N, 768, H, W)
        c4 = feat_dict['c4']  # (B*N, 1024, H, W)
        
        # Multi-scale depth prediction (LSS v2)
        depth = self.depth_net(c3, c4)  # (B*N, D, H, W) at C3 resolution
        
        # Camera encoding with depth (use C3 features which match depth resolution)
        cam_feats = self.cam_encode(c3, depth)  # (B*N, C, D, H, W)
        
        # Reshape for voxel pooling
        _, C, D, H, W = cam_feats.shape
        cam_feats = cam_feats.view(B, N, C, D, H, W)
        cam_feats = cam_feats.permute(0, 1, 3, 4, 5, 2)  # (B, N, D, H, W, C)
        
        # Get 3D geometry
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        
        # Voxel pooling to BEV
        bev_feats = self.voxel_pooling(geom, cam_feats)  # (B, C*Z, X, Y)
        
        # BEV encoder with transformer
        bev_seg, bev_refined = self.bev_encoder(bev_feats)
        
        # Crop center region for action/desc (same as original)
        bev_crop = bev_refined[:, :, 60:140, 56:144]  # (B, 256, 80, 88)
        bev_feat = self.bev_post(bev_crop)  # (B, 8, 19, 22)
        bev_feat_flat = bev_feat.flatten(1)  # (B, 8*19*22)
        
        # Action & Description prediction
        action = self.action_head(bev_feat_flat)
        desc = self.desc_head(bev_feat_flat)
        
        return bev_seg, action, desc


def compile_model_vovnet_transformer(bsize, grid_conf, data_aug_conf, outC, vovnet_type='vovnet39', pretrained=True):
    """Factory function to create model
    
    Args:
        vovnet_type (str): 'vovnet39' (default), 'vovnet57', or 'vovnet99'
            - vovnet39: 22M params, ~14.5GB memory, fastest (RECOMMENDED for fair comparison)
            - vovnet57: 36M params, ~15.5GB memory, balanced
            - vovnet99: 54M params, ~16.5GB memory, best performance (not available pretrained)
        pretrained (bool): Load ImageNet pretrained weights (recommended)
    """
    return VoVNetBEVTransformer(bsize, grid_conf, data_aug_conf, outC, vovnet_type=vovnet_type, pretrained=pretrained)


def test_model():
    """Test full model"""
    # Model config
    bsize = 2
    grid_conf = {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
    }
    data_aug_conf = {
        'resize_lim': (0.193, 0.225),
        'final_dim': (128, 352),
        'rot_lim': (-5.4, 5.4),
        'H': 900, 'W': 1600,
        'rand_flip': True,
        'bot_pct_lim': (0.0, 0.22),
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': 6,
    }
    outC = 4
    
    print("=" * 70)
    print("Testing VoVNetV2-BEV-Transformer Models")
    print("=" * 70)
    
    # Only test available models (vovnet99 has no pretrained weights in timm)
    for vovnet_type in ['vovnet39', 'vovnet57']:
        print(f"\n{'='*70}")
        print(f"Model: VoVNetV2-{vovnet_type[6:].upper()} + LSS v2 + Transformer")
        print(f"{'='*70}")
        
        model = compile_model_vovnet_transformer(
            bsize, grid_conf, data_aug_conf, outC=4, vovnet_type=vovnet_type
        )
        
        # Dummy inputs
        imgs = torch.randn(bsize * 6, 3, 128, 352)
        rots = torch.randn(bsize, 6, 3, 3)
        trans = torch.randn(bsize, 6, 3)
        intrins = torch.randn(bsize, 6, 3, 3)
        post_rots = torch.randn(bsize, 6, 3, 3)
        post_trans = torch.randn(bsize, 6, 3)
        
        # Forward pass
        bev_seg, action, desc = model(imgs, rots, trans, intrins, post_rots, post_trans)
        
        print(f"BEV seg shape: {bev_seg.shape}")
        print(f"Action shape: {action.shape}")
        print(f"Description shape: {desc.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable / 1e6:.2f}M")
        
        del model  # Free memory
    
    print("\n" + "=" * 70)
    print("All models tested successfully!")
    print("=" * 70)


if __name__ == '__main__':
    test_model()
