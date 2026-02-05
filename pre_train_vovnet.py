"""
Pre-training script for VoVNet + LSS v2
Following paper: 60 epochs, batch_size=12, train encoder+BEV on nuScenes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import argparse
from tqdm import tqdm

from src.data_pretrain import compile_data
from src.model_vovnet_transformer import VoVNetBEVTransformer
from src.vovnet_timm import VoVNetV2


class PreTrainingModel(nn.Module):
    """Pre-training model: VoVNet encoder + LSS v2 + BEV decoder (no action/desc heads)"""
    def __init__(self, bsize, grid_conf, data_aug_conf, outC=4, vovnet_type='vovnet39', pretrained=True):
        super(PreTrainingModel, self).__init__()
        
        # Create full model
        from src.model_vovnet_transformer import compile_model_vovnet_transformer
        full_model = compile_model_vovnet_transformer(
            bsize, grid_conf, data_aug_conf, outC=outC,
            vovnet_type=vovnet_type, pretrained=pretrained
        )
        
        # Extract only encoder and BEV components (no action/desc heads)
        self.backbone = full_model.backbone
        self.depth_net = full_model.depth_net
        self.cam_encode = full_model.cam_encode
        self.bev_encoder = full_model.bev_encoder
        
        # For geometry calculations
        self.frustum = full_model.frustum
        self.dx = full_model.dx
        self.bx = full_model.bx
        self.nx = full_model.nx
        self.D = full_model.D
        self.C = full_model.C
        self.downsample = full_model.downsample
        self.use_quickcumsum = full_model.use_quickcumsum
    
    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Copy from full model"""
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
        """Copy from full model"""
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        
        # Flatten
        x = x.reshape(Nprime, C)
        
        # Batch and flatten geometry
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix, device=x.device, dtype=torch.long) 
                             for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        
        # Filter out of bounds
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
             & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
             & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # QuickCumsum
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
              + geom_feats[:, 1] * (self.nx[2] * B) \
              + geom_feats[:, 2] * B \
              + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        
        # Reshape to BEV
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        
        # Collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # (B, C*Z, H, W)
        
        return final
    
    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans):
        """Forward pass (BEV segmentation only)"""
        # Note: imgs comes in as [B, N, C, H, W] from dataloader
        # Save B and N for later reshaping
        if len(imgs.shape) == 5:
            B, N, C, H, W = imgs.shape
            imgs = imgs.view(B * N, C, H, W)
        else:
            # Already reshaped, infer B and N from rots
            B = rots.shape[0]
            N = rots.shape[1]
        
        # Extract multi-scale features (backbone returns dict)
        features = self.backbone(imgs)
        c3 = features['c3']
        c4 = features['c4']
        
        # Multi-scale depth prediction
        depth = self.depth_net(c3, c4)
        
        # Camera encoding with depth
        cam_features = self.cam_encode(c3, depth)  # (B*N, C, D, H, W)
        
        # Permute to (B*N, D, H, W, C) for voxel pooling
        cam_features = cam_features.permute(0, 2, 3, 4, 1)  # (B*N, D, H, W, C)
        
        # Reshape cam_features from (B*N, D, H, W, C) to (B, N, D, H, W, C)
        BN, D, H_feat, W_feat, C_feat = cam_features.shape
        cam_features = cam_features.view(B, N, D, H_feat, W_feat, C_feat)
        
        # Get geometry (frustum shape must match cam_features H, W)
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        
        # Debug: Check shapes
        # print(f"DEBUG: cam_features shape: {cam_features.shape}")
        # print(f"DEBUG: geom shape: {geom.shape}")
        # print(f"DEBUG: frustum shape: {self.frustum.shape}")
        
        # Verify geom and cam_features have matching spatial dimensions
        if geom.shape[2:5] != cam_features.shape[2:5]:
            raise RuntimeError(
                f"Geometry shape {geom.shape} doesn't match cam_features {cam_features.shape}. "
                f"Frustum H,W ({self.frustum.shape[1]},{self.frustum.shape[2]}) vs "
                f"cam_features H,W ({H_feat},{W_feat})"
            )
        
        # Voxel pooling to BEV
        bev_features = self.voxel_pooling(geom, cam_features)
        
        # BEV encoding with transformer
        bev_seg, _ = self.bev_encoder(bev_features)
        
        return bev_seg


def cumsum_trick(x, geom_feats, ranks):
    """QuickCumsum helper"""
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    
    return x, geom_feats
        
    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans):
        """Forward pass (BEV segmentation only)"""
        # Camera encoding
        cam_features = self.camencode(imgs, rots, trans, intrins, post_rots, post_trans)
        
        # BEV encoding
        bev_features = self.bevencode(cam_features)
        
        # BEV segmentation
        bev_seg = self.seg_decoder(bev_features)
        
        return bev_seg


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, criterion, device, epoch):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Pre-train Epoch {epoch}")
    
    for batch in pbar:
        imgs = batch[0].to(device)
        rots = batch[1].to(device)
        trans = batch[2].to(device)
        intrins = batch[3].to(device)
        post_rots = batch[4].to(device)
        post_trans = batch[5].to(device)
        binimgs = batch[6].to(device)
        
        # Note: imgs is [B, N, C, H, W], model will handle reshaping
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            bev_pred = model(imgs, rots, trans, intrins, post_rots, post_trans)
            loss = criterion(bev_pred, binimgs)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            imgs = batch[0].to(device)
            rots = batch[1].to(device)
            trans = batch[2].to(device)
            intrins = batch[3].to(device)
            post_rots = batch[4].to(device)
            post_trans = batch[5].to(device)
            binimgs = batch[6].to(device)
            
            # Note: imgs is [B, N, C, H, W], model will handle reshaping
            
            # Forward pass
            with autocast():
                bev_pred = model(imgs, rots, trans, intrins, post_rots, post_trans)
                loss = criterion(bev_pred, binimgs)
            
            total_loss += loss.item()
            all_preds.append(bev_pred.cpu())
            all_gts.append(binimgs.cpu())
    
    # Compute IoU
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    
    pred_classes = all_preds.argmax(1)
    
    # Calculate mIoU
    num_classes = all_preds.shape[1]
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred_classes == cls)
        gt_mask = (all_gts == cls)
        intersection = (pred_mask & gt_mask).sum().float()
        union = (pred_mask | gt_mask).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    
    miou = np.mean(ious) if ious else 0.0
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, miou


def main():
    parser = argparse.ArgumentParser(description="Pre-training VoVNet + LSS v2")
    
    # Model config
    parser.add_argument('--vovnet_type', default='vovnet39', choices=['vovnet39', 'vovnet57'],
                        help='VoVNet backbone type')
    parser.add_argument('--pretrained', default=True, type=bool,
                        help='Use ImageNet pretrained weights')
    
    # Training config (matching paper)
    parser.add_argument('--epochs', default=60, type=int, help='Paper: 60 epochs for pre-training')
    parser.add_argument('--bsize', default=12, type=int, help='Paper: batch_size=12 for pre-training')
    parser.add_argument('--lr', default=1e-4, type=float, help='Paper: 1e-4')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='Paper: 1e-8')
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    # Data config
    parser.add_argument('--dataroot', default='./data/trainval/nu-A2D-20260129T100537Z-3-001/nu-A2D',
                        help='Path to dataset')
    parser.add_argument('--version', default='trainval', help='Dataset version')
    
    # Save config
    parser.add_argument('--save_dir', default='./pretrain_vovnet',
                        help='Directory to save pre-trained weights')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Grid and augmentation config
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
    
    # Create model
    print(f"Creating pre-training model: VoVNet-{args.vovnet_type[6:].upper()} + LSS v2")
    model = PreTrainingModel(
        args.bsize, grid_conf, data_aug_conf, outC=4,
        vovnet_type=args.vovnet_type, pretrained=args.pretrained
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    # Load data
    print("Loading data...")
    trainloader, valloader = compile_data(
        args.version, args.dataroot,
        data_aug_conf=data_aug_conf,
        grid_conf=grid_conf,
        bsz=args.bsize,
        nworkers=args.num_workers,
        parser_name='segmentationdata'
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Optimizer (Paper: Adam, not AdamW)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    num_training_steps = args.epochs * len(trainloader)
    num_warmup_steps = args.warmup_epochs * len(trainloader)
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    best_miou = 0
    best_epoch = 0
    
    print("=" * 60)
    print("Starting pre-training...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.bsize}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, trainloader, optimizer, scheduler, scaler, criterion, device, epoch
        )
        
        print(f"\nEpoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}")
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            val_loss, val_miou = validate(model, valloader, criterion, device)
            
            print(f"Validation - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")
            
            # Save best model
            if val_miou > best_miou:
                best_miou = val_miou
                best_epoch = epoch
                
                # Save encoder + BEV weights only (for loading into main training)
                checkpoint = {
                    'epoch': epoch,
                    'backbone_state_dict': model.backbone.state_dict(),
                    'depth_net_state_dict': model.depth_net.state_dict(),
                    'cam_encode_state_dict': model.cam_encode.state_dict(),
                    'bev_encoder_state_dict': model.bev_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'miou': best_miou,
                }
                
                save_path = os.path.join(args.save_dir, 'best_pretrained.pth')
                torch.save(checkpoint, save_path)
                print(f"✓ Saved best pre-trained model (mIoU: {best_miou:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'backbone_state_dict': model.backbone.state_dict(),
                'depth_net_state_dict': model.depth_net.state_dict(),
                'cam_encode_state_dict': model.cam_encode.state_dict(),
                'bev_encoder_state_dict': model.bev_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_path = os.path.join(args.save_dir, f'pretrained_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
        
        print("=" * 60)
    
    print(f"\nPre-training completed!")
    print(f"Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
    print(f"Pre-trained weights saved to: {args.save_dir}/best_pretrained.pth")


if __name__ == '__main__':
    main()
