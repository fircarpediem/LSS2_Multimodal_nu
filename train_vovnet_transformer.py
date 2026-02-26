"""
Training script for VoVNet-99 + LSS v2 + Transformer
Optimized for RTX 3090 24GB VRAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm

# Optional: Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Training will proceed without logging to W&B.")

from src.data import compile_data
from src.model_vovnet_transformer import compile_model_vovnet_transformer


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


class MultiTaskLoss(nn.Module):
    """Multi-task loss for BEV segmentation + Action + Description
    
    Fixed issues:
    1. Remove per-class weights for BCE (not supported)
    2. Add task balancing weights
    3. Use pos_weight for class imbalance in BCE
    """
    def __init__(self, device='cuda', bev_weight=1.0, action_weight=0.5, desc_weight=0.5):
        super().__init__()
        
        # Task balancing weights
        self.bev_weight = bev_weight
        self.action_weight = action_weight
        self.desc_weight = desc_weight
        
        # BEV segmentation: class weights for CrossEntropyLoss
        bev_class_weights = torch.FloatTensor([1.0, 10.0, 5.0, 10.0]).to(device)
        self.bev_loss_fn = nn.CrossEntropyLoss(weight=bev_class_weights)
        
        # Action: pos_weight for class imbalance (not per-class weight!)
        # Positive class more important (5x weight)
        self.action_pos_weight = torch.FloatTensor([5.0, 5.0, 5.0, 5.0]).to(device)
        
        # Description: pos_weight for class imbalance
        # First 4 classes more important
        self.desc_pos_weight = torch.FloatTensor([5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]).to(device)
    
    def forward(self, bev_pred, action_pred, desc_pred, bev_gt, action_gt, desc_gt):
        """
        Args:
            bev_pred: (B, 4, H, W) BEV segmentation logits
            action_pred: (B, 4) Action logits
            desc_pred: (B, 8) Description logits
            bev_gt: (B, H, W) BEV ground truth
            action_gt: (B, 4) Action labels (0 or 1)
            desc_gt: (B, 8) Description labels (0 or 1)
        Returns:
            loss_total, loss_bev, loss_action, loss_desc
        """
        # BEV loss (spatial segmentation)
        loss_bev = self.bev_loss_fn(bev_pred, bev_gt)
        
        # Action loss (binary classification with pos_weight for imbalance)
        loss_action = F.binary_cross_entropy_with_logits(
            action_pred, 
            action_gt.float(),
            pos_weight=self.action_pos_weight
        )
        
        # Description loss (binary classification with pos_weight)
        loss_desc = F.binary_cross_entropy_with_logits(
            desc_pred, 
            desc_gt.float(),
            pos_weight=self.desc_pos_weight
        )
        
        # Total loss with task balancing
        loss_total = (
            self.bev_weight * loss_bev + 
            self.action_weight * loss_action + 
            self.desc_weight * loss_desc
        )
        
        return loss_total, loss_bev, loss_action, loss_desc


def get_parameter_groups(model, lr, backbone_lr_mult=0.1):
    """Layer-wise learning rate decay
    
    Pretrained modules (lower LR):
    - backbone, depth_net, cam_encode, bev_encoder
    
    New modules (higher LR):
    - vovnet_adapter, sceneunder, embeders, predictors, bev_post
    """
    param_groups = [
        # Backbone (VoVNet) with lower LR (pretrained)
        {
            'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad],
            'lr': lr * backbone_lr_mult,
            'name': 'backbone'
        },
        # Pretrained BEV branch modules with lower LR
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(x in n for x in ['depth_net', 'cam_encode', 'bev_encoder']) and p.requires_grad],
            'lr': lr * backbone_lr_mult,  # Lower LR for pretrained modules
            'name': 'pretrained_bev'
        },
        # New TXT branch and task heads with higher LR (trained from scratch)
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(x in n for x in ['backbone', 'depth_net', 'cam_encode', 'bev_encoder']) 
                      and p.requires_grad],
            'lr': lr,  # Higher LR for new modules
            'name': 'new_modules'
        }
    ]
    
    return param_groups


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    bev_loss_total = 0
    action_loss_total = 0
    desc_loss_total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        imgs = batch[0].to(device)
        rots = batch[1].to(device)
        trans = batch[2].to(device)
        intrins = batch[3].to(device)
        post_rots = batch[4].to(device)
        post_trans = batch[5].to(device)
        binimgs = batch[6].to(device)
        action_labels = batch[7].to(device)
        desc_labels = batch[8].to(device)
        
        # Reshape imgs from [B, N, C, H, W] to [B*N, C, H, W]
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        
        # Mixed precision training
        with autocast(device_type='cuda', dtype=torch.float16):
            # Forward pass
            bev_pred, action_pred, desc_pred = model(
                imgs, rots, trans, intrins, post_rots, post_trans
            )
            
            # Compute loss
            loss, bev_loss, action_loss, desc_loss = criterion(
                bev_pred, action_pred, desc_pred,
                binimgs, action_labels, desc_labels
            )
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # Optimizer step BEFORE scheduler (fix warning)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Logging
        total_loss += loss.item()
        bev_loss_total += bev_loss.item()
        action_loss_total += action_loss.item()
        desc_loss_total += desc_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'bev': f'{bev_loss.item():.4f}',
            'act': f'{action_loss.item():.4f}',
            'desc': f'{desc_loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_bev = bev_loss_total / len(dataloader)
    avg_action = action_loss_total / len(dataloader)
    avg_desc = desc_loss_total / len(dataloader)
    
    return avg_loss, avg_bev, avg_action, avg_desc


def validate(model, dataloader, criterion, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    
    all_bev_preds = []
    all_bev_gts = []
    all_action_preds = []
    all_action_gts = []
    all_desc_preds = []
    all_desc_gts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            imgs = batch[0].to(device)
            rots = batch[1].to(device)
            trans = batch[2].to(device)
            intrins = batch[3].to(device)
            post_rots = batch[4].to(device)
            post_trans = batch[5].to(device)
            binimgs = batch[6].to(device)
            action_labels = batch[7].to(device)
            desc_labels = batch[8].to(device)
            
            # Reshape imgs from [B, N, C, H, W] to [B*N, C, H, W]
            B, N, C, H, W = imgs.shape
            imgs = imgs.view(B * N, C, H, W)
            
            # Forward pass
            with autocast(device_type='cuda', dtype=torch.float16):
                bev_pred, action_pred, desc_pred = model(
                    imgs, rots, trans, intrins, post_rots, post_trans
                )
                
                loss, _, _, _ = criterion(
                    bev_pred, action_pred, desc_pred,
                    binimgs, action_labels, desc_labels
                )
            
            total_loss += loss.item()
            
            # Collect predictions
            all_bev_preds.append(bev_pred.cpu())
            all_bev_gts.append(binimgs.cpu())
            all_action_preds.append(action_pred.cpu())
            all_action_gts.append(action_labels.cpu())
            all_desc_preds.append(desc_pred.cpu())
            all_desc_gts.append(desc_labels.cpu())
    
    # Concatenate all predictions
    all_bev_preds = torch.cat(all_bev_preds, dim=0)
    all_bev_gts = torch.cat(all_bev_gts, dim=0)
    all_action_preds = torch.cat(all_action_preds, dim=0)
    all_action_gts = torch.cat(all_action_gts, dim=0)
    all_desc_preds = torch.cat(all_desc_preds, dim=0)
    all_desc_gts = torch.cat(all_desc_gts, dim=0)
    
    # Compute metrics
    val_info = compute_metrics_from_predictions(
        all_bev_preds, all_action_preds, all_desc_preds,
        all_bev_gts, all_action_gts, all_desc_gts
    )
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, val_info


def compute_metrics_from_predictions(bev_preds, action_preds, desc_preds, 
                                      bev_gts, action_gts, desc_gts):
    """
    Compute evaluation metrics from collected predictions
    
    Args:
        bev_preds: (N, 4, H, W) BEV segmentation logits
        action_preds: (N, 4) Action logits
        desc_preds: (N, 8) Description logits
        bev_gts: (N, H, W) BEV ground truth
        action_gts: (N, 4) Action labels
        desc_gts: (N, 8) Description labels
    
    Returns:
        Dictionary with metrics: {'iou', 'action_f1', 'desc_f1'}
    """
    from src.tools import ConfusionMatrix
    from sklearn.metrics import f1_score
    
    # 1. BEV Segmentation mIoU
    pred_classes = bev_preds.argmax(1)
    confmat = ConfusionMatrix(4)
    confmat.update(bev_gts.flatten(), pred_classes.flatten())
    _, _, ious = confmat.compute()
    bev_iou = ious.mean().item()
    
    # 2. Action F1 Score
    action_preds_binary = (torch.sigmoid(action_preds) > 0.5).cpu().numpy()
    action_gts_numpy = action_gts.cpu().numpy()
    action_f1 = f1_score(action_gts_numpy.flatten(), 
                         action_preds_binary.flatten(), 
                         average='macro', 
                         zero_division=0)
    
    # 3. Description F1 Score  
    desc_preds_binary = (torch.sigmoid(desc_preds) > 0.5).cpu().numpy()
    desc_gts_numpy = desc_gts.cpu().numpy()
    desc_f1 = f1_score(desc_gts_numpy.flatten(), 
                       desc_preds_binary.flatten(), 
                       average='macro', 
                       zero_division=0)
    
    return {
        'iou': bev_iou,
        'action_f1': action_f1,
        'desc_f1': desc_f1
    }


def main():
    # ==================== Configuration ====================
    # Model config
    # For fair comparison with EfficientNet-B4 (19M): use vovnet39 (22M)
    # For maximum performance: use vovnet57 (36M)
    vovnet_type = 'vovnet39'  # Options: 'vovnet39' (22M, fair), 'vovnet57' (36M, best)
    pretrained = True  # Load ImageNet pretrained weights (mimics paper's pre-training)
    bsize = 8  # Paper: batch_size=8 for main training
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
    
    # Training config (matching paper)
    epochs = 60  # Paper: 60 epochs for main training
    lr = 1e-4  # Paper: 1×10^-4
    weight_decay = 1e-8  # Paper: 1×10^-8 (very small!)
    warmup_epochs = 5
    num_workers = 4
    save_dir = './checkpoints_vovnet_transformer'
    
    # Pre-training config
    use_pretrained_weights = True  # Load pre-trained encoder+BEV weights
    pretrained_path = './pretrain_vovnet/best_pretrained.pth'  # Path to pre-trained checkpoint
    
    # Data paths  
    dataroot = './data/trainval/nu-A2D-20260129T100537Z-3-001/nu-A2D'
    version = 'trainval'
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # ==================== Setup ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating VoVNetV2-{vovnet_type[6:].upper()} + LSS v2 + Transformer...")
    print(f"Pretrained: {pretrained}")
    model = compile_model_vovnet_transformer(
        bsize, grid_conf, data_aug_conf, outC=4, 
        vovnet_type=vovnet_type, pretrained=pretrained
    )
    model = model.to(device)
    
    # Load pre-trained weights if available
    if use_pretrained_weights and os.path.exists(pretrained_path):
        print(f"Loading pre-trained weights from: {pretrained_path}")
        pretrained_ckpt = torch.load(pretrained_path, map_location=device)
        
        # Load encoder + BEV weights (skip action/desc heads)
        model.backbone.load_state_dict(pretrained_ckpt['backbone_state_dict'])
        model.depth_net.load_state_dict(pretrained_ckpt['depth_net_state_dict'])
        model.cam_encode.load_state_dict(pretrained_ckpt['cam_encode_state_dict'])
        model.bev_encoder.load_state_dict(pretrained_ckpt['bev_encoder_state_dict'])
        
        print(f"✓ Loaded pre-trained encoder + BEV (mIoU: {pretrained_ckpt.get('miou', 0):.4f})")
        print("  Action and Description heads initialized randomly")
    elif use_pretrained_weights:
        print(f"⚠️  Pre-trained weights not found at {pretrained_path}")
        print("  Training from ImageNet pretrained weights only")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Create dataloaders
    print("Loading data...")
    trainloader, valloader = compile_data(
        version, dataroot, data_aug_conf=data_aug_conf,
        grid_conf=grid_conf, bsz=bsize, nworkers=num_workers,
        parser_name='segmentationdata'
    )
    
    # Loss function with task balancing
    # bev_weight=1.0: Spatial segmentation (main task)
    # action_weight=0.5: Action prediction (important for safety)
    # desc_weight=0.5: Description (scene understanding)
    criterion = MultiTaskLoss(
        device=device,
        bev_weight=1.0,
        action_weight=0.5,
        desc_weight=0.5
    )
    
    # Optimizer with layer-wise LR (Paper uses Adam, not AdamW)
    param_groups = get_parameter_groups(model, lr, backbone_lr_mult=0.1)
    optimizer = optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)  # Paper: Adam
    
    # Learning rate scheduler
    num_training_steps = epochs * len(trainloader)
    num_warmup_steps = warmup_epochs * len(trainloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6
    )
    
    # Mixed precision scaler (updated API)
    scaler = GradScaler('cuda')
    
    # Initialize Weights & Biases (optional)
    if WANDB_AVAILABLE:
        wandb.init(
            project="Multimodal-XAD",
            name=f"VoVNet-{vovnet_type[6:]}_LSS-v2_Transformer",
            config={
                "architecture": f"VoVNet-{vovnet_type[6:]} + LSS v2 + Transformer",
                "dataset": "nu-A2D",
                "epochs": epochs,
                "batch_size": bsize,
                "learning_rate": lr,
                "backbone_lr_mult": 0.1,
                "weight_decay": weight_decay,
                "warmup_epochs": warmup_epochs,
                "total_params_M": f"{total_params / 1e6:.2f}",
                "trainable_params_M": f"{trainable_params / 1e6:.2f}",
                "vovnet_type": vovnet_type,
                "pretrained": pretrained,
                "grid_conf": grid_conf,
                "data_aug_conf": data_aug_conf,
            }
        )
        wandb.watch(model, log="all", log_freq=100)
    
    # ==================== Training Loop ====================
    best_miou = 0
    best_epoch = 0
    
    print("Starting training...")
    print(f"Total epochs: {epochs}")
    print(f"Batch size: {bsize}")
    print(f"Initial LR: {lr}")
    print("=" * 60)
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_bev, train_action, train_desc = train_one_epoch(
            model, trainloader, optimizer, scheduler, scaler, criterion, device, epoch
        )
        
        print(f"\nEpoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} "
              f"(BEV: {train_bev:.4f}, Act: {train_action:.4f}, Desc: {train_desc:.4f})")
        
        # Log training metrics
        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/bev_loss": train_bev,
                "train/action_loss": train_action,
                "train/desc_loss": train_desc,
                "train/lr": optimizer.param_groups[0]['lr'],
            })
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            val_loss, val_info = validate(model, valloader, criterion, device)
            
            bev_iou = val_info['iou']
            action_f1 = val_info['action_f1']
            desc_f1 = val_info['desc_f1']
            
            print(f"Validation - Loss: {val_loss:.4f}")
            print(f"BEV mIoU: {bev_iou:.4f}")
            print(f"Action F1: {action_f1:.4f}")
            print(f"Description F1: {desc_f1:.4f}")
            
            # Log validation metrics
            if WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "val/loss": val_loss,
                    "val/bev_miou": bev_iou,
                    "val/action_f1": action_f1,
                    "val/desc_f1": desc_f1,
                })
            
            # Save best model
            if bev_iou > best_miou:
                best_miou = bev_iou
                best_epoch = epoch
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_miou': best_miou,
                    'val_info': val_info,
                }
                
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"✓ Saved best model (mIoU: {best_miou:.4f})")
                
                # Log best model to wandb
                if WANDB_AVAILABLE:
                    wandb.run.summary["best_miou"] = best_miou
                    wandb.run.summary["best_epoch"] = best_epoch
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        print("=" * 60)
    
    print(f"\nTraining completed!")
    print(f"Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
    
    # Finish wandb run
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()
