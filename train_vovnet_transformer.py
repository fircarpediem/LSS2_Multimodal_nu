"""
Training script for VoVNet-99 + LSS v2 + Transformer
Optimized for RTX 3090 24GB VRAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
import wandb

from src.data import compile_data
from src.model_vovnet_transformer import compile_model_vovnet_transformer
from src.tools import MultiLoss, get_val_info_new


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
    """Multi-task loss for BEV segmentation + Action + Description"""
    def __init__(self, device='cuda'):
        super().__init__()
        # BEV segmentation weights (4 classes)
        bev_weights = torch.FloatTensor([1, 10, 5, 10]).to(device)
        self.bev_loss_fn = nn.CrossEntropyLoss(weight=bev_weights)
        
        # Action classification weights (4 classes)
        self.action_weights = torch.FloatTensor([1, 5, 5, 5]).to(device)
        
        # Description classification weights (8 classes)
        self.desc_weights = torch.FloatTensor([1, 5, 5, 5, 1, 1, 1, 1]).to(device)
    
    def forward(self, bev_pred, action_pred, desc_pred, bev_gt, action_gt, desc_gt):
        """
        Args:
            bev_pred: (B, 4, H, W) BEV segmentation logits
            action_pred: (B, 4) Action logits
            desc_pred: (B, 8) Description logits
            bev_gt: (B, H, W) BEV ground truth
            action_gt: (B, 4) Action labels
            desc_gt: (B, 8) Description labels
        Returns:
            loss_total, loss_bev, loss_action, loss_desc
        """
        # BEV loss
        loss_bev = self.bev_loss_fn(bev_pred, bev_gt)
        
        # Action loss (binary cross entropy)
        loss_action = F.binary_cross_entropy_with_logits(
            action_pred, action_gt, weight=self.action_weights
        )
        
        # Description loss (binary cross entropy)
        loss_desc = F.binary_cross_entropy_with_logits(
            desc_pred, desc_gt, weight=self.desc_weights
        )
        
        # Total loss
        loss_total = loss_bev + loss_action + loss_desc
        
        return loss_total, loss_bev, loss_action, loss_desc


def get_parameter_groups(model, lr, backbone_lr_mult=0.1):
    """Layer-wise learning rate decay"""
    param_groups = [
        # Backbone (VoVNet) with lower LR
        {
            'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad],
            'lr': lr * backbone_lr_mult,
            'name': 'backbone'
        },
        # Depth network
        {
            'params': [p for n, p in model.named_parameters() if 'depth_net' in n and p.requires_grad],
            'lr': lr,
            'name': 'depth_net'
        },
        # Transformer
        {
            'params': [p for n, p in model.named_parameters() if 'transformer' in n and p.requires_grad],
            'lr': lr,
            'name': 'transformer'
        },
        # Task heads
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(x in n for x in ['backbone', 'depth_net', 'transformer']) and p.requires_grad],
            'lr': lr,
            'name': 'heads'
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
        with autocast():
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
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update learning rate
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
            with autocast():
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
    val_info = get_val_info_new(
        all_bev_preds, all_action_preds, all_desc_preds,
        all_bev_gts, all_action_gts, all_desc_gts
    )
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, val_info


def main():
    # ==================== Configuration ====================
    # Model config
    # For fair comparison with EfficientNet-B4 (19M): use vovnet39 (22M)
    # For maximum performance: use vovnet57 (36M)
    vovnet_type = 'vovnet39'  # Options: 'vovnet39' (22M, fair), 'vovnet57' (36M, best)
    pretrained = True  # Load ImageNet pretrained weights
    bsize = 5  # vovnet39 can handle batch size 5-6 on 3090
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
    
    # Training config
    epochs = 80
    lr = 2e-4
    weight_decay = 1e-4
    warmup_epochs = 5
    num_workers = 4
    save_dir = './checkpoints_vovnet_transformer'
    
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
    
    # Loss function
    criterion = MultiTaskLoss(device=device)
    
    # Optimizer with layer-wise LR
    param_groups = get_parameter_groups(model, lr, backbone_lr_mult=0.1)
    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    num_training_steps = epochs * len(trainloader)
    num_warmup_steps = warmup_epochs * len(trainloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Initialize Weights & Biases
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
    wandb.finish()


if __name__ == '__main__':
    main()
