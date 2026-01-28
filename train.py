"""
洪水預測模型訓練腳本
使用 ConvLSTM 進行時序預測
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import json

from config import CONFIG
from dataset import StochasticRainDataset
from model import HydroNetRainOnly
from utils import find_typhoon_data, masked_mse_loss, weighted_flood_loss


# 從 CONFIG 字典取出設定值
train_data_dir = CONFIG.get('train_data_dir', 'train_data')
val_data_dir = CONFIG.get('val_data_dir', 'val_data')
checkpoint_dir = CONFIG.get('save_dir', 'checkpoints')
batch_size = CONFIG.get('batch_size', 2)
learning_rate = CONFIG.get('learning_rate', 1e-4)
weight_decay = CONFIG.get('weight_decay', 1e-5)
epochs = CONFIG.get('num_epochs', 64)
rain_seq_len = CONFIG.get('input_seq_len', 9)
output_steps = CONFIG.get('output_seq_len', 3)
hidden_dim = CONFIG.get('hidden_dim', 64)
num_layers = CONFIG.get('num_layers', 2)
flood_weight = CONFIG.get('flood_weight', 10.0)
early_stopping_patience = CONFIG.get('early_stopping_patience', 10)


def init_weights(m):
    """初始化模型權重"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def save_validation_comparison(pred, target, mask, epoch, save_dir, sample_idx=0):
    """
    保存驗證階段的預測與真實值比較圖
    
    Args:
        pred: 預測值 [B, 3, 1, H, W] - 對應 t+1, t+2, t+3
        target: 真實值 [B, 3, 1, H, W]
        mask: 遮罩 [B, 3, 1, H, W]
        epoch: 當前 epoch
        save_dir: 保存目錄
        sample_idx: 要視覺化的樣本索引
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 取出指定樣本，轉為 numpy
    pred_np = pred[sample_idx].detach().cpu().numpy()      # [3, 1, H, W]
    target_np = target[sample_idx].detach().cpu().numpy()  # [3, 1, H, W]
    mask_np = mask[sample_idx].detach().cpu().numpy()      # [3, 1, H, W]
    
    # 創建 2x3 的圖表：上排預測，下排真實值
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    titles = ['t+1', 't+2', 't+3']
    
    # 計算統一的顏色範圍
    all_values = np.concatenate([pred_np.flatten(), target_np.flatten()])
    vmin = np.percentile(all_values, 1)
    vmax = np.percentile(all_values, 99)
    
    for i in range(3):
        # 應用遮罩
        pred_masked = np.ma.masked_where(mask_np[i, 0] == 0, pred_np[i, 0])
        target_masked = np.ma.masked_where(mask_np[i, 0] == 0, target_np[i, 0])
        
        # 上排：預測值
        im1 = axes[0, i].imshow(pred_masked, cmap='Blues', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'Prediction {titles[i]}')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # 下排：真實值
        im2 = axes[1, i].imshow(target_masked, cmap='Blues', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Ground Truth {titles[i]}')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Epoch {epoch} - Validation Comparison', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'val_comparison_epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  驗證比較圖已保存: {save_path}")


def format_time(seconds):
    """將秒數格式化為 HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def plot_training_curves(history, save_dir):
    """
    繪製訓練曲線並保存
    
    Args:
        history: 訓練歷史字典
        save_dir: 保存目錄
    """
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 曲線
    axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 標記最佳驗證損失
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = min(history['val_loss'])
    axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
    axes[0].scatter([best_epoch], [best_val], color='g', s=100, zorder=5)
    
    # 學習率曲線
    axes[1].plot(epochs_range, history['learning_rate'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"訓練曲線已保存: {save_path}")


def train():
    """主訓練函數"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 準備數據
    print("\n載入訓練數據...")
    train_sequences = find_typhoon_data(train_data_dir)
    print(f"訓練序列數: {len(train_sequences)}")
    
    print("載入驗證數據...")
    val_sequences = find_typhoon_data(val_data_dir)
    print(f"驗證序列數: {len(val_sequences)}")
    
    # 創建數據集
    train_dataset = StochasticRainDataset(train_sequences, CONFIG)
    val_dataset = StochasticRainDataset(val_sequences, CONFIG)
    
    # 創建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\n訓練批次數: {len(train_loader)}")
    print(f"驗證批次數: {len(val_loader)}")
    
    # 創建模型
    model = HydroNetRainOnly(output_steps=output_steps).to(device)
    
    model.apply(init_weights)
    
    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型參數量: {total_params:,} (可訓練: {trainable_params:,})")
    
    # 損失函數和優化器 (使用 masked_mse_loss 函數)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 學習率調度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # 混合精度訓練
    scaler = torch.amp.GradScaler('cuda')
    
    # 訓練狀態
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 訓練歷史記錄
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_flood_mse': [],
        'learning_rate': [],
        'epoch_time': []
    }
    
    # 創建保存目錄
    os.makedirs(checkpoint_dir, exist_ok=True)
    vis_dir = os.path.join('visualizations', 'validation')
    os.makedirs(vis_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("開始訓練")
    print("=" * 60)
    
    # 記錄整體訓練開始時間
    training_start_time = time.time()
    epoch_times = []  # 儲存每個 epoch 的耗時
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # ==================== 訓練階段 ====================
        model.train()
        train_loss = 0.0
        
        for batch_idx, (rain_input, flood_target, mask) in enumerate(train_loader):
            rain_input = rain_input.to(device, non_blocking=True)
            flood_target = flood_target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred = model(rain_input)
                loss = weighted_flood_loss(pred, flood_target, mask, flood_weight=flood_weight)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}")
        
        train_loss /= len(train_loader)
        
        # ==================== 驗證階段 ====================
        model.eval()
        val_loss = 0.0
        val_pred_for_vis = None
        val_target_for_vis = None
        val_mask_for_vis = None
        
        # 額外指標追蹤
        val_mae = 0.0
        val_flood_mse = 0.0  # 只計算有淹水變化區域的 MSE
        
        with torch.no_grad():
            for batch_idx, (rain_input, flood_target, mask) in enumerate(val_loader):
                rain_input = rain_input.to(device, non_blocking=True)
                flood_target = flood_target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    pred = model(rain_input)
                    loss = weighted_flood_loss(pred, flood_target, mask, flood_weight=flood_weight)
                
                val_loss += loss.item()
                
                # 計算 MAE
                mae = (torch.abs(pred - flood_target) * mask).sum() / (mask.sum() + 1e-6)
                val_mae += mae.item()
                
                # 計算淹水區域 MSE (閾值需要配合 target_scale)
                target_scale = CONFIG.get('target_scale', 1.0)
                flood_threshold = 0.001 * target_scale  # 縮放後的閾值
                flood_mask = (torch.abs(flood_target) > flood_threshold).float() * mask
                if flood_mask.sum() > 0:
                    flood_mse = ((pred - flood_target) ** 2 * flood_mask).sum() / (flood_mask.sum() + 1e-6)
                    val_flood_mse += flood_mse.item()
                
                # 保存第一個批次用於視覺化
                if batch_idx == 0:
                    val_pred_for_vis = pred
                    val_target_for_vis = flood_target
                    val_mask_for_vis = mask
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_flood_mse /= len(val_loader)
        
        # 更新學習率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 計算時間統計
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # 記錄訓練歷史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_flood_mse'].append(val_flood_mse)
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        
        # 每個 epoch 保存一次歷史記錄
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        # 計算剩餘時間
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        
        elapsed_total = time.time() - training_start_time
        
        # 輸出訓練資訊
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f} | MAE: {val_mae:.6f} | Flood MSE: {val_flood_mse:.6f}")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Epoch 耗時: {format_time(epoch_time)} | "
              f"已訓練: {format_time(elapsed_total)} | "
              f"預估剩餘: {format_time(eta_seconds)}")
        
        # 每個 epoch 都保存驗證比較圖
        if val_pred_for_vis is not None:
            save_validation_comparison(
                val_pred_for_vis,
                val_target_for_vis,
                val_mask_for_vis,
                epoch,
                vis_dir
            )
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"  ★ 最佳模型已保存 (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  Early stopping counter: {patience_counter}/{early_stopping_patience}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n早停觸發！驗證損失已 {early_stopping_patience} 個 epoch 未改善")
            break
        
        # 定期保存 checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"  Checkpoint 已保存: {checkpoint_path}")
    
    # 訓練結束統計
    total_time = time.time() - training_start_time
    print("\n" + "=" * 60)
    print("訓練完成")
    print("=" * 60)
    print(f"總訓練時間: {format_time(total_time)}")
    print(f"最佳驗證損失: {best_val_loss:.6f}")
    print(f"平均每 Epoch 耗時: {format_time(sum(epoch_times) / len(epoch_times))}")
    
    # 繪製訓練曲線
    plot_training_curves(history, checkpoint_dir)


if __name__ == '__main__':
    train()
