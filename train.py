"""
洪水預測模型訓練腳本
使用 ConvLSTM 進行時序預測
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import json

# 設定 matplotlib 支援中文顯示並避免負號字型缺字警告
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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


def save_validation_comparison(all_preds, all_targets, all_masks, all_inputs, epoch, save_dir):
    """
    保存驗證階段的預測與真實值比較圖 (第一筆、中間、最後一筆)
    包含輸入降雨序列與輸入淹水圖
    
    Args:
        all_preds: 所有預測值列表，每個元素 [B, 3, 1, H, W]
        all_targets: 所有真實值列表
        all_masks: 所有遮罩列表
        all_inputs: 所有輸入列表 [B, 9, 2, H, W] (2通道: 降雨+初始淹水)
        epoch: 當前 epoch
        save_dir: 保存目錄
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 合併所有 batch 的資料
    preds = torch.cat(all_preds, dim=0)    # [N, 3, 1, H, W]
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)
    inputs = torch.cat(all_inputs, dim=0)  # [N, 9, 2, H, W]
    
    n_samples = preds.shape[0]
    
    # 選擇第一筆、中間、最後一筆
    sample_indices = [0, n_samples // 2, n_samples - 1]
    sample_names = ['first', 'middle', 'last']
    
    # 還原 target_scale 縮放
    target_scale = CONFIG.get('target_scale', 1.0)
    
    for sample_idx, sample_name in zip(sample_indices, sample_names):
        # 取出指定樣本，轉為 numpy
        pred_np = preds[sample_idx].detach().cpu().numpy() / target_scale      # [3, 1, H, W]
        target_np = targets[sample_idx].detach().cpu().numpy() / target_scale  # [3, 1, H, W]
        mask_np = masks[sample_idx].detach().cpu().numpy()                     # [3, 1, H, W]
        input_np = inputs[sample_idx].detach().cpu().numpy()                   # [9, C, H, W]
        input_rain = input_np[:, 0]                                            # [9, H, W]
        input_flood = None
        if input_np.shape[1] > 1:
            input_flood = input_np[:, 1] / target_scale                        # [9, H, W] 還原淹水縮放
            input_flood = np.clip(input_flood, 0.0, None)                      # 避免負值影響視覺化
        
        # 創建 4x3 的圖表：第一行輸入降雨，第二行輸入淹水，第三行預測增量，第四行真實增量
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        
        titles = ['t+1', 't+2', 't+3']
        
        # 計算統一的顏色範圍 (淹水增量，可能有正負值)
        valid_pred = pred_np[mask_np > 0]
        valid_target = target_np[mask_np > 0]
        if len(valid_pred) > 0 and len(valid_target) > 0:
            all_valid = np.concatenate([valid_pred, valid_target])
            # 使用對稱範圍：取 1% 和 99% 百分位數
            abs_max = max(abs(np.percentile(all_valid, 1)), abs(np.percentile(all_valid, 99)))
            flood_vmax = max(abs_max, 0.01)
        else:
            flood_vmax = 0.1
        flood_vmin = -flood_vmax  # 對稱範圍
        
        # 降雨的色彩範圍
        rain_vmax = np.percentile(input_rain[6:9], 95)  # 未來 3 小時的雨量
        rain_vmin = 0.0
        
        # 第一行：輸入序列 (未來 3 小時的預報降雨，t+1, t+2, t+3)
        for i in range(3):
            ax = axes[0, i]
            rain_frame = input_rain[6 + i]  # 未來 3 小時的降雨
            # 應用遮罩於輸入
            rain_masked = np.ma.masked_where(mask_np[i, 0] == 0, rain_frame)
            im = ax.imshow(rain_masked, cmap='Blues', vmin=rain_vmin, vmax=rain_vmax)
            ax.set_title(f'Rain Input {titles[i]} (mm/hr)', fontweight='bold')
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('mm/hr')

        # 第二行：輸入淹水圖 (使用 t 時刻狀態，顯示在 t+1, t+2, t+3 欄位)
        if input_flood is not None:
            flood_input_vmax = max(np.percentile(input_flood, 95), 1e-6)
            for i in range(3):
                ax = axes[1, i]
                flood_frame = input_flood[0]  # t 時刻淹水狀態
                flood_masked = np.ma.masked_where(mask_np[i, 0] == 0, flood_frame)
                im = ax.imshow(flood_masked, cmap='Blues', vmin = 0, vmax=flood_input_vmax)
                ax.set_title(f'Flood Input t (m)', fontweight='bold')
                ax.axis('off')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('m')
        else:
            for i in range(3):
                axes[1, i].axis('off')
                axes[1, i].set_title('Flood Input (N/A)', fontweight='bold')
        
        # 第三行：預測值
        for i in range(3):
            ax = axes[2, i]
            pred_masked = np.ma.masked_where(mask_np[i, 0] == 0, pred_np[i, 0])
            im = ax.imshow(pred_masked, cmap='RdBu_r', vmin=flood_vmin, vmax=flood_vmax)
            ax.set_title(f'Prediction Δh {titles[i]} (m)', fontweight='bold')
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('m')
        
        # 第四行：真實值
        for i in range(3):
            ax = axes[3, i]
            target_masked = np.ma.masked_where(mask_np[i, 0] == 0, target_np[i, 0])
            im = ax.imshow(target_masked, cmap='RdBu_r', vmin=flood_vmin, vmax=flood_vmax)
            ax.set_title(f'Ground Truth Δh {titles[i]} (m)', fontweight='bold')
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('m')
        
        plt.suptitle(f'Epoch {epoch} - Sample {sample_idx+1}/{n_samples} ({sample_name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'val_epoch_{epoch:03d}_{sample_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"  驗證比較圖已保存: {save_dir}/val_epoch_{epoch:03d}_[first|middle|last].png")


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
                target_scale = CONFIG.get('target_scale', 1.0)
                zero_weight = CONFIG.get('zero_weight', 0.0)
                loss = weighted_flood_loss(
                    pred,
                    flood_target,
                    mask,
                    flood_weight=flood_weight,
                    flood_threshold=CONFIG.get('flood_threshold', 0.005),
                    target_scale=target_scale,
                    zero_weight=zero_weight
                )
            
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
        
        # 收集所有驗證資料用於視覺化
        all_preds = []
        all_targets = []
        all_masks = []
        all_inputs = []
        
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
                    target_scale = CONFIG.get('target_scale', 1.0)
                    zero_weight = CONFIG.get('zero_weight', 0.0)
                    loss = weighted_flood_loss(
                        pred,
                        flood_target,
                        mask,
                        flood_weight=flood_weight,
                        flood_threshold=CONFIG.get('flood_threshold', 0.005),
                        target_scale=target_scale,
                        zero_weight=zero_weight,

                    )
                
                val_loss += loss.item()
                
                # 計算 MAE
                mae = (torch.abs(pred - flood_target) * mask).sum() / (mask.sum() + 1e-6)
                val_mae += mae.item()
                
                # 計算淹水區域 MSE (閾值需要配合 target_scale)
                target_scale = CONFIG.get('target_scale', 1.0)
                flood_threshold = CONFIG.get('flood_threshold', 0.005) * target_scale  # 縮放後的閾值
                flood_mask = (torch.abs(flood_target) > flood_threshold).float() * mask
                if flood_mask.sum() > 0:
                    flood_mse = ((pred - flood_target) ** 2 * flood_mask).sum() / (flood_mask.sum() + 1e-6)
                    val_flood_mse += flood_mse.item()
                
                # 收集所有 batch 的資料 (移到 CPU 節省 GPU 記憶體)
                all_preds.append(pred.cpu())
                all_targets.append(flood_target.cpu())
                all_masks.append(mask.cpu())
                all_inputs.append(rain_input.cpu())
        
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
        
        # 每個 epoch 都保存驗證比較圖 (第一筆、中間、最後一筆)
        if len(all_preds) > 0:
            save_validation_comparison(
                all_preds,
                all_targets,
                all_masks,
                all_inputs,
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
