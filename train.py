import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
from datetime import datetime

# 導入自定義模組
from config import CONFIG
from utils import find_typhoon_data, masked_mse_loss
from dataset import StochasticRainDataset
from model import HydroNetRainOnly

def init_weights(m):
    """初始化模型權重"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train():
    # 設置隨機種子以確保可重現性
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 1. 準備資料
    print("正在搜尋訓練資料...")
    train_sequences = find_typhoon_data(CONFIG['train_data_dir'])
    
    print("正在搜尋驗證資料...")
    val_sequences = find_typhoon_data('val_data')
    
    if len(train_sequences) == 0:
        print("未找到訓練資料，請檢查路徑。")
        return
    
    if len(val_sequences) == 0:
        print("未找到驗證資料，請檢查路徑。")
        return
    
    # 建立訓練集和驗證集
    train_ds = StochasticRainDataset(train_sequences, CONFIG)
    val_ds = StochasticRainDataset(val_sequences, CONFIG)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # 2. 初始化模型
    model = HydroNetRainOnly().to(CONFIG['device'])
    model.apply(init_weights)  # 初始化權重
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 學習率調度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    # 追蹤最佳模型和早停
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # 創建訓練日誌
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(CONFIG['save_dir'], f'training_log_{timestamp}.csv')
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate', 'best_val_loss'])
    
    print(f"\n開始訓練... (日誌: {log_path})")
    print("=" * 80)
    
    # 3. 訓練迴圈
    for epoch in range(CONFIG['num_epochs']):
        # ===== 訓練階段 =====
        model.train()
        train_loss = 0
        
        for i, (inputs, targets, masks) in enumerate(train_loader):
            inputs = inputs.to(CONFIG['device'])
            targets = targets.to(CONFIG['device'])
            masks = masks.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 使用 Utils 中的遮罩損失函數
            loss = masked_mse_loss(outputs, targets, masks)
            
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Batch {i}/{len(train_loader)}] Loss: {loss.item():.6f}")
                
        avg_train_loss = train_loss / len(train_loader)
        
        # ===== 驗證階段 =====
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets, masks in val_loader:
                inputs = inputs.to(CONFIG['device'])
                targets = targets.to(CONFIG['device'])
                masks = masks.to(CONFIG['device'])
                
                outputs = model(inputs)
                loss = masked_mse_loss(outputs, targets, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 更新學習率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型並檢查早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
        else:
            patience_counter += 1
        
        # 輸出訓練信息
        print("=" * 80)
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} 完成")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")
        print(f"  Best Val:   {best_val_loss:.6f}")
        
        if avg_val_loss == best_val_loss:
            print(f"  🌟 新的最佳模型！")
        else:
            print(f"  ⏳ 沒有改善 ({patience_counter}/{patience})")
        
        # 記錄日誌
        log_writer.writerow([epoch+1, avg_train_loss, avg_val_loss, current_lr, best_val_loss])
        log_file.flush()
        
        # 每 5 個 epoch 存檔一次
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(CONFIG['save_dir'], f'model_ep{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  💾 已保存檢查點: model_ep{epoch+1}.pth")
        
        # 檢查早停
        if patience_counter >= patience:
            print(f"\n⚠️  Early Stopping at Epoch {epoch+1}")
            print(f"   最佳 Val Loss: {best_val_loss:.6f}")
            break
        
        print("=" * 80)
    
    # 關閉日誌文件
    log_file.close()
    
    print("\n✅ 訓練完成！")
    print(f"   最終最佳 Val Loss: {best_val_loss:.6f}")
    print(f"   訓練日誌: {log_path}")
    print(f"   最佳模型: {os.path.join(CONFIG['save_dir'], 'best_model.pth')}")

if __name__ == "__main__":
    train()
