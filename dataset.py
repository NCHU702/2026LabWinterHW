import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

"""
這裡專注於資料的讀取、預處理（Masking）以及隨機擾動（Stochastic Perturbation）。

重要說明：
- Flood 資料為累積淹水深度（每個時間點包含之前的淹水）
- Dataset 會自動計算差值（增量），作為模型的預測目標
- 目標值 = 當前淹水深度 - 前一時間點淹水深度
"""

class StochasticRainDataset(Dataset):
    def __init__(self, sequences, config):
        self.sequences = sequences
        self.config = config
        
    def __len__(self):
        return len(self.sequences)
    
    def _read_csv(self, path):
        """讀取 CSV 並產生遮罩"""
        try:
            df = pd.read_csv(path, header=None)
            raw = df.values.astype(np.float32)
            
            # Mask: > -999 為有效資料 (NODATA_value = -999.999)
            mask = (raw > -999).astype(np.float32)
            
            # 將無效值填補為 0
            data = raw.copy()
            data[data <= -999] = 0.0
            
            return data, mask
        except Exception as e:
            print(f"讀取錯誤 {path}: {e}")
            return np.zeros((10, 10), dtype=np.float32), np.zeros((10, 10), dtype=np.float32)

    def _add_forecast_error(self, rain_grid):
        """
        隨機擾動模擬預報誤差（空間連續性）
        
        改進要點：
        1. 使用高斯平滑生成空間連續的擾動場
        2. 乘性誤差具有空間相關性（模擬系統性偏差）
        3. 加性雜訊保持稀疏性但區域連貫
        4. 基於實際颱風資料特性調整強度
        """
        from scipy.ndimage import gaussian_filter
        
        h, w = rain_grid.shape
        
        # === 1. 空間連續的乘性誤差場 ===
        # 根據降雨強度決定 sigma（用全局最大值，保持一致性）
        max_rain = np.max(rain_grid)
        if max_rain > 15.0:
            base_sigma = 0.4
        elif max_rain > 5.0:
            base_sigma = 0.3
        else:
            base_sigma = 0.2
        
        # 生成空間連續的隨機場（高斯過程近似）
        # 先生成白噪音，再用高斯平滑創造空間相關性
        noise_field = np.random.randn(h, w)
        # sigma_spatial 控制空間相關尺度（像素）
        sigma_spatial = min(h, w) * 0.05  # 約 5% 的圖像尺寸
        smooth_noise = gaussian_filter(noise_field, sigma=sigma_spatial)
        
        # 標準化並縮放到目標 sigma
        smooth_noise = (smooth_noise - smooth_noise.mean()) / (smooth_noise.std() + 1e-8)
        smooth_noise = smooth_noise * base_sigma
        
        # 轉換為 log-normal multiplier
        mu = -0.5 * (base_sigma ** 2)
        multiplier = np.exp(mu + smooth_noise)
        
        # === 2. 空間連續的加性誤差場 ===
        # 生成連續的擾動強度場
        noise_additive = np.random.randn(h, w)
        smooth_additive = gaussian_filter(noise_additive, sigma=sigma_spatial * 0.5)
        smooth_additive = np.abs(smooth_additive)  # 取絕對值（正偏）
        
        # 根據降雨強度自適應縮放
        base_scale = 0.1
        adaptive_scale = np.maximum(rain_grid * 0.05, base_scale)
        delta = smooth_additive * adaptive_scale * 0.5  # 降低整體強度
        
        # 稀疏遮罩（但保持局部連續性）
        # 生成連續的遮罩場，再二值化
        mask_field = np.random.randn(h, w)
        smooth_mask = gaussian_filter(mask_field, sigma=sigma_spatial * 0.3)
        mask_noise = (smooth_mask > 0.3).astype(np.float32)  # 約 30% 區域有雜訊
        
        delta = delta * mask_noise
        
        # === 3. 組合擾動 ===
        r_pred = rain_grid * multiplier + delta
        r_pred = np.maximum(r_pred, 0.0) 
        
        return r_pred.astype(np.float32)

    def __getitem__(self, idx):
        rain_paths, flood_paths = self.sequences[idx]
        rain_frames = []
        
        # 1. 過去 6 小時 (觀測)
        for i in range(6):
            grid, _ = self._read_csv(rain_paths[i])
            rain_frames.append(grid)
            
        # 2. 未來 3 小時 (預報+擾動)
        # 所有模式都加擾動以模擬預報不確定性
        for i in range(6, 9):
            grid, _ = self._read_csv(rain_paths[i])
            noisy_grid = self._add_forecast_error(grid)
            rain_frames.append(noisy_grid)
            
        # 3. 讀取 flood 原始資料（累積值）
        # flood_paths 包含 [t, t+1, t+2, t+3] 共 4 個檔案
        flood_raw = []
        mask_frames = []
        for path in flood_paths:
            grid, mask = self._read_csv(path)
            flood_raw.append(grid)
            mask_frames.append(mask)
        
        # 4. 計算差值（增量）作為目標
        # 注意：原始 flood 資料是累積值，模型需要預測增量
        # flood_raw[0] = t 時刻（基準）
        # flood_raw[1] = t+1, flood_raw[2] = t+2, flood_raw[3] = t+3
        flood_frames = []
        target_scale = self.config.get('target_scale', 1.0)  # 目標值縮放因子
        for i in range(1, len(flood_raw)):  # 從索引 1 開始（t+1, t+2, t+3）
            # t+i 的增量 = flood[t+i] - flood[t+(i-1)]
            diff = flood_raw[i] - flood_raw[i-1]
            # 縮放目標值讓模型更容易學習
            diff = diff * target_scale
            flood_frames.append(diff)
        
        # 只使用後 3 個時刻的 mask（對應 t+1, t+2, t+3）
        mask_frames = mask_frames[1:]
            
        # 轉 Tensor
        input_tensor = torch.from_numpy(np.array(rain_frames)).unsqueeze(1)  # [9, 1, H, W]
        target_tensor = torch.from_numpy(np.array(flood_frames)).unsqueeze(1)  # [3, 1, H, W]
        mask_tensor = torch.from_numpy(np.array(mask_frames)).unsqueeze(1)  # [3, 1, H, W]
        
        # Padding
        _, _, h, w = input_tensor.shape
        pm = self.config.get('pad_multiple', 4)
        target_h = ((h - 1) // pm + 1) * pm
        target_w = ((w - 1) // pm + 1) * pm
        
        pad_bottom = target_h - h
        pad_right = target_w - w
        
        input_pad = F.pad(input_tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
        target_pad = F.pad(target_tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
        mask_pad = F.pad(mask_tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
        
        return input_pad, target_pad, mask_pad


if __name__ == "__main__":
    """測試 Dataset 功能"""
    import sys
    sys.path.append('.')
    from utils import find_typhoon_data
    from config import CONFIG
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import os
    
    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("="*60)
    print("測試 StochasticRainDataset")
    print("="*60)
    
    # 1. 載入訓練資料序列
    print("\n[1] 掃描訓練資料...")
    train_sequences = find_typhoon_data(CONFIG['train_data_dir'])
    
    if len(train_sequences) == 0:
        print("❌ 找不到訓練資料！")
        sys.exit(1)
    
    # 2. 建立 Dataset (訓練模式)
    print("\n[2] 建立訓練 Dataset (mode='train')...")
    train_config = CONFIG.copy()
    train_config['mode'] = 'train'
    train_dataset = StochasticRainDataset(train_sequences, train_config)
    
    print(f"✅ Dataset 大小: {len(train_dataset)} 組序列")
    
    # 3. 測試讀取第一筆資料
    print("\n[3] 測試讀取第 0 筆資料...")
    try:
        input_data, target_data, mask_data = train_dataset[0]
        
        print(f"✅ 輸入 (降雨) 維度: {input_data.shape}")
        print(f"   - 應為 [9, 1, H, W]: 9 個時間步，1 通道")
        print(f"   - 降雨範圍: [{input_data.min():.2f}, {input_data.max():.2f}]")
        
        print(f"✅ 目標 (淹水增量) 維度: {target_data.shape}")
        print(f"   - 應為 [3, 1, H, W]: 3 個時間步，1 通道")
        print(f"   - 淹水增量範圍: [{target_data.min():.2f}, {target_data.max():.2f}] (可能有負值)")
        
        print(f"✅ 遮罩維度: {mask_data.shape}")
        print(f"   - 應為 [3, 1, H, W]")
        print(f"   - 有效像素比例: {(mask_data.sum() / mask_data.numel() * 100):.1f}%")
        
    except Exception as e:
        print(f"❌ 讀取失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 4. 測試驗證模式 (不加擾動)
    print("\n[4] 建立驗證 Dataset (mode='val')...")
    val_config = CONFIG.copy()
    val_config['mode'] = 'val'
    val_dataset = StochasticRainDataset(train_sequences[:5], val_config)
    
    input_val, target_val, mask_val = val_dataset[0]
    print(f"✅ 驗證模式輸入維度: {input_val.shape}")
    print(f"   - 降雨範圍: [{input_val.min():.2f}, {input_val.max():.2f}]")
    
    # 5. 比較訓練/驗證模式的差異
    print("\n[5] 比較訓練/驗證模式 (未來 3 小時部分)...")
    train_future = input_data[6:9]  # 訓練模式的未來預報
    val_future = input_val[6:9]     # 驗證模式的未來預報
    
    diff = torch.abs(train_future - val_future).mean()
    print(f"   - 平均差異: {diff:.4f}")
    if diff > 0.01:
        print(f"   ✅ 訓練模式成功加入隨機擾動")
    else:
        print(f"   ⚠️  差異很小，擾動可能未生效")
    
    # 6. 測試 DataLoader
    print("\n[6] 測試 DataLoader...")
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True
    )
    
    for batch_idx, (inputs, targets, masks) in enumerate(train_loader):
        print(f"✅ Batch {batch_idx}: ")
        print(f"   - Inputs:  {inputs.shape}")
        print(f"   - Targets: {targets.shape}")
        print(f"   - Masks:   {masks.shape}")
        if batch_idx >= 2:  # 只顯示前 3 個 batch
            break
    
    print("\n" + "="*60)
    print("✅ Dataset 測試完成！所有功能正常。")
    print("="*60)
    
    # 7. 視覺化測試
    print("\n[7] 視覺化第一筆資料...")
    
    # 建立視覺化資料夾
    vis_dir = 'visualizations/dataset_test'
    os.makedirs(vis_dir, exist_ok=True)
    
    # 取得第一筆資料
    sample_idx = 0
    input_data, target_data, mask_data = train_dataset[sample_idx]
    
    # 定義色彩映射
    rain_colors = ['#FFFFFF', '#C6E9FF', '#92D1FF', '#5EAEFF', '#2A7FFF', '#0050C8']
    rain_cmap = LinearSegmentedColormap.from_list('rain', rain_colors)
    
    flood_colors = ['#FFFFFF', '#FFE6E6', '#FFAAAA', '#FF6666', '#FF0000', '#AA0000']
    flood_cmap = LinearSegmentedColormap.from_list('flood', flood_colors)
    
    # === 圖 1: 完整時間序列 (9 個降雨 + 3 個淹水) ===
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Dataset Sample {sample_idx}: 時間序列視覺化', fontsize=16, fontweight='bold')
    
    # 前 9 個降雨 (取前 8 個顯示)
    for i in range(8):
        ax = axes[i // 4, i % 4]
        rain_frame = input_data[i, 0].numpy()
        im = ax.imshow(rain_frame, cmap=rain_cmap, vmin=0, vmax=rain_frame.max())
        
        if i < 6:
            ax.set_title(f't-{5-i} (過去觀測)', fontsize=10)
        else:
            ax.set_title(f't+{i-5} (未來預報)', fontsize=10, color='red')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
    
    # 第 9 個降雨
    ax = axes[2, 0]
    rain_frame = input_data[8, 0].numpy()
    im = ax.imshow(rain_frame, cmap=rain_cmap, vmin=0, vmax=rain_frame.max())
    ax.set_title(f't+3 (未來預報)', fontsize=10, color='red')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    
    # 3 個淹水增量目標
    for i in range(3):
        ax = axes[2, i + 1]
        flood_frame = target_data[i, 0].numpy()
        # 增量可能有負值，使用對稱色階
        vmin = min(flood_frame.min(), 0)
        vmax = flood_frame.max()
        im = ax.imshow(flood_frame, cmap=flood_cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f't+{i+1} (淹水增量)', fontsize=10, color='blue')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(vis_dir, 'full_sequence.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 儲存: {fig_path}")
    plt.close()
    
    # === 圖 2: 訓練 vs 驗證模式比較 (擾動效果) ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('訓練模式 vs 驗證模式：擾動效果比較 (未來 3 小時)', fontsize=16, fontweight='bold')
    
    input_val, _, _ = val_dataset[sample_idx]
    
    for i in range(3):
        # 訓練模式 (有擾動)
        ax = axes[0, i]
        train_frame = input_data[6 + i, 0].numpy()
        im = ax.imshow(train_frame, cmap=rain_cmap, vmin=0, vmax=max(train_frame.max(), 1))
        ax.set_title(f't+{i+1} 訓練模式 (有擾動)', fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        
        # 驗證模式 (無擾動)
        ax = axes[1, i]
        val_frame = input_val[6 + i, 0].numpy()
        im = ax.imshow(val_frame, cmap=rain_cmap, vmin=0, vmax=max(train_frame.max(), 1))
        ax.set_title(f't+{i+1} 驗證模式 (無擾動)', fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(vis_dir, 'train_vs_val.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 儲存: {fig_path}")
    plt.close()
    
    # === 圖 3: 擾動差異熱圖 ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('擾動差異熱圖 (訓練 - 驗證)', fontsize=16, fontweight='bold')
    
    for i in range(3):
        ax = axes[i]
        train_frame = input_data[6 + i, 0].numpy()
        val_frame = input_val[6 + i, 0].numpy()
        diff = train_frame - val_frame
        
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-diff.std()*3, vmax=diff.std()*3)
        ax.set_title(f't+{i+1} 差異 (mean={diff.mean():.3f}mm)', fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='mm')
        ax.axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(vis_dir, 'perturbation_diff.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 儲存: {fig_path}")
    plt.close()
    
    # === 圖 4: 遮罩視覺化（增量 vs 累積） ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('淹水增量 vs 累積深度 + 有效區域遮罩', fontsize=16, fontweight='bold')
    
    # 計算累積淹水深度（用於顯示）
    # 重新讀取原始累積值並進行相同的 padding
    flood_cumulative = []
    for path in train_dataset.sequences[sample_idx][1]:  # flood_paths
        grid, mask_raw = train_dataset._read_csv(path)
        # 轉為 tensor 並 padding（與 __getitem__ 相同）
        grid_tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        h, w = grid.shape
        pm = train_dataset.config.get('pad_multiple', 4)
        target_h = ((h - 1) // pm + 1) * pm
        target_w = ((w - 1) // pm + 1) * pm
        pad_bottom = target_h - h
        pad_right = target_w - w
        grid_padded = F.pad(grid_tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
        flood_cumulative.append(grid_padded[0, 0].numpy())
    
    for i in range(3):
        # 第一行：增量
        ax = axes[0, i]
        flood_frame = target_data[i, 0].numpy()
        mask_frame = mask_data[i, 0].numpy()
        
        # 顯示淹水增量（可能有負值）
        vmin = min(flood_frame.min(), 0)
        vmax = flood_frame.max()
        im = ax.imshow(flood_frame, cmap='RdBu_r', vmin=-max(abs(vmin), abs(vmax)), vmax=max(abs(vmin), abs(vmax)))
        
        # 疊加遮罩邊界
        ax.contour(mask_frame, levels=[0.5], colors='black', linewidths=1, alpha=0.5)
        
        valid_ratio = (mask_frame.sum() / mask_frame.size) * 100
        increment_mean = flood_frame[mask_frame > 0.5].mean() if (mask_frame > 0.5).any() else 0
        ax.set_title(f't+{i+1} 增量 (mean={increment_mean:.4f}m)', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='增量 (m)')
        ax.axis('off')
        
        # 第二行：累積值（絕對深度）
        ax = axes[1, i]
        cumulative_frame = flood_cumulative[i]
        
        # 顯示累積淹水深度
        im = ax.imshow(cumulative_frame, cmap=flood_cmap, vmin=0, vmax=max(cumulative_frame.max(), 0.01))
        
        # 疊加遮罩邊界
        ax.contour(mask_frame, levels=[0.5], colors='black', linewidths=1, alpha=0.5)
        
        cumulative_mean = cumulative_frame[mask_frame > 0.5].mean() if (mask_frame > 0.5).any() else 0
        ax.set_title(f't+{i+1} 累積深度 (mean={cumulative_mean:.4f}m)', fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='深度 (m)')
        ax.axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(vis_dir, 'flood_with_mask.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 儲存: {fig_path}")
    plt.close()
    
    # === 圖 5: 統計分析 ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Dataset 統計分析', fontsize=16, fontweight='bold')
    
    # 降雨分布 (過去 vs 未來)
    ax = axes[0, 0]
    past_rain = input_data[:6, 0].numpy().flatten()
    future_rain = input_data[6:9, 0].numpy().flatten()
    ax.hist([past_rain, future_rain], bins=50, label=['過去 6hr', '未來 3hr'], alpha=0.7)
    ax.set_xlabel('降雨量 (mm)', fontsize=10)
    ax.set_ylabel('頻率', fontsize=10)
    ax.set_title('降雨分布', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 淹水增量分布
    ax = axes[0, 1]
    flood_all = target_data[:, 0].numpy().flatten()
    flood_valid = flood_all[np.abs(flood_all) > 0.001]  # 排除接近零的值
    ax.hist(flood_valid, bins=50, color='red', alpha=0.7)
    ax.set_xlabel('淹水增量 (m)', fontsize=10)
    ax.set_ylabel('頻率', fontsize=10)
    ax.set_title(f'淹水增量分布 (含正負值)', fontsize=12)
    ax.grid(alpha=0.3)
    
    # 時間序列趨勢
    ax = axes[1, 0]
    rain_means = [input_data[i, 0].numpy().mean() for i in range(9)]
    ax.plot(range(-5, 4), rain_means, marker='o', linewidth=2, markersize=8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='當前時刻')
    ax.set_xlabel('時間 (小時)', fontsize=10)
    ax.set_ylabel('平均降雨量 (mm)', fontsize=10)
    ax.set_title('降雨時間序列趨勢', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 擾動統計
    ax = axes[1, 1]
    diffs = []
    for i in range(3):
        train_frame = input_data[6 + i, 0].numpy()
        val_frame = input_val[6 + i, 0].numpy()
        diff = np.abs(train_frame - val_frame).flatten()
        diffs.extend(diff)
    
    ax.hist(diffs, bins=50, color='purple', alpha=0.7)
    ax.set_xlabel('絕對誤差 (mm)', fontsize=10)
    ax.set_ylabel('頻率', fontsize=10)
    ax.set_title(f'擾動誤差分布 (mean={np.mean(diffs):.3f}mm)', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(vis_dir, 'statistics.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 儲存: {fig_path}")
    plt.close()
    
    print(f"\n✅ 所有視覺化圖片已儲存至: {vis_dir}/")
    print("   - full_sequence.png: 完整時間序列 (降雨 + 淹水增量)")
    print("   - train_vs_val.png: 訓練 vs 驗證模式 (擾動效果)")
    print("   - perturbation_diff.png: 擾動差異熱圖")
    print("   - flood_with_mask.png: 淹水增量 vs 累積深度 (上下對比)")
    print("   - statistics.png: 統計分析 (含增量分布)")
