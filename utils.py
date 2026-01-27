import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# 設定 matplotlib 支援中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

"""
這裡放置通用的功能，例如「搜尋颱風資料」、「遮罩損失函數」和「可視化功能」。
這樣可以讓主訓練程式保持乾淨。
"""

def find_typhoon_data(root_dir):
    """
    掃描資料夾，尋找符合時序的 (Rain, Flood) 配對序列
    """
    all_sequences = []
    typhoon_folders = glob.glob(os.path.join(root_dir, 't*'))
    
    for t_folder in typhoon_folders:
        rain_dir = os.path.join(t_folder, 'rain')
        flood_dir = os.path.join(t_folder, 'flood')
        
        if not (os.path.exists(rain_dir) and os.path.exists(flood_dir)):
            continue
            
        rain_files = glob.glob(os.path.join(rain_dir, '*.csv'))
        flood_files = glob.glob(os.path.join(flood_dir, '*.csv'))
        
        # 過濾掉特殊檔案
        rain_files = [f for f in rain_files if 'rain_max.csv' not in f]
        flood_files = [f for f in flood_files if 'dm1maxd0.csv' not in f]
        
        def get_rain_id(fname):
            try: return int(os.path.basename(fname).split('_')[-1].replace('.csv', ''))
            except: return -1

        def get_flood_id(fname):
            try: return int(os.path.basename(fname).replace('dm1d', '').replace('.csv', ''))
            except: return -1
        
        rain_map = {get_rain_id(f): f for f in rain_files if get_rain_id(f) != -1}
        flood_map = {get_flood_id(f): f for f in flood_files if get_flood_id(f) != -1}
        
        sorted_keys = sorted(rain_map.keys())
        
        for t in sorted_keys:
            # 檢查 t-5 到 t+3 (共9幀)
            rain_seq_paths = []
            missing = False
            for offset in range(-5, 4): 
                idx = t + offset
                if idx in rain_map: rain_seq_paths.append(rain_map[idx])
                else:
                    missing = True
                    break
            if missing: continue
                
            # 檢查 t+1 到 t+3 (共3幀)
            flood_seq_paths = []
            for offset in range(1, 4):
                idx = t + offset
                if idx in flood_map: flood_seq_paths.append(flood_map[idx])
                else:
                    missing = True
                    break
            if missing: continue
                
            all_sequences.append((rain_seq_paths, flood_seq_paths))
            
    print(f"找到 {len(all_sequences)} 組有效序列。")
    return all_sequences


def masked_mse_loss(pred, target, mask):
    """
    只計算 mask=1 (有效區域) 的 MSE 損失
    """
    diff = pred - target
    squared_err = diff ** 2
    masked_err = squared_err * mask
    loss = masked_err.sum() / (mask.sum() + 1e-6)
    return loss


def load_csv_data(csv_path):
    """
    讀取 CSV 檔案並處理缺失值
    
    Args:
        csv_path: CSV 檔案路徑
    
    Returns:
        data: numpy array，缺失值 (-999.999) 已轉為 np.nan
    """
    data = pd.read_csv(csv_path, header=None).values
    data = data.astype(float)
    # 將 -999.999 轉換為 nan
    data[data <= -999] = np.nan
    return data


def visualize_sequence_to_gif(
    typhoon_folder,
    start_time=0,
    end_time=None,
    output_path='visualization.gif',
    fps=2,
    figsize=(14, 6),
    rain_vmax=None,
    flood_vmax=None,
    show_colorbar=True,
    show_title=True
):
    """
    生成降雨量和淹水深度的動畫 GIF
    
    Args:
        typhoon_folder: 颱風資料夾路徑 (例如 'test_data/t12')
        start_time: 起始時間索引 (預設 0)
        end_time: 結束時間索引 (預設 None，自動使用最大可用時間)
        output_path: 輸出 GIF 路徑
        fps: 幀率 (幀/秒)
        figsize: 圖片大小 (寬, 高)
        rain_vmax: 降雨量最大值 (預設 None，自動計算)
        flood_vmax: 淹水深度最大值 (預設 None，自動計算)
        show_colorbar: 是否顯示色條
        show_title: 是否顯示標題
    
    Returns:
        生成的 GIF 檔案路徑
    """
    # 建立輸出資料夾
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 取得所有 rain 和 flood 檔案
    rain_dir = os.path.join(typhoon_folder, 'rain')
    flood_dir = os.path.join(typhoon_folder, 'flood')
    
    if not os.path.exists(rain_dir):
        raise FileNotFoundError(f"找不到降雨資料夾: {rain_dir}")
    if not os.path.exists(flood_dir):
        raise FileNotFoundError(f"找不到淹水資料夾: {flood_dir}")
    
    # 讀取並排序檔案
    rain_files = sorted(glob.glob(os.path.join(rain_dir, '*.csv')))
    flood_files = sorted(glob.glob(os.path.join(flood_dir, '*.csv')))
    
    # 過濾掉特殊檔案
    rain_files = [f for f in rain_files if 'rain_max.csv' not in f]
    flood_files = [f for f in flood_files if 'dm1maxd0.csv' not in f]
    
    if len(rain_files) == 0:
        raise ValueError(f"降雨資料夾中沒有 CSV 檔案: {rain_dir}")
    
    # 確定時間範圍
    max_rain_time = len(rain_files) - 1
    if end_time is None:
        end_time = max_rain_time
    else:
        end_time = min(end_time, max_rain_time)
    
    if start_time > end_time:
        raise ValueError(f"起始時間 ({start_time}) 不能大於結束時間 ({end_time})")
    
    print(f"生成 GIF：時間範圍 {start_time} 到 {end_time}，共 {end_time - start_time + 1} 幀")
    print(f"  Rain 檔案數: {len(rain_files)}, Flood 檔案數: {len(flood_files)}")
    
    # 預先讀取所有資料以計算 vmax
    rain_data_list = []
    flood_data_list = []
    
    for t in range(start_time, end_time + 1):
        if t < len(rain_files):
            rain_data = load_csv_data(rain_files[t])
            rain_data_list.append(rain_data)
            print(f"  載入 rain[{t}]: {os.path.basename(rain_files[t])}, 最大值={np.nanmax(rain_data):.2f}")
        
        if t < len(flood_files):
            flood_data = load_csv_data(flood_files[t])
            flood_data_list.append(flood_data)
            print(f"  載入 flood[{t}]: {os.path.basename(flood_files[t])}, 最大值={np.nanmax(flood_data):.2f}")
    
    # 自動計算 vmax（排除 nan）
    if rain_vmax is None and rain_data_list:
        rain_vmax = np.nanmax([np.nanmax(d) for d in rain_data_list if d is not None])
    if flood_vmax is None and flood_data_list:
        valid_floods = [np.nanmax(d) for d in flood_data_list if d is not None and np.any(~np.isnan(d))]
        flood_vmax = np.nanmax(valid_floods) if valid_floods else 1.0
    
    # 建立圖表
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # 初始化圖像
    rain_im = axes[0].imshow(np.zeros((100, 100)), cmap='Blues', vmin=0, vmax=rain_vmax, aspect='auto')
    flood_im = axes[1].imshow(np.zeros((100, 100)), cmap='YlOrRd', vmin=0, vmax=flood_vmax, aspect='auto')
    
    # 設定標題和標籤
    axes[0].set_title('降雨量 (m)', fontsize=14, fontweight='bold')
    axes[1].set_title('淹水深度 (m)', fontsize=14, fontweight='bold')
    
    for ax in axes:
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.grid(False)
    
    # 加入色條
    if show_colorbar:
        plt.colorbar(rain_im, ax=axes[0], fraction=0.046, pad=0.04, label='m')
        plt.colorbar(flood_im, ax=axes[1], fraction=0.046, pad=0.04, label='m')
    
    # 總標題
    typhoon_name = os.path.basename(typhoon_folder)
    if show_title:
        fig.suptitle(f'颱風: {typhoon_name}', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # 動畫更新函數
    def update(frame_idx):
        t = start_time + frame_idx
        
        # 更新降雨量
        if frame_idx < len(rain_data_list):
            rain_data = rain_data_list[frame_idx]
            rain_im.set_data(rain_data)
            axes[0].set_title(f'降雨量 (m) - 時刻 {t}', fontsize=14, fontweight='bold')
        
        # 更新淹水深度
        if frame_idx < len(flood_data_list):
            flood_data = flood_data_list[frame_idx]
            flood_im.set_data(flood_data)
            axes[1].set_title(f'淹水深度 (m) - 時刻 {t}', fontsize=14, fontweight='bold')
        else:
            # 如果沒有對應的淹水數據，顯示空白
            axes[1].set_title(f'淹水深度 (m) - 時刻 {t} (無數據)', fontsize=14, fontweight='bold')
        
        return [rain_im, flood_im]
    
    # 建立動畫
    n_frames = end_time - start_time + 1
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000/fps, blit=True
    )
    
    # 儲存 GIF
    print(f"正在儲存 GIF 到: {output_path}")
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    
    print(f"✓ GIF 已儲存: {output_path}")
    return output_path


# ===== 測試區 =====
if __name__ == "__main__":
    print("=" * 60)
    print("測試 1: find_typhoon_data")
    print("=" * 60)
    
    # 測試訓練資料
    train_sequences = find_typhoon_data('train_data')
    if train_sequences:
        print(f"✓ 訓練資料：找到 {len(train_sequences)} 組序列")
        print(f"  第一組範例：")
        rain_paths, flood_paths = train_sequences[0]
        print(f"    Rain 序列: {len(rain_paths)} 個檔案")
        for i, p in enumerate(rain_paths):
            print(f"      [{i}] {os.path.basename(p)}")
        print(f"    Flood 序列: {len(flood_paths)} 個檔案")
        for i, p in enumerate(flood_paths):
            print(f"      [{i}] {os.path.basename(p)}")
    else:
        print("✗ 訓練資料：未找到序列")
    
    print()
    
    # 測試驗證資料
    val_sequences = find_typhoon_data('val_data')
    if val_sequences:
        print(f"✓ 驗證資料：找到 {len(val_sequences)} 組序列")
    else:
        print("✗ 驗證資料：未找到序列")
    
    # 測試測試資料
    test_sequences = find_typhoon_data('test_data')
    if test_sequences:
        print(f"✓ 測試資料：找到 {len(test_sequences)} 組序列")
    else:
        print("✗ 測試資料：未找到序列")
    
    print()
    print("=" * 60)
    print("測試 2: masked_mse_loss")
    print("=" * 60)
    
    # 建立測試張量
    pred = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])
    target = torch.tensor([[1.5, 2.5, 3.5],
                           [4.5, 5.5, 6.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0],
                         [1.0, 0.0, 1.0]])
    
    loss = masked_mse_loss(pred, target, mask)
    print(f"預測值:\n{pred}")
    print(f"目標值:\n{target}")
    print(f"遮罩:\n{mask}")
    print(f"計算損失: {loss.item():.6f}")
    
    # 手動驗證
    diff = pred - target  # [[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]]
    squared = diff ** 2   # [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]
    masked = squared * mask  # [[0.25, 0.25, 0.0], [0.25, 0.0, 0.25]]
    expected = masked.sum() / mask.sum()  # 1.0 / 4.0 = 0.25
    print(f"預期損失: {expected.item():.6f}")
    
    if abs(loss.item() - expected.item()) < 1e-6:
        print("✓ 損失函數計算正確")
    else:
        print("✗ 損失函數計算錯誤")
    
    print()
    print("=" * 60)
    print("測試 3: 序列配對邏輯驗證")
    print("=" * 60)
    
    # 驗證序列索引邏輯
    t = 5  # 假設當前時刻
    rain_indices = list(range(t-5, t+4))  # t-5 到 t+3
    flood_indices = list(range(t+1, t+4))  # t+1 到 t+3
    
    print(f"當前時刻 t = {t}")
    print(f"Rain 序列索引 (t-5 到 t+3): {rain_indices}")
    print(f"  → 共 {len(rain_indices)} 個時刻")
    print(f"Flood 序列索引 (t+1 到 t+3): {flood_indices}")
    print(f"  → 共 {len(flood_indices)} 個時刻")
    
    print()
    print("邏輯說明：")
    print("  - Rain 輸入包含：過去5個時刻 + 當前時刻 + 未來3個時刻 = 9個")
    print("  - Flood 輸出包含：未來3個時刻 (t+1, t+2, t+3)")
    print("  - 這樣模型可以看到 t-5 到 t+3 的降雨，預測 t+1 到 t+3 的淹水")
    
    print()
    print("=" * 60)
    print("測試 4: 可視化功能 - 生成 GIF")
    print("=" * 60)
    
    # 測試可視化功能
    test_folder = 'train_data/t6'
    if os.path.exists(test_folder):
        print(f"測試資料夾: {test_folder}")
        
        # 建立 visualizations 資料夾
        vis_dir = 'visualizations'
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
            print(f"✓ 已建立資料夾: {vis_dir}")
        
        # 生成範例 GIF
        try:
            output_gif = os.path.join(vis_dir, 'demo.gif')
            visualize_sequence_to_gif(
                typhoon_folder=test_folder,
                start_time=0,
                end_time=None,
                output_path=output_gif,
                fps=2,
                show_colorbar=True,
                show_title=True
            )
            print(f"✓ 範例 GIF 已生成")
        except Exception as e:
            print(f"✗ 生成 GIF 時發生錯誤: {e}")
    else:
        print(f"✗ 找不到測試資料夾: {test_folder}")
    
    print()
    print("=" * 60)
    print("所有測試完成！")
    print("=" * 60)
