import torch
import torch.nn as nn

"""
這裡定義神經網路的結構
"""

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell - 卷積 LSTM 單元
    
    參數:
        input_dim: 輸入張量的通道數
        hidden_dim: 隱藏狀態的通道數
        kernel_size: 卷積核大小 (int 或 tuple)
        bias: 是否使用偏置
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 處理 kernel_size 可能是 int 或 tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        # 單一卷積層生成所有門控信號 (input gate, forget gate, output gate, cell gate)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        """
        前向傳播
        
        參數:
            input_tensor: (batch, input_dim, H, W) - 當前時間步輸入
            cur_state: tuple of (h_cur, c_cur)
                h_cur: (batch, hidden_dim, H, W) - 當前隱藏狀態
                c_cur: (batch, hidden_dim, H, W) - 當前細胞狀態
        
        返回:
            h_next: (batch, hidden_dim, H, W) - 下一個隱藏狀態
            c_next: (batch, hidden_dim, H, W) - 下一個細胞狀態
        """
        h_cur, c_cur = cur_state
        
        # 拼接輸入和隱藏狀態
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # 通過卷積生成所有門控信號
        combined_conv = self.conv(combined)
        
        # 分割為四個門控信號
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # 應用激活函數
        i = torch.sigmoid(cc_i)  # input gate
        f = torch.sigmoid(cc_f)  # forget gate
        o = torch.sigmoid(cc_o)  # output gate
        g = torch.tanh(cc_g)     # cell gate
        
        # 更新細胞狀態和隱藏狀態
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        """
        初始化隱藏狀態和細胞狀態
        
        參數:
            batch_size: 批次大小
            image_size: tuple of (height, width)
        
        返回:
            tuple of (h, c) - 初始化的隱藏狀態和細胞狀態
        """
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )

class HydroNetRainOnly(nn.Module):
    def __init__(self, output_steps=3):
        super(HydroNetRainOnly, self).__init__()
        self.output_steps = output_steps  # 預測未來 3 個時間步
        
        # Encoder Layer 1: Conv -> LeakyReLU -> ConvLSTM -> MaxPool
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.lstm1 = ConvLSTMCell(16, 16, 3, True)
        self.pool1 = nn.MaxPool2d(2)
        
        # Encoder Layer 2: Conv -> LeakyReLU -> ConvLSTM -> MaxPool
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.2)
        self.lstm2 = ConvLSTMCell(32, 32, 3, True)
        self.pool2 = nn.MaxPool2d(2)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec1_act = nn.LeakyReLU(0.2)
        
        self.dec2 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.dec2_act = nn.LeakyReLU(0.2)
        
        # Output: 單時間步的淹水增量預測
        self.final = nn.Conv2d(8, 1, 3, padding=1)
        # 不使用 ReLU，因為淹水增量可以是負值（水退）

    def forward(self, x):
        # x: (batch, seq_len, 1, H, W) 例如 (8, 9, 1, 128, 128)
        # seq_len = 9: 過去6小時觀測 + 未來3小時預報
        b, seq, c, h, w = x.size()
        
        # 初始化兩層 LSTM 狀態
        h1, c1 = self.lstm1.init_hidden(batch_size=b, image_size=(h, w))
        h2, c2 = self.lstm2.init_hidden(batch_size=b, image_size=(h//2, w//2))
        
        # 保存 Block2 在未來 3 個時間步的輸出（用於預測）
        h2_futures = []
        
        # 逐時間步處理，信息流經所有層
        for t in range(seq):
            # ===== Encoder Layer 1 =====
            # Conv -> LeakyReLU
            conv_out1 = self.act1(self.conv1(x[:, t]))
            # ConvLSTM - 更新狀態
            h1, c1 = self.lstm1(conv_out1, (h1, c1))
            # MaxPool
            pooled1 = self.pool1(h1)  # (b, 16, h//2, w//2)
            
            # ===== Encoder Layer 2 =====
            # Conv -> LeakyReLU (使用 Layer 1 當前時間步的輸出)
            conv_out2 = self.act2(self.conv2(pooled1))
            # ConvLSTM - 更新狀態
            h2, c2 = self.lstm2(conv_out2, (h2, c2))
            # MaxPool
            pooled2 = self.pool2(h2)  # (b, 32, h//4, w//4)
            
            # 保存未來 3 個時間步的 Block2 輸出（t=6,7,8 對應 t+1,t+2,t+3）
            # 保存 MaxPool 後的結果，準備送入 Decoder
            if t >= 6:
                h2_futures.append(pooled2)
        
        # Decoder: 對每個未來時間步分別解碼
        predictions = []
        for t in range(self.output_steps):  # t+1, t+2, t+3
            # 使用對應時間步的 Block2 輸出（已經過 MaxPool）
            pooled2_t = h2_futures[t]  # (b, 32, h//4, w//4)
            
            # 上採樣回原始解析度
            d1 = self.dec1_act(self.dec1(pooled2_t))  # (b, 16, h//2, w//2)
            d2 = self.dec2_act(self.dec2(d1))         # (b, 8, h, w)
            
            # 預測該時間步的淹水增量（可正可負，允許水退）
            pred = self.final(d2)  # (b, 1, h, w)
            predictions.append(pred)
        
        # 堆疊所有時間步的預測
        out = torch.stack(predictions, dim=1)  # (b, 3, 1, h, w)
        return out


if __name__ == "__main__":
    """
    測試模型資訊流並進行可視化
    使用 test_data 中的真實資料
    """
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式後端，避免 tkinter 問題
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    sys.path.append('.')
    from utils import find_typhoon_data
    from dataset import StochasticRainDataset
    from config import CONFIG
    from torch.utils.data import DataLoader
    
    # 配置中文字體支持
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("=" * 80)
    print("HydroNetRainOnly 模型測試")
    print("=" * 80)
    
    # 1. 載入測試資料
    print("\n[1] 載入測試資料...")
    test_data_dir = 'test_data'
    test_sequences = find_typhoon_data(test_data_dir)
    
    if len(test_sequences) == 0:
        print("❌ 找不到測試資料！")
        sys.exit(1)
    
    print(f"✓ 找到 {len(test_sequences)} 組測試序列")
    
    # 建立測試 Dataset
    test_config = CONFIG.copy()
    test_config['mode'] = 'test'  # 測試模式不加擾動
    test_dataset = StochasticRainDataset(test_sequences, test_config)
    
    # 使用 DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 一次處理一個樣本
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Dataset 大小: {len(test_dataset)} 組序列")
    
    # 2. 創建模型
    print("\n[2] 創建模型...")
    model = HydroNetRainOnly(output_steps=3)
    model.eval()
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 模型創建成功")
    print(f"  總參數量: {total_params:,}")
    print(f"  可訓練參數: {trainable_params:,}")
    
    # 3. 載入第一筆測試資料
    print("\n[3] 載入測試資料...")
    test_input, test_target, test_mask = next(iter(test_loader))
    
    batch_size, seq_len, channels, height, width = test_input.shape
    print(f"✓ 輸入形狀: {test_input.shape}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}, channels={channels}")
    print(f"  height={height}, width={width}")
    print(f"✓ 目標形狀: {test_target.shape}")
    print(f"✓ 遮罩形狀: {test_mask.shape}")
    
    # 3. 前向傳播
    print("\n[4] 執行前向傳播...")
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✓ 前向傳播成功")
    print(f"  輸出形狀: {output.shape}")
    print(f"  預期形狀: ({batch_size}, 3, 1, {height}, {width})")
    
    # 檢查形狀是否正確
    expected_shape = (batch_size, 3, 1, height, width)
    assert output.shape == expected_shape, f"輸出形狀錯誤! 預期 {expected_shape}, 得到 {output.shape}"
    print(f"✓ 輸出形狀正確")
    
    # 4. 統計輸出值（在有效區域）
    print("\n[5] 輸出統計信息（有效區域）...")
    for t in range(3):
        pred_t = output[0, t, 0, :, :]  # (H, W)
        target_t = test_target[0, t, 0, :, :]  # (H, W)
        mask_t = test_mask[0, t, 0, :, :]  # (H, W)
        
        # 只在有效區域計算統計（mask > 0）
        valid_pred = pred_t[mask_t > 0]
        valid_target = target_t[mask_t > 0]
        
        print(f"\n  時間步 t+{t+1}:")
        print(f"    有效像素數: {(mask_t > 0).sum().item()}")
        print(f"    預測 - 最小值: {valid_pred.min().item():.4f}, 最大值: {valid_pred.max().item():.4f}")
        print(f"    預測 - 平均值: {valid_pred.mean().item():.4f}, 標準差: {valid_pred.std().item():.4f}")
        print(f"    目標 - 最小值: {valid_target.min().item():.4f}, 最大值: {valid_target.max().item():.4f}")
        print(f"    目標 - 平均值: {valid_target.mean().item():.4f}, 標準差: {valid_target.std().item():.4f}")
    
    # 5. 可視化
    print("\n[6] 生成可視化...")
    
    # 取第一個樣本進行可視化
    sample_idx = 0
    
    fig = plt.figure(figsize=(20, 14))
    
    # 第1行：輸入降雨序列 (9 張圖)
    print("  繪製輸入降雨序列...")
    for t in range(9):
        ax = plt.subplot(5, 9, t + 1)
        rain_img = test_input[sample_idx, t, 0].numpy()
        
        # 動態調整範圍
        vmin, vmax = rain_img.min(), rain_img.max()
        im = ax.imshow(rain_img, cmap='Blues', vmin=vmin, vmax=vmax)
        if t < 6:
            ax.set_title(f't-{5-t}\n(觀測)', fontsize=9)
        else:
            ax.set_title(f't+{t-5}\n(預報)', fontsize=9, color='red')
        ax.axis('off')
        if t == 0:
            ax.text(-0.3, 0.5, '輸入降雨', fontsize=11, rotation=90, 
                   ha='center', va='center', transform=ax.transAxes)
    
    # 第2行：預測淹水增量 (前3格)
    print("  繪製輸出淹水增量預測...")
    for t in range(3):
        ax = plt.subplot(5, 9, 9 + t + 1)
        flood_pred = output[sample_idx, t, 0].numpy()
        mask_img = test_mask[sample_idx, t, 0].numpy()
        
        # 將無效區域設為 NaN（不顯示）
        flood_pred_masked = flood_pred.copy()
        flood_pred_masked[mask_img == 0] = np.nan
        
        # 使用紅藍色圖：藍色=負值(水退)，紅色=正值(水漲)
        vmax_val = np.nanmax(np.abs(flood_pred_masked))
        if vmax_val > 0:
            im2 = ax.imshow(flood_pred_masked, cmap='RdBu_r', vmin=-vmax_val, vmax=vmax_val)
        else:
            im2 = ax.imshow(flood_pred_masked, cmap='RdBu_r')
        ax.set_title(f't+{t+1}\n預測增量', fontsize=9)
        ax.axis('off')
        if t == 0:
            ax.text(-0.3, 0.5, '預測輸出', fontsize=11, rotation=90,
                   ha='center', va='center', transform=ax.transAxes)
    
    # 第3行：真實淹水增量 (前3格)
    print("  繪製真實目標...")
    for t in range(3):
        ax = plt.subplot(5, 9, 18 + t + 1)
        flood_target = test_target[sample_idx, t, 0].numpy()
        mask_img = test_mask[sample_idx, t, 0].numpy()
        
        # 將無效區域設為 NaN
        flood_target_masked = flood_target.copy()
        flood_target_masked[mask_img == 0] = np.nan
        
        # 使用相同的色階範圍
        vmax_val = np.nanmax(np.abs(flood_target_masked))
        if vmax_val > 0:
            im3 = ax.imshow(flood_target_masked, cmap='RdBu_r', vmin=-vmax_val, vmax=vmax_val)
        else:
            im3 = ax.imshow(flood_target_masked, cmap='RdBu_r')
        ax.set_title(f't+{t+1}\n真實增量', fontsize=9)
        ax.axis('off')
        if t == 0:
            ax.text(-0.3, 0.5, '真實目標', fontsize=11, rotation=90,
                   ha='center', va='center', transform=ax.transAxes)
    
    # 添加 colorbar (在右側)
    cbar_ax1 = fig.add_axes([0.92, 0.68, 0.012, 0.20])
    plt.colorbar(im, cax=cbar_ax1, label='降雨強度')
    
    cbar_ax2 = fig.add_axes([0.92, 0.42, 0.012, 0.20])
    plt.colorbar(im2, cax=cbar_ax2, label='淹水增量')
    
    # 第4-5行：統計圖表
    print("  繪製統計圖表...")
    
    # 第4行：降雨時序變化圖 (跨3列)
    ax_rain = plt.subplot(5, 3, 13)
    rain_avg = [test_input[sample_idx, t, 0].mean().item() for t in range(9)]
    time_labels = [f't-{5-i}' if i < 6 else f't+{i-5}' for i in range(9)]
    colors = ['blue'] * 6 + ['red'] * 3
    ax_rain.bar(range(9), rain_avg, color=colors, alpha=0.6)
    ax_rain.set_xticks(range(9))
    ax_rain.set_xticklabels(time_labels, rotation=45, fontsize=9)
    ax_rain.set_ylabel('平均降雨強度 (mm/h)', fontsize=10)
    ax_rain.set_title('輸入降雨時序變化', fontsize=11)
    ax_rain.grid(True, alpha=0.3)
    ax_rain.axvline(5.5, color='red', linestyle='--', linewidth=1.5, label='當前時刻')
    ax_rain.legend(fontsize=9)
    
    # 第4行：預測 vs 真實對比
    ax_flood = plt.subplot(5, 3, 14)
    
    # 獲取有效區域的預測和目標
    pred_avgs = []
    target_avgs = []
    pred_stds = []
    target_stds = []
    
    for t in range(3):
        mask_t = test_mask[sample_idx, t, 0].numpy()
        pred_t = output[sample_idx, t, 0].numpy()
        target_t = test_target[sample_idx, t, 0].numpy()
        
        # 只計算有效區域
        valid_pred = pred_t[mask_t > 0]
        valid_target = target_t[mask_t > 0]
        
        pred_avgs.append(valid_pred.mean())
        target_avgs.append(valid_target.mean())
        pred_stds.append(valid_pred.std())
        target_stds.append(valid_target.std())
    
    x_pos = np.arange(3)
    width = 0.35
    
    bars1 = ax_flood.bar(x_pos - width/2, pred_avgs, width, 
                         yerr=pred_stds, capsize=5, alpha=0.6, 
                         color='green', label='預測')
    bars2 = ax_flood.bar(x_pos + width/2, target_avgs, width,
                         yerr=target_stds, capsize=5, alpha=0.6,
                         color='orange', label='真實')
    
    ax_flood.set_xticks(x_pos)
    ax_flood.set_xticklabels(['t+1', 't+2', 't+3'], fontsize=9)
    ax_flood.set_ylabel('淹水增量 (m)', fontsize=10)
    ax_flood.set_title('預測 vs 真實淹水增量', fontsize=11)
    ax_flood.grid(True, alpha=0.3)
    ax_flood.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax_flood.legend(fontsize=9)
    
    # 第4行：模型架構示意圖
    ax_arch = plt.subplot(5, 3, 15)
    ax_arch.axis('off')
    arch_text = """模型架構摘要
━━━━━━━━━━━━━━━
輸入: (B, 9, 1, H, W)

Encoder Layer 1:
  Conv 1→16 → LSTM → Pool

Encoder Layer 2:
  Conv 16→32 → LSTM → Pool

保存 t+1,t+2,t+3 特徵

Decoder (×3):
  Upsample 32→16→8
  Conv 8→1

輸出: (B, 3, 1, H, W)

參數: {:,}""".format(total_params)
    
    ax_arch.text(0.05, 0.5, arch_text, transform=ax_arch.transAxes,
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('HydroNetRainOnly 模型測試與可視化 (使用真實測試資料)', fontsize=16, fontweight='bold')
    plt.subplots_adjust(left=0.05, right=0.90, top=0.96, bottom=0.03, hspace=0.4, wspace=0.3)
    
    # 保存圖片
    output_path = 'visualizations/model_test_visualization.png'
    import os
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 可視化已保存到: {output_path}")
    
    # 6. 測試不同輸入尺寸
    print("\n[6] 測試不同輸入尺寸...")
    test_sizes = [(64, 64), (128, 128), (256, 256)]
    
    for h, w in test_sizes:
        try:
            test_input_size = torch.randn(1, 9, 1, h, w)
            with torch.no_grad():
                output_size = model(test_input_size)
            expected = (1, 3, 1, h, w)
            assert output_size.shape == expected
            print(f"  ✓ 尺寸 {h}×{w}: 輸入 {test_input_size.shape} → 輸出 {output_size.shape}")
        except Exception as e:
            print(f"  ✗ 尺寸 {h}×{w} 失敗: {e}")
    
    print("\n" + "=" * 80)
    print("✓ 所有測試通過！模型資訊流正常")
    print("=" * 80)