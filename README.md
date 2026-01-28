# 洪水淹水預測模型

基於 ConvLSTM 的時序淹水增量預測系統，使用降雨資料預測未來 1-3 小時的淹水變化。

## 環境需求

- Python 3.10+
- NVIDIA GPU (建議 8GB+ VRAM)

## 安裝步驟

### 1. 建立虛擬環境

```bash
python -m venv .venv
```

### 2. 啟動虛擬環境

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Windows CMD:
```cmd
.venv\Scripts\activate.bat
```

Linux/macOS:
```bash
source .venv/bin/activate
```

### 3. 安裝 PyTorch (CUDA 11.8)

### 4. 安裝其他依賴

```bash
pip install numpy pandas matplotlib scipy
```

### 5. 驗證 CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## 資料結構

```
train_data/
    t1/
        rain/           # 降雨 CSV (時序)
        flood/          # 淹水 CSV (時序)
    t2/
    ...
val_data/
    t11/
        rain/
        flood/
test_data/
    t12/
        rain/
        flood/
```

每個 CSV 為 636x772 的網格資料，無效值為 -999.999。

## 訓練

```bash
python train.py
```

訓練過程會輸出:
- `checkpoints/best_model.pth` - 最佳模型權重
- `checkpoints/training_history.json` - 訓練記錄
- `checkpoints/training_curves.png` - 損失曲線圖
- `visualizations/` - 驗證階段預測比較圖

## 配置參數

修改 `config.py` 調整超參數:

| 參數 | 預設值 | 說明 |
|------|--------|------|
| batch_size | 2 | 批次大小 (1個batch 約需 4.5GB VRAM) |
| learning_rate | 1e-4 | 學習率 |
| num_epochs | 64 | 訓練輪數 |
| early_stopping_patience | 10 | 早停耐心值 |
| flood_weight | 50.0 | 淹水區域權重 |
| target_scale | 10.0 | 目標值縮放因子 |

## 模型架構

HydroNetRainOnly: 3 層 ConvLSTM Encoder + Decoder

- 輸入: 9 張降雨圖 (過去 6 小時 + 未來 3 小時預報)
- 輸出: 3 張淹水增量圖 (t+1, t+2, t+3)
- 參數量: ~435,000

詳細架構見 `model_flow_diagram.txt`。

## 檔案說明

| 檔案 | 功能 |
|------|------|
| config.py | 超參數配置 |
| model.py | 模型定義 |
| dataset.py | 資料載入與預處理 |
| train.py | 訓練腳本 |
| utils.py | 工具函數與損失函數 |
