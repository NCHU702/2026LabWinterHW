import torch
import os

"""
這個檔案負責管理所有的超參數，若要調整 Batch Size 或學習率，只需修改這裡。
"""

CONFIG = {
    # 路徑設定
    'train_data_dir': 'train_data',  
    'save_dir': 'checkpoints',
    
    # 序列長度
    'input_seq_len': 9,      # 6 (過去) + 3 (未來預報)
    'output_seq_len': 3,     # 預測未來 3 小時
    
    # 訓練參數
    'batch_size': 2,
    'learning_rate': 1e-4,
    'num_epochs': 64,
    
    # 資料處理
    'pad_multiple': 4,       # 尺寸需為 4 的倍數
    
    # 硬體
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 自動建立存檔目錄
if not os.path.exists(CONFIG['save_dir']):
    os.makedirs(CONFIG['save_dir'])

for key, value in CONFIG.items():
    print(f"{key}: {value}")