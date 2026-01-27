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