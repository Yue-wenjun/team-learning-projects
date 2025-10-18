import torch
import torch.nn as nn


class CSPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, use_wear_net=True):
        super(CSPricePredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_wear_net = use_wear_net
        
        # 主LSTM处理时序特征
        if use_wear_net:
            # 如果使用磨损网络，则LSTM处理除磨损外的所有特征
            self.lstm = nn.LSTM(input_size-1, hidden_size, num_layers, batch_first=True, dropout=0.2)
        else:
            # 否则处理所有特征
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # 磨损特征处理网络(因交易的物品磨损数据未获得)
        if use_wear_net:
            self.wear_net = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 64),
                nn.ReLU()
            )
            # 融合层：LSTM输出 + 磨损网络输出
            fusion_input_size = hidden_size + 64
        else:
            fusion_input_size = hidden_size
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )
        
        # 批量归一化
        self.batch_norm = nn.BatchNorm1d(fusion_input_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        if self.use_wear_net:
            # 分离磨损特征和其他特征
            wear_features = x[:, :, -1:]  # 假设磨损是最后一个特征
            other_features = x[:, :, :-1]
            
            # 处理主要时序特征
            lstm_out, _ = self.lstm(other_features)
            lstm_final = lstm_out[:, -1, :]  # 取最后一个时间步
            
            # 处理磨损特征（使用最后一个时间步的磨损值）
            wear_out = self.wear_net(wear_features[:, -1, :])
            
            # 特征融合
            combined = torch.cat([lstm_final, wear_out], dim=1)
        else:
            # 直接处理所有特征
            lstm_out, _ = self.lstm(x)
            combined = lstm_out[:, -1, :]
        
        # 批量归一化
        combined = self.batch_norm(combined)
        
        # 最终输出
        output = self.output_layers(combined)
        return output
if __name__ == "__main__":
    # 参数设置
    input_size = 24  # 特征数量
    hidden_size = 128  # 隐藏单元数
    num_layers = 3  # LSTM层数
    output_size = 1  # 输出维度
    batch_size = 64  # 批次大小
    time_step = 30  # 时间步长
   
    print("测试CS饰品价格预测模型")
    model = CSPricePredictor(input_size, hidden_size, num_layers, output_size, use_wear_net=True)
    test_input = torch.randn(batch_size, time_step, input_size)
    # 将模型设置为评估模式，并检查输出 
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        print(f"输入维度: {test_input.shape}")
        print(f"输出维度: {output.shape}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    # 保存模型    
    torch.save(model.state_dict(), 'cs_price_predictor.pth')
    print("\n模型已保存为 'cs_price_predictor.pth'")