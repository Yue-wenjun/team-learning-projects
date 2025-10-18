import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime, timedelta
# 因爬虫还未掌握完全问题，未能找到有效获取需要数据手段（如交易数，磨损等）故在此使用ai进行辅助获取或生成数据
class CSDataset(Dataset):
    """CS饰品价格数据集类"""
    
    def __init__(self, csv_file, time_step=30, transform=None, target_col='price'):
        """
            csv_file: CS饰品数据CSV文件路径
            time_step: 时间步长，用多少历史数据预测未来
            transform: 数据转换函数
            target_col: 目标列名（要预测的价格列）
        """
        self.time_step = time_step
        self.transform = transform
        self.target_col = target_col
        
        # 加载数据
        self.df = self.load_data(csv_file)
        
        # 准备特征和目标
        self.features, self.targets, self.feature_names = self.prepare_features()
        
        # 创建序列
        self.X, self.Y = self.create_sequences()
        
    def load_data(self, csv_file):
        """加载和基础数据清洗"""
        print(f"加载数据文件: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # 基础数据清洗
        df = self.clean_data(df)
        
        # 特征工程
        df = self.feature_engineering(df)
        
        return df
    
    def clean_data(self, df):
        """数据清洗"""
        # 确保时间戳列存在并排序
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 移除明显的异常值（价格不能为负）
        if 'price' in df.columns:
            df = df[df['price'] > 0]
        
        print(f"数据清洗完成，剩余样本数: {len(df)}")
        return df
    
    def feature_engineering(self, df):
        """特征工程 - 创建技术指标和衍生特征"""
        
        # 基础价格特征
        if 'price' in df.columns:
            # 价格变化率
            df['price_change'] = df['price'].pct_change().fillna(0)
            
            # 价格波动率（滚动标准差）
            df['price_volatility'] = df['price'].rolling(window=7, min_periods=1).std().fillna(0)
            
            # 移动平均线
            df['price_ma_7'] = df['price'].rolling(window=7, min_periods=1).mean().fillna(method='bfill')
            df['price_ma_30'] = df['price'].rolling(window=30, min_periods=1).mean().fillna(method='bfill')
            
            # 价格动量
            df['momentum_5'] = df['price'] - df['price'].shift(5)
            df['momentum_5'] = df['momentum_5'].fillna(0)
            
            # 价格在历史区间的位置
            df['price_min_30'] = df['price'].rolling(window=30, min_periods=1).min().fillna(method='bfill')
            df['price_max_30'] = df['price'].rolling(window=30, min_periods=1).max().fillna(method='bfill')
            df['price_position'] = (df['price'] - df['price_min_30']) / (df['price_max_30'] - df['price_min_30'] + 1e-8)
        
        # 交易量特征
        if 'volume' in df.columns:
            # 交易量变化率
            df['volume_change'] = df['volume'].pct_change().fillna(0)
            
            # 交易量移动平均
            df['volume_ma_7'] = df['volume'].rolling(window=7, min_periods=1).mean().fillna(method='bfill')
            df['volume_ratio'] = df['volume'] / (df['volume_ma_7'] + 1e-8)
        
        # 磨损特征处理
        if 'wear' in df.columns:
            # 磨损值通常在0-1之间，但我们可以创建一些衍生特征
            df['wear_category'] = pd.cut(df['wear'], 
                                       bins=[0, 0.07, 0.15, 0.38, 1.0], 
                                       labels=[0, 1, 2, 3])  # 分类磨损
        
        # 处理分类变量（如果有的话）
        categorical_columns = ['rarity', 'weapon_type', 'quality', 'wear_category']
        for col in categorical_columns:
            if col in df.columns:
                # 简单的标签编码
                df[col + '_encoded'] = pd.factorize(df[col])[0]
        
        print("特征工程完成")
        return df
    
    def prepare_features(self):
        """准备用于训练的特征和目标"""
        
        # 选择数值型特征列
        numeric_columns = [
            'price', 'wear', 'volume', 'price_change', 'volume_change',
            'price_volatility', 'price_ma_7', 'price_ma_30', 'momentum_5',
            'price_position', 'volume_ratio'
        ]
        
        # 选择编码后的分类特征
        encoded_columns = [col for col in self.df.columns if col.endswith('_encoded')]
        
        # 合并所有特征列
        feature_columns = [col for col in numeric_columns + encoded_columns 
                          if col in self.df.columns and col != self.target_col]
        
        # 确保目标列存在
        if self.target_col not in self.df.columns:
            raise ValueError(f"目标列 '{self.target_col}' 不存在于数据中")
        
        # 提取特征和目标
        features = self.df[feature_columns].values.astype(np.float32)
        targets = self.df[self.target_col].values.astype(np.float32)
        
        print(f"特征数量: {len(feature_columns)}")
        print(f"特征名称: {feature_columns}")
        print(f"总样本数: {len(features)}")
        
        return features, targets, feature_columns
    
    def create_sequences(self):
        """创建时间序列数据"""
        X, Y = [], []
        
        for i in range(len(self.features) - self.time_step):
            X.append(self.features[i:(i + self.time_step)])
            Y.append(self.targets[i + self.time_step])
        
        X = np.array(X)
        Y = np.array(Y)
        
        print(f"创建序列完成: X.shape={X.shape}, Y.shape={Y.shape}")
        return X, Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.Y[idx], dtype=torch.float32)
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, target
    
    def get_feature_names(self):
        """获取特征名称"""
        return self.feature_names


def get_data_loaders(csv_file, time_step=30, batch_size=32, train_ratio=0.7, val_ratio=0.15):
    """
    获取训练、验证、测试数据加载器
    
    参数:
        csv_file: 数据文件路径
        time_step: 时间步长
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    from torch.utils.data import random_split
    
    # 创建完整数据集
    dataset = CSDataset(csv_file, time_step=time_step)
    
    # 计算分割大小
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"数据分割: 训练集 {train_size}, 验证集 {val_size}, 测试集 {test_size}")
    
    return train_loader, val_loader, test_loader, dataset.get_feature_names()


def create_sample_data(save_path='cs_sample_data.csv', num_samples=1000):
    """
    创建示例CS饰品数据（用于测试）
    """
    np.random.seed(42)
    
    # 生成时间序列
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(num_samples)]
    
    # 生成模拟数据
    base_price = 100
    prices = []
    for i in range(num_samples):
        # 随机游走价格
        if i == 0:
            price = base_price
        else:
            change = np.random.normal(0, 5)
            price = max(10, prices[-1] + change)
        prices.append(price)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'wear': np.random.uniform(0, 1, num_samples),  # 磨损值 0-1
        'volume': np.random.poisson(50, num_samples),  # 交易量
        'rarity': np.random.choice(['Consumer', 'Industrial', 'Mil-Spec', 'Restricted', 'Classified'], 
                                 num_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),  # 稀有度
        'weapon_type': np.random.choice(['AK-47', 'M4A4', 'AWP', 'Desert Eagle', 'Knife'], 
                                      num_samples)  # 武器类型
    })
    
    # 保存到CSV
    df.to_csv(save_path, index=False)
    print(f"示例数据已保存到: {save_path}")
    return df


# 使用示例
if __name__ == "__main__":
    # 创建示例数据（如果没有真实数据）
    sample_file = 'csgo_prices.csv'
    if not os.path.exists(sample_file):
        print("创建示例数据...")
        create_sample_data(sample_file, num_samples=500)
    
    # 测试数据加载
    print("\n测试数据加载...")
    try:
        train_loader, val_loader, test_loader, feature_names = get_data_loaders(
            csv_file=sample_file,
            time_step=30,
            batch_size=32
        )
        
        # 查看一个批次的数据
        for batch_X, batch_Y in train_loader:
            print(f"批次特征维度: {batch_X.shape}")  # [batch_size, time_step, num_features]
            print(f"批次目标维度: {batch_Y.shape}")  # [batch_size]
            print(f"特征名称: {feature_names}")
            break
            
    except Exception as e:
        print(f"数据加载错误: {e}")
        print("请确保CSV文件格式正确，包含必要的列")