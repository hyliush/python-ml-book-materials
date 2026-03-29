# 导入依赖库
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

oil_data = pd.read_csv("./data/yf_oil_data.csv")
# 筛选收盘价并简化列名，将日期从索引转为列（便于后续时间匹配）
df = oil_data[['Date', 'Close']]
df.columns = ['date', 'price']
df["date"] = pd.to_datetime(df["date"])
# 初步探索数据特征
print("=== WTI原油价格数据概况 ===")
print(f"数据覆盖时间：{df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
print(f"总交易日数量：{len(df)} 个")
print(f"缺失值数量：{df.isnull().sum().sum()} 个")
print("\n=== 价格统计特征（美元/桶） ===")
print(df['price'].describe().round(2))

# 绘制原始价格趋势图，直观观察波动规律
plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['price'], color='#1f77b4', linewidth=1.5)
plt.title('2010-2023年WTI原油期货收盘价趋势', fontsize=14)
plt.xlabel('日期', fontsize=12)
plt.ylabel('价格（美元/桶）', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

from sklearn.preprocessing import MinMaxScaler

# 初始化归一化器，设定缩放范围为[0,1]
scaler = MinMaxScaler(feature_range=(0, 1))

# 对价格列进行归一化（模型要求输入为二维数组，故用reshape(-1,1)转换）
df['scaled_price'] = scaler.fit_transform(df[['price']])

# 查看归一化前后的对比，验证处理效果
print("=== 归一化前后数据对比 ===")
comparison = df[['date', 'price', 'scaled_price']].sample(5, random_state=42)
print(comparison.round(4))

import numpy as np

def create_sequences(data, window_size):
    """
    将归一化后的时序数据转化为样本集
    data：二维数组格式的归一化价格数据
    window_size：滑动窗口大小，即使用前window_size天预测第window_size+1天
    """
    X, y = [], []
    # 从window_size开始循环，确保每个样本有足够的历史数据支撑
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])  # 输入特征：前window_size天的价格
        y.append(data[i, 0])                  # 目标值：第i天的价格
    return np.array(X), np.array(y)

# 提取归一化后的价格数据，转为二维数组
scaled_data = df['scaled_price'].values.reshape(-1, 1)

# 设定窗口大小为30，生成样本集
window_size = 30
X, y = create_sequences(scaled_data, window_size)

# 查看样本集形状，确认处理结果符合预期
print(f"=== 样本集结构 ===")
print(f"输入特征 X 形状： {X.shape} （样本数： {X.shape[0]}, 时间步： {X.shape[1]}）")
print(f"目标值y形状：{y.shape}（样本数：{y.shape[0]}）")

# 按8:2的比例划分训练集与测试集
train_ratio = 0.8
train_size = int(len(X) * train_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 调整LSTM输入格式：增加“特征数”维度（此处仅用价格一个特征，故特征数=1）
X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 输出划分结果，确认数据量与格式正确性
print(f"=== 训练集与测试集划分结果 ===")
print(f"训练集样本数：{len(X_train)}（占比{train_ratio*100:.0f}%）")
print(f"测试集样本数：{len(X_test)}（占比{(1-train_ratio)*100:.0f}%）")
print(f"LSTM训练集输入形状：{X_train_lstm.shape}（样本数, 时间步, 特征数）")
print(f"LSTM测试集输入形状：{X_test_lstm.shape}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 初始化并训练线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 生成测试集预测结果
y_pred_lr = lr_model.predict(X_test)

# 将归一化结果还原为原始价格（模型评估需基于实际业务尺度）
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_lr_original = scaler.inverse_transform(y_pred_lr.reshape(-1, 1))

# 计算评估指标
lr_mse = mean_squared_error(y_test_original, y_pred_lr_original)
lr_mae = mean_absolute_error(y_test_original, y_pred_lr_original)

print("=== 线性回归模型评估结果 ===")
print(f"均方误差（MSE）：{lr_mse:.2f}")
print(f"平均绝对误差（MAE）：{lr_mae:.2f} 美元/桶")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 构建LSTM模型
lstm_model = Sequential(name="WTI_Oil_LSTM")
# 隐层：50个神经元，输入形状为(时间步, 特征数)
lstm_model.add(
    LSTM(
        units=50,
        return_sequences=False,  # 最后一层LSTM无需返回序列，仅输出最终状态
        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]),
        name="LSTM_Hidden_Layer"
    )
)
# 输出层：1个神经元，对应单步价格预测
lstm_model.add(Dense(units=1, name="Output_Layer"))

# 编译模型
lstm_model.compile(optimizer="adam", loss="mean_squared_error")

# 查看模型结构，确认各层参数与输入输出格式
print("=== LSTM模型结构 ===")
lstm_model.summary()

# 训练模型：epochs=20（平衡训练效果与过拟合风险），batch_size=32（平衡训练速度与稳定性）
print("\n=== 开始训练LSTM模型 ===")
history = lstm_model.fit(
    x=X_train_lstm,
    y=y_train,
    epochs=20,
    batch_size=32,
    verbose=1,
    validation_data=(X_test_lstm, y_test)
)

# 生成测试集预测结果
y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0)

# 还原为原始价格并评估
y_pred_lstm_original = scaler.inverse_transform(y_pred_lstm)
lstm_mse = mean_squared_error(y_test_original, y_pred_lstm_original)
lstm_mae = mean_absolute_error(y_test_original, y_pred_lstm_original)

print("\n=== LSTM模型评估结果 ===")
print(f"均方误差（MSE）：{lstm_mse:.2f}")
print(f"平均绝对误差（MAE）：{lstm_mae:.2f} 美元/桶")

# 提取测试集对应的日期（确保时间轴准确）
test_date_start = train_size + window_size
test_dates = df['date'].iloc[test_date_start:].values

# 创建画布，设置合适尺寸
plt.figure(figsize=(14, 8))

# 绘制三条核心曲线
plt.plot(test_dates, y_test_original, color='#1f77b4', linewidth=2, label='真实价格')
plt.plot(test_dates, y_pred_lr_original, color='#ff7f0e', linewidth=1.5, linestyle='--', label='线性回归预测')
plt.plot(test_dates, y_pred_lstm_original, color='#2ca02c', linewidth=1.5, linestyle='-.', label='LSTM预测')

# 优化图表样式
plt.title('WTI原油价格预测结果对比（测试集）', fontsize=16)
plt.xlabel('日期', fontsize=12)
plt.ylabel('价格（美元/桶）', fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xticks(rotation=45)  # 旋转日期标签，避免重叠
plt.tight_layout()  # 自动调整布局，防止标签截断

# 保存图表（可选，便于后续报告使用）
plt.savefig('oil_price_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
