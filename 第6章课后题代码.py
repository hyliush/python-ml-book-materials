# 1. 导入工具库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 2. 加载与预处理数据
data = pd.read_csv("./data/monthly_sales_data.csv")
X = data[["广告投入", "线下门店数量", "竞品价格", "节假日天数", "居民可支配收入"]]
y = data["月度销售额"]

# 划分数据集（训练集70%，测试集30%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 模型训练（带L2正则化的Ridge回归，缓解过拟合/多重共线性）
model = Ridge(alpha=1.0)  # alpha为正则化强度，越大惩罚越重
model.fit(X_train_scaled, y_train)

# 4. 模型评估（R²、MSE）
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
print(f"训练集R²系数：{r2_score(y_train, y_train_pred):.4f}")
print(f"测试集R²系数：{r2_score(y_test, y_test_pred):.4f}")
print(f"测试集均方误差MSE：{mean_squared_error(y_test, y_test_pred):.4f}")

# 5. 新数据预测（需与训练数据同尺度）
new_data = np.array([[50, 20, 80, 5, 3000]])  # 示例新数据：广告投入等5个特征
new_data_scaled = scaler.transform(new_data)
sales_pred = model.predict(new_data_scaled)
print(f"预测月度销售额：{sales_pred[0]:.2f}")
