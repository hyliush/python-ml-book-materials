# 1. 导入工具库
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from imblearn.over_sampling import SMOTE

# 2. 加载与预处理数据
data = pd.read_csv("./data/credit_risk_data.csv")
# -------------------- 第一步：处理缺失值 --------------------
# 分析缺失特征类型：收入（连续型）用中位数填充（抗极端值），教育程度（离散型）用众数填充
# 计算填充值
income_median = data["income"].median()  # 收入中位数
education_mode = data["education"].mode()[0]  # 教育程度众数（取第一个众数）

# 填充缺失值
data["income"].fillna(income_median, inplace=True)
data["education"].fillna(education_mode, inplace=True)

X = data[["income", "education", "credit_score", "overdue_times", "loan_amount", "loan_term", "debt_income_ratio"]]
y = data["is_default"]  # 1=违约，0=不违约

# 缺失值处理
X.fillna(X.median(), inplace=True)
# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 样本均衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# 3. 构建MLP模型
model = Sequential([
    Dense(16, activation="relu", input_shape=(7,)),  # 输入层+第一层隐藏层
    Dropout(0.2),  # Dropout缓解过拟合
    Dense(8, activation="relu"),  # 第二层隐藏层
    Dense(1, activation="sigmoid")  # 输出层
])

# 编译模型
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 早停策略
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# 训练模型
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# 4. 模型评估
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
print(f"准确率：{accuracy_score(y_test, y_pred):.4f}")
print(f"违约样本召回率：{recall_score(y_test, y_pred):.4f}")