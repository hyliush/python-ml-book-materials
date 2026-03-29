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
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

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

# 3. 构建MLP模型（结合本章知识点，加入Dropout缓解过拟合）
model = Sequential([
    # 输入层+第一层隐藏层：7个输入特征，16个神经元（特征数的2倍，避免过少导致欠拟合）
    Dense(16, activation="relu", input_shape=(7,)),
    Dropout(0.2),  # 随机丢弃20%神经元，缓解过拟合
    # 第二层隐藏层：8个神经元，逐步降维，提取高阶特征
    Dense(8, activation="relu"),
    # 输出层：1个神经元，Sigmoid激活函数，输出违约概率
    Dense(1, activation="sigmoid")
])

# 编译模型：优化器用Adam（自适应学习率，解决学习率不当问题），损失函数用二分类交叉熵
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# 4. 早停策略：监控验证集损失，耐心为3（3轮无改善则停止），恢复最优权重
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# 训练模型：迭代20次，批次大小32，验证集比例20%
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,  # 从训练集中划分20%作为验证集
    callbacks=[early_stop],  # 加入早停策略
    verbose=1  # 显示训练过程
)


# 5. 绘制Loss曲线（训练损失 vs 验证损失）
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='训练损失', color='blue')
plt.plot(history.history['val_loss'], label='验证损失', color='red')
plt.xlabel('迭代次数（epochs）')
plt.ylabel('损失值（binary_crossentropy）')
plt.title('训练损失与验证损失曲线')
plt.legend()
plt.show()


# 6. 构建无过拟合缓解的模型（用于演示过拟合）
overfit_model = Sequential([
    Dense(16, activation="relu", input_shape=(7,)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])
overfit_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 不加入早停策略，训练20次
overfit_history = overfit_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 对比两条Loss曲线，演示过拟合
plt.figure(figsize=(8, 4))
plt.plot(overfit_history.history['loss'], label='过拟合模型-训练损失', color='blue', linestyle='--')
plt.plot(overfit_history.history['val_loss'], label='过拟合模型-验证损失', color='red', linestyle='--')
plt.plot(history.history['loss'], label='优化模型-训练损失', color='blue')
plt.plot(history.history['val_loss'], label='优化模型-验证损失', color='red')
plt.xlabel('迭代次数（epochs）')
plt.ylabel('损失值')
plt.title('过拟合模型与优化模型Loss曲线对比')
plt.legend()
plt.show()