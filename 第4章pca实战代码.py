import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv("./data/loanrisk.csv")
# 分离特征与标签
X = data.iloc[:, :-1].values  # 10维特征
y = data["风险等级"].values    # 风险标签

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 标准化后的特征

# 应用PCA降维至2维（便于可视化）
pca = PCA(n_components=2)  # 保留2个主成分
X_pca = pca.fit_transform(X_scaled)  # 降维后的特征

# 查看主成分解释的方差比例
print(f"\n主成分解释方差比例：{pca.explained_variance_ratio_}")
print(f"累计解释方差比例：{np.sum(pca.explained_variance_ratio_):.4f}")

import matplotlib.pyplot as plt
# 设置全局字体为 SimHei (黑体) 或其他中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

# 构建降维后的数据框
pca_df = pd.DataFrame(X_pca, columns=["主成分1", "主成分2"])
pca_df["风险等级"] = y

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=pca_df,
    x="主成分1",
    y="主成分2",
    hue="风险等级",
    palette=["green", "orange", "red"],  # 低/中/高风险对应绿/橙/红
    s=50,  # 点大小
    alpha=0.7  # 透明度
)
plt.title("PCA降维后客户信用特征分布（2维）", fontsize=14)
plt.xlabel(f"主成分1（解释方差：{pca.explained_variance_ratio_[0]:.2%}）", fontsize=12)
plt.ylabel(f"主成分2（解释方差：{pca.explained_variance_ratio_[1]:.2%}）", fontsize=12)
plt.legend(title="风险等级")
plt.savefig("./figure/kehu.png")
plt.show()

# 082916
# 原始特征建模
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model_original = LogisticRegression(max_iter=1000)
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)

# 降维特征建模
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
model_pca = LogisticRegression(max_iter=1000)
model_pca.fit(X_pca_train, y_pca_train)
y_pred_pca = model_pca.predict(X_pca_test)

# 对比模型效果
print("\n原始特征模型分类报告：")
print(classification_report(y_test, y_pred_original))

print("\nPCA降维特征模型分类报告：")
print(classification_report(y_pca_test, y_pred_pca))