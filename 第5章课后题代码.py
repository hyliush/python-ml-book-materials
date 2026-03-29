# 1. 导入工具库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 2. 加载与预处理数据
data = pd.read_csv("./data/customer_consumption_data.csv")
# 划分特征类型
numeric_features = ["消费频次", "平均客单价", "复购周期"]
cat_features = ["消费品类", "会员等级"]
# 构建预处理流水线
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])
X_processed = preprocessor.fit_transform(data)

# 3. K-Means聚类（假设最优K=4）
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(X_processed)
data["聚类标签"] = cluster_labels

# 4. 聚类效果评估（轮廓系数）
sil_score = silhouette_score(X_processed, cluster_labels)
print(f"聚类轮廓系数：{sil_score:.4f}")

# 5. 降维至2维实现聚类结果可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7)
plt.title("客户消费数据K-Means聚类结果可视化")
plt.xlabel("主成分1")
plt.ylabel("主成分2")
plt.colorbar(label="聚类簇标签")
plt.show()