import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 加载信用卡交易数据
card_data = pd.read_csv("./data/credit_card_transactions.csv")

# -------------------------- 第一步：分离特征与目标变量 --------------------------
# 特征变量：排除目标变量is_fraud，其余为输入特征
X = card_data.drop("is_fraud", axis=1)
# 目标变量：仅用于后续模型评估（K-Means为无监督，建模时不使用）
y_true = card_data["is_fraud"]

# -------------------------- 第二步：处理缺失值 --------------------------
# 仅交易金额（amount）有缺失值，用中位数填充（抗大额欺诈交易的极端值影响）
amount_median = X["amount"].median()
X["amount"].fillna(amount_median, inplace=True)

# -------------------------- 第三步：特征预处理（离散编码+连续标准化） --------------------------
# 区分特征类型：连续特征（需标准化）、离散特征（需独热编码）
continuous_features = ["amount", "hour", "freq_24h"]  # 交易金额、时间、频次（数值型）
categorical_features = ["merchant_type", "payment_method"]  # 商户类型、支付方式（离散型）

# 构建预处理流水线：对不同类型特征应用不同处理
# 连续特征：标准化（均值=0，标准差=1，消除量纲）
# 离散特征：独热编码（将类别转换为二进制向量，避免数值大小误导距离计算）
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuous_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)  # drop="first"避免多重共线性
    ])

# 执行预处理：生成最终建模用的特征矩阵
X_processed = preprocessor.fit_transform(X)

# 输出预处理后的数据信息
print(f"预处理前特征数：{X.shape[1]}")
print(f"预处理后特征数：{X_processed.shape[1]}")  # 独热编码后特征数增加
print(f"预处理后数据形状（交易数×处理后特征数）：{X_processed.shape}")  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 第一步：训练K-Means模型（k=1） --------------------------
# 簇数k=1的原因：信用卡欺诈检测中，正常交易占绝大多数，会形成单一紧密簇；
# 欺诈交易因特征异常，会分散在簇外，通过距离判定
kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)  # n_init=10确保聚类稳定
kmeans.fit(X_processed)  # 用全部预处理数据训练（无监督，不区分正常/欺诈）

# -------------------------- 第二步：计算样本到簇中心的距离 --------------------------
# 欧氏距离：衡量每个交易样本与正常交易簇中心的偏离程度，距离越大越可能是欺诈
distances = np.linalg.norm(X_processed - kmeans.cluster_centers_, axis=1)

# 加入距离到原数据框，便于后续分析
card_data_with_dist = card_data.copy()
card_data_with_dist["distance_to_center"] = distances

# 可视化距离分布（观察正常与欺诈交易的距离差异）
plt.figure(figsize=(10, 5))
# 子图1：整体距离分布直方图
plt.subplot(1, 2, 1)
plt.hist(distances, bins=50)
plt.xlabel("样本到簇中心的欧氏距离")
plt.ylabel("交易次数")
plt.title("K-Means聚类后样本距离分布")

# 子图2：正常与欺诈交易的距离箱线图（直观对比）
plt.subplot(1, 2, 2)
box_data = [
    card_data_with_dist[card_data_with_dist["is_fraud"]==0]["distance_to_center"],
    card_data_with_dist[card_data_with_dist["is_fraud"]==1]["distance_to_center"]
]
plt.boxplot(box_data, labels=["正常交易", "欺诈交易"])
plt.ylabel("样本到簇中心的欧氏距离")
plt.title("正常与欺诈交易的距离对比")
plt.tight_layout()
plt.show()

# -------------------------- 第三步：确定欺诈检测阈值 --------------------------
# 阈值选择逻辑：基于距离分布的95%分位数（覆盖95%正常交易，将剩余5%高距离样本判定为欺诈）
# 该阈值可根据业务需求调整（如希望减少漏判欺诈，可降低阈值；减少误判正常，可提高阈值）
threshold = np.percentile(distances, 95)
print(f"\n欺诈检测阈值（距离95%分位数）：{threshold:.4f}")

# 判定欺诈：距离>阈值为欺诈（1），否则为正常（0）
y_pred = (distances > threshold).astype(int)

# -------------------------- 第四步：模型评估（基于真实标签） --------------------------
# 欺诈检测核心指标：召回率（Recall，不漏掉欺诈）、精确率（Precision，减少误判正常）、F1值
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\nK-Means聚类欺诈检测模型评估指标：")
print(f"精确率（Precision）：{precision:.4f}（预测为欺诈的样本中实际是欺诈的比例）")
print(f"召回率（Recall）：{recall:.4f}（实际欺诈样本中被正确检测出的比例）")
print(f"F1值：{f1:.4f}（精确率与召回率的平衡）")

# 混淆矩阵可视化（展示四类判定结果：TP、TN、FP、FN）
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测正常（0）', '预测欺诈（1）'],
            yticklabels=['真实正常（0）', '真实欺诈（1）'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('K-Means欺诈检测混淆矩阵')
plt.tight_layout()
plt.show()

# 输出误判与漏判分析
fp_count = cm[0, 1]  # 误判：正常交易被判定为欺诈
fn_count = cm[1, 0]  # 漏判：欺诈交易被判定为正常
print(f"\n业务分析：")
print(f"误判正常交易为欺诈：{fp_count}笔（可能引发用户投诉，需平衡阈值）")
print(f"漏判欺诈交易：{fn_count}笔（会导致资金损失，需优先控制）")

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# -------------------------- 第一步：加载训练好的模型与预处理工具 --------------------------
# 注意：需先运行5.8.2（数据预处理）和5.8.3（K-Means建模），确保以下变量已存在：
# preprocessor（训练好的预处理流水线）、kmeans（训练好的K-Means模型）、threshold（距离阈值）

# -------------------------- 第二步：构造新的信用卡交易数据 --------------------------
# 模拟5笔新交易（涵盖正常交易、典型欺诈交易、边缘案例）
new_transactions = pd.DataFrame({
    "amount": [899.50, 6500.00, 230.80, 9800.20, 450.00],  # 交易金额（元）
    "hour": [18, 3, 12, 2, 20],  # 交易时间（小时）：3、2为凌晨（欺诈高发时段）
    "merchant_type": [2, 4, 1, 4, 3],  # 商户类型：4为跨境消费（欺诈高发类型）
    "payment_method": [2, 4, 1, 4, 3],  # 支付方式：4为无验证（欺诈高发方式）
    "freq_24h": [2, 8, 1, 10, 2]  # 24小时交易频次：8、10为高频（欺诈可能）
})
print("新信用卡交易数据：")
print(new_transactions)

# -------------------------- 第三步：新交易数据预处理（与训练集规则一致） --------------------------
# 1. 处理缺失值（若有，此处模拟数据无缺失，直接复用训练集填充逻辑）
# 若新交易有金额缺失，用训练集的金额中位数填充（此处训练集median已在5.8.2中计算为amount_median）
# new_transactions["amount"].fillna(amount_median, inplace=True)

# 2. 执行预处理（用训练好的preprocessor，确保与训练数据规则一致）
new_transactions_processed = preprocessor.transform(new_transactions)

# -------------------------- 第四步：欺诈检测（距离计算与阈值判定） --------------------------
# 1. 计算新交易到K-Means簇中心的欧氏距离
new_distances = np.linalg.norm(new_transactions_processed - kmeans.cluster_centers_, axis=1)

# 2. 判定欺诈：距离>阈值为欺诈（1），否则为正常（0）
new_fraud_pred = (new_distances > threshold).astype(int)

# -------------------------- 第五步：整理检测结果与解读 --------------------------
# 加入距离与预测结果到新交易数据
new_result = new_transactions.copy()
new_result["distance_to_center"] = new_distances.round(4)
new_result["is_fraud_pred"] = new_fraud_pred
new_result["detection_result"] = new_result["is_fraud_pred"].map({0: "正常交易", 1: "疑似欺诈"})

# 输出检测结果
print(f"\n信用卡新交易欺诈检测结果（阈值：{threshold:.4f}）：")
print(new_result[["amount", "hour", "merchant_type", "payment_method", "freq_24h", "distance_to_center", "detection_result"]])

# 结合特征异常点解读结果（基于K-Means距离大的核心原因）
print("\n欺诈检测结果解读：")
for i in range(len(new_result)):
    trans = new_result.iloc[i]
    print(f"\n交易{i+1}（{trans['detection_result']}）：")
    # 分析异常特征（距离大的主要贡献因素）
    anomalies = []
    if trans["amount"] > 5000:
        anomalies.append(f"交易金额{trans['amount']:.2f}元（远超正常交易均值）")
    if trans["hour"] < 7:
        anomalies.append(f"交易时间{trans['hour']}点（凌晨欺诈高发时段）")
    if trans["merchant_type"] == 4:
        anomalies.append(f"跨境消费（欺诈高发商户类型）")
    if trans["payment_method"] == 4:
        anomalies.append(f"无验证支付（盗刷常用方式）")
    if trans["freq_24h"] > 5:
        anomalies.append(f"24小时交易{trans['freq_24h']}次（高频交易疑似盗刷）")
    
    if anomalies:
        print(f"  异常特征：{', '.join(anomalies)}")
        if trans["is_fraud_pred"] == 1:
            print(f"  风控建议：触发二次验证（如短信验证码），暂停交易并通知持卡人")
        else:
            print(f"  风控建议：正常放行，持续监控后续交易频次")
    else:
        print(f"  特征分析：交易金额、时间、商户类型等均符合正常消费模式")
        print(f"  风控建议：直接放行，无需额外验证")