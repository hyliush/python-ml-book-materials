# 1. 导入核心工具库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 数据预处理与样本均衡
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
# SVM模型与超参数调优
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
# 模型评估指标
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")  # 忽略无关警告

# 2. 定义金融信贷数据预处理函数（适配SVM对数据的要求）
def preprocess_credit_data(file_path):
    # 加载数据：4个财务特征+目标变量default(0=不违约,1=违约)
    data = pd.read_csv(file_path)
    feature_cols = ["资产负债率", "净利润率", "流动比率", "营收增长率"]
    X = data[feature_cols]
    y = data["default"]
    
    # 步骤1：填充缺失值（财务指标用中位数，抗极端值影响）
    X.fillna(X.median(), inplace=True)
    # 步骤2：特征标准化（SVM对距离敏感，必须消除量纲差异）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 步骤3：SMOTE过采样解决样本不均衡（违约样本少）
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    print(f"采样前：违约样本{sum(y)}, 不违约样本{len(y)-sum(y)}")
    print(f"采样后：违约样本{sum(y_resampled)}, 不违约样本{len(y_resampled)-sum(y_resampled)}")
    return X_resampled, y_resampled, scaler, feature_cols

# 加载并预处理数据
X, y, scaler, features = preprocess_credit_data("./data/credit_data.csv")
# 分层划分训练集/测试集，保证违约/不违约样本分布一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. SVM模型构建+超参数网格搜索调优（风控重点关注召回率）
# 初始化SVM分类器
svm_base = SVC(random_state=42)
# 定义超参数网格：惩罚系数C + RBF核参数γ（金融场景优先RBF核）
param_grid = {
    "C": [0.1, 1, 10, 100, 1000],  # C越大，对分类误差惩罚越重
    "kernel": ["rbf"],              # 适配金融数据的非线性特征
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],  # γ越大，模型越复杂
    "class_weight": [None, "balanced"]  # 样本权重，解决潜在不均衡
}

# 网格搜索：5折交叉验证，以召回率为核心评估指标（减少违约漏判）
grid_search = GridSearchCV(
    estimator=svm_base,
    param_grid=param_grid,
    cv=5,
    scoring="recall",  # 风控场景优先保证违约样本召回率
    n_jobs=-1,         # 利用所有CPU核心加速
    verbose=1
)
# 训练并寻找最优超参数
grid_search.fit(X_train, y_train)
best_svm = grid_search.best_estimator_
print(f"\n最优超参数组合：{grid_search.best_params_}")
print(f"训练集最优召回率：{grid_search.best_score_:.4f}")

# 4. 模型评估（输出风控核心指标）
y_pred = best_svm.predict(X_test)
# 计算多维度评估指标，兼顾准确率与违约召回率
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)  # 违约样本召回率（核心）
f1 = f1_score(y_test, y_pred)
# 打印评估结果
print("\n===== 模型测试集评估结果 =====")
print(f"整体准确率：{acc:.4f}")
print(f"违约样本召回率：{rec:.4f}")  # 风控重点关注
print(f"F1-Score（平衡准确率与召回率）：{f1:.4f}")

# 5. 预测结果可视化（混淆矩阵，直观展示分类效果）
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
plt.rcParams["axes.unicode_minus"] = False    # 显示负号
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6), dpi=100)
# 绘制混淆矩阵
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("SVM信贷违约预测混淆矩阵（风控版）", fontsize=14, pad=20)
plt.colorbar(shrink=0.8)
# 设置坐标轴标签
class_labels = ["不违约", "违约"]
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, fontsize=12)
plt.yticks(tick_marks, class_labels, fontsize=12)
# 标注混淆矩阵数值
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                 color="white" if cm[i,j] > thresh else "black", fontsize=14)
# 坐标轴标签
plt.xlabel("预测标签", fontsize=12, labelpad=10)
plt.ylabel("真实标签", fontsize=12, labelpad=10)
plt.tight_layout()
plt.savefig("svm_credit_confusion_matrix.png")  # 保存图片
plt.show()

# 6. 新企业信贷违约预测（实战示例）
# 新申请企业财务数据：[资产负债率, 净利润率, 流动比率, 营收增长率]
new_company = np.array([[0.65, 0.08, 1.2, 0.15], [0.85, 0.02, 0.8, -0.05]])
# 必须使用训练好的标准化器处理新数据
new_company_scaled = scaler.transform(new_company)
# 预测违约结果
new_pred = best_svm.predict(new_company_scaled)
# 输出预测结果
print("\n===== 新企业信贷违约预测结果 =====")
for i, res in enumerate(new_pred):
    print(f"企业{i+1}：{'【高风险：违约】' if res==1 else '【低风险：不违约】'}")