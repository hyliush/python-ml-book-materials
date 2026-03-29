#### 企业欺诈
# 基础数据处理库
import numpy as np
import pandas as pd
import os
import re
import warnings
warnings.filterwarnings('ignore')

# 文本处理（基础NLP相关）
import string
from collections import Counter

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 建模与评估库
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  # 处理数据不平衡

# 加载数据集（路径根据实际环境调整）
df = pd.read_csv('./data/Final_Dataset.csv')
# 去重列（避免重复字段干扰建模）
df = df.loc[:, ~df.columns.duplicated()]

# 查看数据基本结构
print("数据集基本信息：")
print(f"数据集形状（样本数×字段数）：{df.shape}")
print("\n前5条数据预览：")
print(df[['Fillings', 'Fraud']].head())
print("\n舞弊标签分布：")
print(df['Fraud'].value_counts(normalize=True).map('{:.2%}'.format))

# 处理缺失值：文本字段填充空字符串，量化字段填充中位数
df['Fillings'] = df['Fillings'].fillna('')
# 筛选量化特征（假设以数值类型字段为量化财务指标）
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 标签编码：将yes/no转换为1/0（适配XGBoost输入要求）
df['Fraud'] = df['Fraud'].map({'yes': 1, 'no': 0})

# 查看预处理后的数据质量
print("\n预处理后数据质量检查：")
print(f"缺失值统计：{df.isnull().sum().sum()} 个")
print(f"舞弊标签编码后分布：{df['Fraud'].value_counts(normalize=True).map('{:.2%}'.format)}")


# 文本预处理函数（基础NLP清洁步骤）
def clean_text(text):
    # 转为小写
    text = text.lower()
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 应用文本预处理
df['cleaned_fillings'] = df['Fillings'].apply(clean_text)

# 定义财务舞弊风险术语词典（基于业务经验与文献整理）
fraud_keywords = [
    'litigation', 'restatement', 'adverse', 'investigation', 'correction',
    'irregularity', 'misstatement', 'non-compliance', 'fraud', 'bribery',
    'embezzlement', 'falsification', 'manipulation'
]

# 提取文本特征：统计每个风险术语的出现频次
for keyword in fraud_keywords:
    df[f'keyword_{keyword}'] = df['cleaned_fillings'].str.count(keyword)

# 构建聚合文本特征：所有风险术语的总频次（风险密度）
df['risk_density'] = df[[f'keyword_{kw}' for kw in fraud_keywords]].sum(axis=1)

# 查看文本特征提取结果
text_features = [f'keyword_{kw}' for kw in fraud_keywords] + ['risk_density']
print("文本特征提取结果（前5条）：")
print(df[text_features].head())

# 筛选量化特征（排除标签字段与文本字段）
exclude_cols = ['Fillings', 'cleaned_fillings', 'Fraud']
quant_features = [col for col in df.columns if col not in exclude_cols + text_features]

# 融合文本特征与量化特征，构建最终特征集
X = df[text_features + quant_features]
y = df['Fraud']

print(f"最终特征集形状（样本数×特征数）：{X.shape}")
print(f"特征类型分布：文本特征{len(text_features)}个，量化特征{len(quant_features)}个")

# 分层划分训练集与验证集（保持舞弊标签比例一致）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# SMOTE过采样平衡训练集（针对实际场景中舞弊样本偏少的情况）
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("数据集划分结果：")
print(f"训练集（平衡前）：{X_train.shape}，舞弊样本占比：{y_train.mean():.2%}")
print(f"训练集（平衡后）：{X_train_balanced.shape}，舞弊样本占比：{y_train_balanced.mean():.2%}")
print(f"验证集：{X_test.shape}，舞弊样本占比：{y_test.mean():.2%}")


# 定义XGBoost超参数搜索范围
param_dist = {
    'n_estimators': [100, 200, 300],  # 决策树数量
    'max_depth': [3, 4, 5, 6],  # 树的最大深度（控制过拟合）
    'learning_rate': [0.01, 0.05, 0.1],  # 学习率
    'subsample': [0.7, 0.8, 0.9, 1.0],  # 样本采样比例
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # 特征采样比例
    'gamma': [0, 0.1, 0.2, 0.3],  # 叶节点分裂阈值
    'min_child_weight': [1, 2, 3]  # 子节点最小样本权重
}

# 初始化XGBoost分类器
xgb = XGBClassifier(
    objective='binary:logistic',  # 二分类任务
    eval_metric='auc',  # 评估指标（AUC-ROC）
    random_state=42,
    use_label_encoder=False  # 禁用标签编码（避免警告）
)

# 随机搜索优化超参数（5折交叉验证）
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=50,  # 搜索次数
    scoring='f1',  # 优化目标（F1分数，适配不平衡数据）
    cv=5,  # 5折交叉验证
    random_state=42,
    verbose=1
)

# 训练模型（使用平衡后的训练集）
random_search.fit(X_train_balanced, y_train_balanced)

# 输出最优参数与最优分数
print("XGBoost最优超参数：")
print(random_search.best_params_)
print(f"\n交叉验证最优F1分数：{random_search.best_score_:.4f}")

# 获取最优模型
best_xgb = random_search.best_estimator_

# 模型预测（类别+概率）
y_pred = best_xgb.predict(X_test)
y_pred_prob = best_xgb.predict_proba(X_test)[:, 1]  # 舞弊概率（类别1的概率）

# 计算多维度评估指标
metrics = {
    '准确率': round(accuracy_score(y_test, y_pred) * 100, 2),
    '精确率': round(precision_score(y_test, y_pred) * 100, 2),
    '召回率': round(recall_score(y_test, y_pred) * 100, 2),
    'F1分数': round(f1_score(y_test, y_pred) * 100, 2),
    'AUC-ROC': round(roc_auc_score(y_test, y_pred_prob) * 100, 2)
}

# 输出评估结果
print("\n=== 模型性能评估结果 ===")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}：{metric_value}%")

# 混淆矩阵可视化
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['正常企业', '舞弊企业'])
disp.plot(cmap='Blues', values_format='d')
plt.title('XGBoost模型混淆矩阵（验证集）')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# 特征重要性可视化（Top10）
feature_importance = pd.DataFrame({
    '特征名称': X.columns,
    '重要性': best_xgb.feature_importances_
}).sort_values('重要性', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征名称', data=feature_importance)
plt.title('XGBoost模型Top10重要特征')
plt.xlabel('特征重要性')
plt.ylabel('特征名称')
plt.tight_layout()
plt.show()

#### 企业破产
# 导入核心库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import ppscore as pps

# 固定随机种子保证结果可复现
np.random.seed(42)

# 加载数据集（路径替换为实际业务路径）
company_df = pd.read_csv("./data/taiwanese_bankruptcy_data.csv")
# 清洗列名（去除首尾空格，避免后续特征调用异常）
company_df = company_df.rename(columns=lambda x: x.strip())
# 重命名标签列，简化后续模型训练中的变量调用
company_df.rename(columns={'Bankrupt?': 'Bankrupt'}, inplace=True)

# 基础数据校验（快速确认数据核心属性）
print("数据维度（行×列）：", company_df.shape)
print("破产企业占比：", round(company_df['Bankrupt'].value_counts(normalize=True)[1]*100, 2), "%")

# 划分训练集与测试集（分层抽样，避免数据泄漏）
X = company_df.drop(['Bankrupt'], axis=1)
y = company_df['Bankrupt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 计算PPS分数并筛选有效特征（仅保留有预测能力的特征）
pps_input = X_train.copy()
pps_input['Bankrupt'] = y_train.astype(str)
pps_df = pps.predictors(pps_input, y="Bankrupt")
pps_features = pps_df[pps_df.ppscore > 0]['x'].tolist()


# 定义决策树超参数搜索范围
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}
# 随机搜索优化决策树超参数（以F1分数为优化目标）
dt = DecisionTreeClassifier(random_state=42)
random_search = RandomizedSearchCV(dt, param_dist, scoring='f1', random_state=42)
random_search.fit(X_train, y_train)

# 提取Top10重要特征
best_dt = random_search.best_estimator_
feat_imp = pd.DataFrame({
    'feature': best_dt.feature_names_in_,
    'importance': best_dt.feature_importances_
}).sort_values("importance", ascending=False)[:10]
dt_features = feat_imp['feature'].tolist()

# 合并特征列表并去重，得到最终核心特征集
final_features = list(set(pps_features + dt_features))
print(f"筛选后特征数量：{len(final_features)}（原特征数：95）")


# 基于筛选后的核心特征重构训练/测试集
X_train_filter = X_train[final_features]
X_test_filter = X_test[final_features]

# SMOTE过采样平衡训练集标签
smote = SMOTE(random_state=42)
X_train_os, y_train_os = smote.fit_resample(X_train_filter, y_train)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_os, y_train_os)

# 模型预测与多维度评估
y_pred = rf.predict(X_test_filter)
accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 输出评估结果
print(f"准确率：{round(accuracy, 4)}")
print(f"平衡准确率：{round(balanced_acc, 4)}")
print(f"F1分数：{round(f1, 4)}")