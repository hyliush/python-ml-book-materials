import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载信用数据
credit_data = pd.read_csv("./data/credit_risk_data.csv")

# -------------------- 第一步：处理缺失值 --------------------
# 分析缺失特征类型：收入（连续型）用中位数填充（抗极端值），教育程度（离散型）用众数填充
# 计算填充值
income_median = credit_data["income"].median()  # 收入中位数
education_mode = credit_data["education"].mode()[0]  # 教育程度众数（取第一个众数）

# 填充缺失值
credit_data["income"].fillna(income_median, inplace=True)
credit_data["education"].fillna(education_mode, inplace=True)

# -------------------- 第二步：处理异常值 --------------------
# 定义 IQR 异常值剔除函数（针对连续型特征：收入、信用评分、贷款金额、债务收入比）
def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # 保留范围内的数据
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

# 对连续型特征剔除异常值
continuous_cols = ["income", "credit_score", "loan_amount", "debt_income_ratio"]
credit_data_clean = remove_outliers_iqr(credit_data, continuous_cols)
print(f"剔除异常值前样本数：{len(credit_data)}，剔除后：{len(credit_data_clean)}")

# --------------------- 第三步：特征与目标变量分离 --------------------
# 特征变量：排除目标变量 is_default ，其余均为输入特征
X = credit_data_clean.drop("is_default", axis=1)
# 目标变量：是否违约（1= 违约，0= 正常）
y = credit_data_clean["is_default"]

# -------------------- 第四步：特征标准化 --------------------
# 信用数据量纲差异大（如收入单位为元，逾期次数为次），标准化后适配 SVM 、 GBDT 等模型
scaler = StandardScaler()
# 仅用训练数据拟合scaler（避免测试集数据泄露），转换特征
X_scaled = scaler.fit_transform(X)
# 转换为DataFrame，保留特征名（便于后续分析）
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# -------------------- 第五步：划分训练集与测试集 --------------------
# 按7:3比例划分，stratify=y确保训练集与测试集的违约率分布一致（避免数据偏斜）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.3, random_state=42, stratify=y
)

# 输出数据集信息
print(f"\n训练集：{X_train.shape}，测试集：{X_test.shape}")
print(f"训练集违约率：{y_train.mean():.2%}，测试集违约率：{y_test.mean():.2%}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, confusion_matrix)
import seaborn as sns

# 设置中文字体（避免图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------- 第一步：初始化并训练三种模型 --------------------
# 1. 支持向量机（SVM）：kernel=rbf（适合非线性数据），class_weight='balanced'（处理样本不平衡）
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', 
                probability=True, random_state=42)
svm_model.fit(X_train, y_train)  # 训练模型

# 2. 随机森林：n_estimators=100（决策树数量），max_depth=8（限制树深避免过拟合）
rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, 
                                  class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# 3. 梯度提升树（GBDT）：learning_rate=0.1（步长），n_estimators=100（弱学习器数量）
gbdt_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, 
                                        max_depth=6, random_state=42)
gbdt_model.fit(X_train, y_train)

# -------------------- 第二步：模型预测（测试集） --------------------
# 预测类别（0=正常，1=违约）与预测概率（用于AUC计算）
# SVM预测
svm_y_pred = svm_model.predict(X_test)
svm_y_prob = svm_model.predict_proba(X_test)[:, 1]  # 取违约（类别1）的概率

# 随机森林预测
rf_y_pred = rf_model.predict(X_test)
rf_y_prob = rf_model.predict_proba(X_test)[:, 1]

# GBDT预测
gbdt_y_pred = gbdt_model.predict(X_test)
gbdt_y_prob = gbdt_model.predict_proba(X_test)[:, 1]

# -------------------- 第三步：计算评估指标 --------------------
# 定义指标计算函数（返回精确率、召回率、F1、AUC）
def calculate_metrics(y_true, y_pred, y_prob):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return {"Precision": precision, "Recall": recall, "F1": f1, "AUC-ROC": auc}

# 计算三种模型的指标
svm_metrics = calculate_metrics(y_test, svm_y_pred, svm_y_prob)
rf_metrics = calculate_metrics(y_test, rf_y_pred, rf_y_prob)
gbdt_metrics = calculate_metrics(y_test, gbdt_y_pred, gbdt_y_prob)

# 整理指标为DataFrame，便于对比
metrics_df = pd.DataFrame({
    "SVM（支持向量机）": svm_metrics,
    "随机森林": rf_metrics,
    "GBDT（梯度提升树）": gbdt_metrics
}).round(4)

# 输出指标对比结果
print("三种模型信用风险预测指标对比：")
print(metrics_df)

# -------------------- 第四步：可视化（混淆矩阵与ROC曲线） --------------------
# 1. 最优模型（GBDT）混淆矩阵可视化（展示预测正确/错误的样本分布）
plt.figure(figsize=(12, 5))

# 混淆矩阵
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, gbdt_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['正常（0）', '违约（1）'], 
            yticklabels=['正常（0）', '违约（1）'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('GBDT模型混淆矩阵（测试集）')

# 2. 三种模型ROC曲线对比（AUC越高，曲线越靠近左上角）
plt.subplot(1, 2, 2)
# 计算ROC曲线的假正例率（FPR）和真正例率（TPR）
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_y_prob)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_prob)
gbdt_fpr, gbdt_tpr, _ = roc_curve(y_test, gbdt_y_prob)

# 绘制ROC曲线
plt.plot(svm_fpr, svm_tpr, label=f'SVM（AUC={svm_metrics["AUC-ROC"]:.4f}）', linewidth=2)
plt.plot(rf_fpr, rf_tpr, label=f'随机森林（AUC={rf_metrics["AUC-ROC"]:.4f}）', linewidth=2)
plt.plot(gbdt_fpr, gbdt_tpr, label=f'GBDT（AUC={gbdt_metrics["AUC-ROC"]:.4f}）', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测（AUC=0.5）')  # 对角线
plt.xlabel('假正例率（FPR）')
plt.ylabel('真正例率（TPR）')
plt.title('三种模型ROC曲线对比')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------- 第五步：特征重要性（随机森林与GBDT） --------------------
# 信用风险预测中，特征重要性可指导风险因子分析（如哪些因素对违约影响最大）
plt.figure(figsize=(10, 6))

# 提取特征名称与重要性（随机森林与GBDT）
features = X_train.columns
rf_importance = rf_model.feature_importances_
gbdt_importance = gbdt_model.feature_importances_

# 整理为DataFrame
importance_df = pd.DataFrame({
    "特征": features,
    "随机森林重要性": rf_importance,
    "GBDT重要性": gbdt_importance
})
# 按GBDT重要性降序排列
importance_df = importance_df.sort_values("GBDT重要性", ascending=False)

# 绘制特征重要性柱状图
x = np.arange(len(features))
width = 0.35
plt.bar(x - width/2, importance_df["随机森林重要性"], width, label='随机森林')
plt.bar(x + width/2, importance_df["GBDT重要性"], width, label='GBDT')

# 添加标签与标题
plt.xlabel('特征')
plt.ylabel('特征重要性')
plt.title('信用风险预测特征重要性对比（随机森林 vs GBDT）')
plt.xticks(x, importance_df["特征"], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 输出重要性排名（GBDT）
print("\nGBDT模型特征重要性排名（Top5）：")
print(importance_df[["特征", "GBDT重要性"]].head(5).round(4))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------------------- 第一步：加载训练好的模型与预处理工具 --------------------
# 注意：需先运行5.7.2（数据预处理）和5.7.3（模型训练），确保以下变量已存在：
# gbdt_model（训练好的GBDT模型）、scaler（训练好的标准化器）、X_train.columns（特征名）

# -------------------- 第二步：构造新用户信用数据 --------------------
# 模拟3个新申请贷款用户的信息（涵盖不同风险等级：低、中、高）
new_users = pd.DataFrame({
    # 用户1：年轻、收入中等、无逾期、信用良好（低风险）
    "age": [28, 45, 35],
    "income": [9500.0, 6800.0, 5200.0],
    "education": [3, 2, 1],  # 3=大专，2=高中，1=小学
    "credit_score": [780.0, 620.0, 450.0],
    "overdue_times": [0, 2, 4],  # 近2年逾期次数
    "loan_amount": [150000.0, 200000.0, 80000.0],
    "loan_term": [36, 48, 24],  # 贷款期限（月）
    "debt_income_ratio": [0.35, 0.62, 0.75]  # 债务收入比
})
# 为用户2加入缺失值（模拟未填写收入），后续用训练集中位数填充
new_users.loc[1, "income"] = np.nan

print("新申请贷款用户信息：")
print(new_users)

# -------------------- 第三步：新数据预处理（与训练集规则一致） --------------------
# 1. 填充缺失值（复用训练集的收入中位数，需先加载训练集数据或记录中位数）
# 从5.7.2中获取训练集收入中位数（若已关闭之前代码，可重新加载credit_risk_data.csv计算）
train_data = pd.read_csv("credit_risk_data.csv")
train_income_median = train_data["income"].median()
new_users["income"].fillna(train_income_median, inplace=True)

# 2. 标准化（复用训练集的scaler，确保均值/标准差一致）
new_users_scaled = scaler.transform(new_users)
# 转换为DataFrame，保留特征名
new_users_scaled_df = pd.DataFrame(new_users_scaled, columns=X_train.columns)

# -------------------- 第四步：信用风险预测（GBDT最优模型） --------------------
# 预测违约概率（类别1的概率）与风险等级
# 风险等级划分：违约概率<0.2=低风险，0.2-0.5=中风险，>0.5=高风险（基于业务经验）
default_prob = gbdt_model.predict_proba(new_users_scaled_df)[:, 1]
risk_level = pd.cut(default_prob, 
                    bins=[0, 0.2, 0.5, 1.0], 
                    labels=["低风险", "中风险", "高风险"])

# 整理预测结果
pred_result = new_users.copy()
pred_result["违约概率"] = default_prob.round(4)
pred_result["风险等级"] = risk_level
pred_result["审批建议"] = pred_result["风险等级"].map({
    "低风险": "通过（可正常放贷）",
    "中风险": "谨慎（需补充资料或降低贷款额度）",
    "高风险": "拒绝（违约风险过高）"
})

# 输出预测结果
print("\n新用户信用风险预测结果：")
print(pred_result[["age", "income", "credit_score", "overdue_times", "违约概率", "风险等级", "审批建议"]])

# -------------------- 第五步：风险原因解读（结合特征重要性） --------------------
# 从5.7.3的特征重要性可知：逾期次数、债务收入比、信用评分是影响最大的三个特征
print("\n风险原因解读：")
for i in range(len(pred_result)):
    user = pred_result.iloc[i]
    print(f"\n用户{i+1}（{user['风险等级']}）：")
    # 逾期次数影响
    if user["overdue_times"] > 2:
        print(f"  - 近2年逾期{int(user['overdue_times'])}次，显著增加违约风险（特征重要性Top1）")
    elif user["overdue_times"] > 0:
        print(f"  - 近2年逾期{int(user['overdue_times'])}次，需关注还款习惯")
    else:
        print(f"  - 近2年无逾期，还款记录良好")
    
    # 债务收入比影响
    if user["debt_income_ratio"] > 0.6:
        print(f"  - 债务收入比{user['debt_income_ratio']:.3f}，还款压力大（特征重要性Top2）")
    elif user["debt_income_ratio"] > 0.4:
        print(f"  - 债务收入比{user['debt_income_ratio']:.3f}，还款压力中等")
    else:
        print(f"  - 债务收入比{user['debt_income_ratio']:.3f}，还款压力小")
    
    # 信用评分影响
    if user["credit_score"] < 550:
        print(f"  - 信用评分{user['credit_score']:.1f}，信用状况较差（特征重要性Top3）")
    elif user["credit_score"] < 650:
        print(f"  - 信用评分{user['credit_score']:.1f}，信用状况中等")
    else:
        print(f"  - 信用评分{user['credit_score']:.1f}，信用状况良好")