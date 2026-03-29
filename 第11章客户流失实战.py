import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 加载数据
churn_data = pd.read_csv('./data/customer_churn_data.csv')

# 1. 分离特征与目标变量
X = churn_data.drop('churn', axis=1)  # 特征变量
y = churn_data['churn']  # 目标变量（1=流失，0=未流失）

# 2. 处理缺失值
# 数值型特征：用中位数填充（抗极端值影响）
X['monthly_fee'].fillna(X['monthly_fee'].median(), inplace=True)
X['cs_calls'].fillna(X['cs_calls'].median(), inplace=True)
print("缺失值处理完成：月消费和客服联系次数用中位数填充")

# 3. 特征类型划分
numeric_features = ['age', 'tenure', 'monthly_fee', 'call_minutes', 'data_usage', 'cs_calls']
categorical_features = ['gender', 'customer_level', 'plan_type', 'complaints']
print(f"\n数值型特征：{numeric_features}")
print(f"分类型特征：{categorical_features}")

# 4. 特征转换
# 数值型：标准化（均值=0，标准差=1，消除量纲影响）
# 分类型：独热编码（将类别转换为二进制向量，避免数值大小误导模型）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# 5. 划分训练集与测试集（7:3比例）
# stratify=y：确保训练集与测试集的流失率分布一致，避免样本偏斜
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. 执行预处理
X_train_processed = preprocessor.fit_transform(X_train)  # 训练集：拟合+转换
X_test_processed = preprocessor.transform(X_test)        # 测试集：仅转换（避免数据泄露）

# 7. 查看处理后的数据信息
# 获取转换后的特征名称（用于后续分析）
cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
feature_names = numeric_features + cat_feature_names

print(f"\n预处理完成：")
print(f"训练集样本数：{X_train_processed.shape[0]}，特征数：{X_train_processed.shape[1]}")
print(f"测试集样本数：{X_test_processed.shape[0]}，特征数：{X_test_processed.shape[1]}")
print(f"训练集流失率：{y_train.mean():.2%}，测试集流失率：{y_test.mean():.2%}（分布一致）")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 假设已完成预处理，加载处理后的数据集和特征名称
# （实际使用时需先运行churn_data_preprocessing.py）
# X_train_processed, X_test_processed, y_train, y_test, feature_names = ...

# 定义模型评估函数
def evaluate_model(y_true, y_pred, y_prob, model_name):
    """计算并可视化模型评估指标"""
    # 计算核心指标
    metrics = {
        "准确率": accuracy_score(y_true, y_pred),
        "精确率": precision_score(y_true, y_pred),  # 预测为流失的客户中实际流失的比例
        "召回率": recall_score(y_true, y_pred),    # 实际流失客户中被正确预测的比例
        "F1值": f1_score(y_true, y_pred),
        "AUC值": roc_auc_score(y_true, y_prob)     # 模型区分能力
    }
    
    # 打印指标
    print(f"\n----- {model_name} 评估结果 -----")
    for name, value in metrics.items():
        print(f"{name}：{value:.4f}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['未流失', '流失'],
                yticklabels=['未流失', '流失'])
    plt.xlabel('预测结果')
    plt.ylabel('实际结果')
    plt.title(f'{model_name} 混淆矩阵')
    plt.show()
    
    return metrics

# 1. 逻辑回归模型（基准模型）
lr_model = LogisticRegression(
    class_weight='balanced',  # 处理样本不平衡（自动调整类别权重）
    max_iter=1000, 
    random_state=42
)
lr_model.fit(X_train_processed, y_train)

# 预测
lr_y_pred = lr_model.predict(X_test_processed)
lr_y_prob = lr_model.predict_proba(X_test_processed)[:, 1]  # 流失概率

# 评估
lr_metrics = evaluate_model(y_test, lr_y_pred, lr_y_prob, "逻辑回归")

# 2. 随机森林模型
rf_model = RandomForestClassifier(
    n_estimators=100,        # 100棵决策树
    max_depth=8,             # 限制树深避免过拟合
    class_weight='balanced', # 平衡类别权重
    random_state=42
)
rf_model.fit(X_train_processed, y_train)

# 预测与评估
rf_y_pred = rf_model.predict(X_test_processed)
rf_y_prob = rf_model.predict_proba(X_test_processed)[:, 1]
rf_metrics = evaluate_model(y_test, rf_y_pred, rf_y_prob, "随机森林")

# 3. 梯度提升树模型
gbt_model = GradientBoostingClassifier(
    n_estimators=100,    # 100个弱学习器
    learning_rate=0.1,   # 学习率（步长）
    max_depth=5,         # 树深
    random_state=42
)
gbt_model.fit(X_train_processed, y_train)

# 预测与评估
gbt_y_pred = gbt_model.predict(X_test_processed)
gbt_y_prob = gbt_model.predict_proba(X_test_processed)[:, 1]
gbt_metrics = evaluate_model(y_test, gbt_y_pred, gbt_y_prob, "梯度提升树")

# 4. 模型对比
metrics_df = pd.DataFrame({
    "逻辑回归": lr_metrics,
    "随机森林": rf_metrics,
    "梯度提升树": gbt_metrics
}).round(4)
print("\n模型性能对比：")
print(metrics_df)

# 选择最优模型（以AUC值最高为标准）
best_model = gbt_model if gbt_metrics["AUC值"] == max(lr_metrics["AUC值"], rf_metrics["AUC值"], gbt_metrics["AUC值"]) else \
             rf_model if rf_metrics["AUC值"] == max(...) else lr_model
print(f"\n最优模型：{'梯度提升树' if best_model == gbt_model else '随机森林' if best_model == rf_model else '逻辑回归'}")

# 5. 特征重要性分析（针对树模型）
if hasattr(best_model, 'feature_importances_'):
    importance = pd.DataFrame({
        '特征': feature_names,
        '重要性': best_model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='重要性', y='特征', data=importance.head(10))
    plt.title('对客户流失影响最大的10个特征')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 5 流失影响因素：")
    print(importance[['特征', '重要性']].head(5).round(4))

import pandas as pd
import numpy as np

# 假设已加载预处理工具、最优模型和训练数据中的填充值
# （实际使用时需先运行前序代码）
# preprocessor = ...  # 预处理管道
# best_model = ...    # 最优模型（如梯度提升树）
# X = ...             # 原始训练数据特征（用于获取填充值）

def predict_churn(new_customers, preprocessor, model, threshold=0.5):
    """
    预测新客户的流失风险
    
    参数:
    - new_customers: 新客户数据DataFrame
    - preprocessor: 训练好的预处理管道
    - model: 训练好的预测模型
    - threshold: 流失概率阈值（超过此值判定为高风险）
    
    返回:
    - 包含预测结果的DataFrame
    """
    # 复制数据避免修改原始输入
    new_data = new_customers.copy()
    
    # 1. 处理缺失值（复用训练集的填充策略）
    if 'monthly_fee' in new_data.columns:
        new_data['monthly_fee'].fillna(X['monthly_fee'].median(), inplace=True)
    if 'cs_calls' in new_data.columns:
        new_data['cs_calls'].fillna(X['cs_calls'].median(), inplace=True)
    
    # 2. 特征预处理（与训练数据保持一致）
    new_processed = preprocessor.transform(new_data)
    
    # 3. 预测流失概率和风险等级
    churn_prob = model.predict_proba(new_processed)[:, 1]  # 流失概率
    churn_pred = (churn_prob >= threshold).astype(int)     # 二分类预测
    
    # 4. 生成风险等级标签
    new_data['流失概率'] = churn_prob.round(4)
    new_data['流失风险'] = pd.cut(
        churn_prob, 
        bins=[0, 0.3, 0.6, 1.0],
        labels=['低风险', '中风险', '高风险']
    )
    new_data['预测结果'] = pd.Series(churn_pred).map({0: '未流失', 1: '可能流失'})
    
    return new_data

# 生成模拟新客户数据
def generate_new_customers(n=10):
    """生成10个新客户数据用于流失预测"""
    np.random.seed(43)  # 新随机种子确保数据与训练集不同
    
    return pd.DataFrame({
        'age': np.random.randint(18, 71, size=n),
        'gender': np.random.choice(["男", "女"], size=n, p=[0.45, 0.55]),
        'customer_level': np.random.choice(['普通', '银', '金', '钻石'], size=n, p=[0.4, 0.3, 0.2, 0.1]),
        'tenure': np.random.randint(1, 61, size=n),
        'plan_type': np.random.choice(['基础', '进阶', '尊享', '无限'], size=n, p=[0.2, 0.4, 0.15, 0.25]),
        'monthly_fee': np.clip(np.random.normal(100, 40, size=n), 20, 300).round(2),
        'call_minutes': np.clip(np.random.normal(300, 100, size=n), 0, 1000).round(1),
        'data_usage': np.clip(np.random.normal(15, 8, size=n), 0, 50).round(2),
        'cs_calls': np.clip(np.random.poisson(0.8, size=n), 0, 5),
        'complaints': np.random.choice(['是', '否'], size=n, p=[0.2, 0.8])
    }).reindex(columns=['age', 'tenure', 'monthly_fee', 'call_minutes', 'data_usage',
       'cs_calls', 'gender', 'customer_level', 'plan_type', 'complaints'])

# 生成新客户数据并预测
new_customers = generate_new_customers(10)
pred_results = predict_churn(new_customers, preprocessor, best_model, threshold=0.4)  # 降低阈值以捕捉更多潜在流失客户

# 展示预测结果
print("新客户流失风险预测结果：")
print(pred_results[['age', 'tenure', 'customer_level', 'monthly_fee', 
                    'cs_calls', 'complaints', '流失概率', '流失风险', '预测结果']])

# 生成高风险客户挽留建议
high_risk = pred_results[pred_results['流失风险'] == '高风险']
if not high_risk.empty:
    print("\n----- 高风险客户挽留策略 -----")
    for i, customer in high_risk.iterrows():
        print(f"\n客户ID：{i+1}（流失概率：{customer['流失概率']:.2%}）")
        
        # 基于特征制定个性化建议
        suggestions = []
        if customer['tenure'] < 12:  # 在网时长较短
            suggestions.append("提供新客户专属6个月套餐折扣，延长在网周期")
        if customer['cs_calls'] >= 2 or customer['complaints'] == 1:  # 有服务不满记录
            suggestions.append("安排高级客服专员1对1回访，解决历史问题")
        if customer['monthly_fee'] > 200:  # 月消费过高
            suggestions.append("推荐性价比更高的融合套餐，降低月支出15%-20%")
        if customer['customer_level'] in ["金", "钻石"]:  # 高价值客户
            suggestions.append("赠送VIP专属权益（如机场贵宾厅、积分加倍）")
        
        print("建议措施：")
        for s in suggestions:
            print(f"- {s}")