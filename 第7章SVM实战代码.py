import pandas as pd
import numpy as np
# 导入数据集划分与归一化工具（财务数据适合用归一化）
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
# 加载本地财务数据：CSV文件需按"特征列+ROE列"整理，行代表企业样本
# 注意：需将路径替换为用户本地数据文件路径，确保列名与后续代码一致
finance_data = pd.read_csv('./data/enterprise_finance.csv')

# 筛选特征变量：8个核心财务指标，分别对应增长能力（revenue_growth）、盈利能力（gross_margin）、
# 风险水平（debt_ratio）、偿债能力（cash_flow_ratio）、运营效率（asset_turnover）、
# 创新能力（rd_ratio）、人力效率（staff_efficiency）、外部环境（industry_competition）
X = finance_data[['revenue_growth', 'gross_margin', 'debt_ratio', 'cash_flow_ratio',
                  'asset_turnover', 'rd_ratio', 'staff_efficiency', 'industry_competition']]
# 筛选目标变量：ROE（净资产收益率，单位：%），衡量企业自有资本盈利效率
y = finance_data['ROE']

# 导入三种回归模型与评估指标（MAE对财务数据异常值更鲁棒）
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 数据预处理
# 缺失值处理：用各特征的样本均值填充缺失值（财务数据缺失多为偶然，均值能保持分布）
X = X.fillna(X.mean())
# 特征归一化：MinMaxScaler将特征映射到[0,1]区间，适合偏态分布的财务数据
scaler = MinMaxScaler()
# 拟合并转换特征：仅用训练数据拟合scaler，避免测试集泄露
X_scaled = scaler.fit_transform(X)

# 划分训练集与测试集：test_size=0.25（25%为测试集），random_state=42确保可重复
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# 2. 支持向量机回归模型训练与评估
# 核函数选择RBF（径向基函数），适合处理财务数据的非线性关系
# C=100和gamma=0.1为财务数据经验值，平衡模型复杂度与泛化能力
svr = SVR(kernel='rbf', C=100, gamma=0.1)
svr.fit(X_train, y_train)  # 训练模型
y_svr_pred = svr.predict(X_test)  # 生成预测值

# 评估指标：MAE（平均绝对误差，值越小精度越高）、R²（解释能力）
print(f"支持向量机回归 - MAE：{mean_absolute_error(y_test, y_svr_pred):.4f}，R²：{r2_score(y_test, y_svr_pred):.4f}")

# 不同核函数对比（可选）
svr_linear = SVR(kernel='linear', C=100)
svr_linear.fit(X_train, y_train)
y_linear_pred = svr_linear.predict(X_test)
print(f"线性核SVM - MAE：{mean_absolute_error(y_test, y_linear_pred):.4f}，R²：{r2_score(y_test, y_linear_pred):.4f}")

svr_poly = SVR(kernel='poly', degree=2, C=100, gamma=0.1)
svr_poly.fit(X_train, y_train)
y_poly_pred = svr_poly.predict(X_test)
print(f"二次多项式核SVM - MAE：{mean_absolute_error(y_test, y_poly_pred):.4f}，R²：{r2_score(y_test, y_poly_pred):.4f}")

# 新企业财务特征：顺序与特征变量一致，数值为模拟的合理财务指标（如营收增长率12%、毛利率35%）
new_enterprise = np.array([[12, 35, 45, 20, 1.2, 5, 80, 6]])
# 新数据归一化：复用训练集scaler，确保预处理规则一致
new_enterprise_scaled = scaler.transform(new_enterprise)

# 用svr模型预测新企业ROE：predict返回数组，取第一个元素为预测值
pred_roe = svr_poly.predict(new_enterprise_scaled)[0]
# 输出预测结果，保留2位小数（符合财务指标精度习惯）
print(f"新企业预测ROE：{pred_roe:.2f}%")

# 盈利质量解读：基于制造业ROE行业标准，划分三个等级
if pred_roe >= 15:
    print("解读：企业盈利能力优秀，符合优质企业标准，投资风险低")
elif pred_roe >= 8:
    print("解读：企业盈利能力良好，处于行业中等偏上水平，需关注后续增长潜力")
else:
    print("解读：企业盈利能力较弱，可能存在成本控制或运营效率问题，需警惕财务风险")