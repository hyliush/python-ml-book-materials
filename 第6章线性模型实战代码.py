import pandas as pd
import numpy as np
# 导入数据集加载工具（加州房价数据集，sklearn内置）
from sklearn.datasets import fetch_california_housing
# 导入后续需用的数据集划分与标准化工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载加州房价数据集，返回Bunch对象（含data、feature_names等属性）
california = fetch_california_housing()
# 特征变量：转换为DataFrame，便于后续数据处理（列名为特征名）
X = pd.DataFrame(california.data, columns=california.feature_names)
# 目标变量：房屋价值中位数，命名为MedHouseVal（单位：万美元）
y = pd.Series(california.target, name='MedHouseVal')

# 基础数据探索1：检查缺失值（加州数据集无缺失值，无需填充）
# sum().sum()：先按列求和（每列缺失值数），再汇总所有列缺失值总数
print("缺失值统计（总数量）：", X.isnull().sum().sum())
# 基础数据探索2：查看特征统计量（均值、最值、分位数等），保留2位小数便于阅读
print("特征数值分布统计：\n", X.describe().round(2))

# 定义IQR法异常值剔除函数：输入数据框与目标列，返回剔除异常值后的数据
def remove_outliers(df, column):
    # 计算25%分位数（Q1）与75%分位数（Q3）
    Q1, Q3 = df[column].quantile([0.25, 0.75])
    # 计算四分位距（IQR = Q3 - Q1）
    IQR = Q3 - Q1
    # 异常值判定边界：[Q1-1.5*IQR, Q3+1.5*IQR]，筛选范围内数据
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

# 处理核心变量异常值：选择MedInc（收入，影响购房能力）、AveRooms（房间数，反映房屋结构）、
# MedHouseVal（目标变量，避免极端房价干扰训练），同步筛选X与y的索引防止数据错位
for col in ['MedInc', 'AveRooms']:
    X = remove_outliers(X, col)
    y = y[X.index]  # 确保目标变量与特征变量的样本一一对应

# 特征标准化：StandardScaler将特征转换为均值=0、标准差=1的分布，消除量纲影响
scaler = StandardScaler()
# 仅用训练数据拟合scaler（避免测试集数据泄露到训练过程），转换特征
X_scaled = scaler.fit_transform(X)

# 划分训练集与测试集：test_size=0.3（30%为测试集），random_state=42确保不同环境下划分结果一致
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 导入三种回归模型、评估指标与网格搜索工具（用于优化正则化参数α）
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# 1. 线性回归训练与评估
lr = LinearRegression()  # 初始化线性回归模型
lr.fit(X_train, y_train)  # 用训练集拟合模型，学习特征与房价的映射关系
y_lr_pred = lr.predict(X_test)  # 用测试集验证模型，生成预测值
# 计算评估指标：MSE（均方误差，值越小精度越高）、R²（决定系数，越近1解释能力越强）
print(f"线性回归 - MSE：{mean_squared_error(y_test, y_lr_pred):.4f}，R²：{r2_score(y_test, y_lr_pred):.4f}")

# 2. 岭回归训练与评估（网格搜索优化α）
# 网格搜索参数：α候选值覆盖0.1-100，包含弱、中、强三种正则化强度
ridge_param = {'alpha': [0.1, 1, 10, 100]}
# 初始化网格搜索：cv=5（5折交叉验证，提升参数选择可靠性），模型为Ridge
ridge_grid = GridSearchCV(Ridge(random_state=42), ridge_param, cv=5)
ridge_grid.fit(X_train, y_train)  # 用训练集训练并筛选最优α
y_ridge_pred = ridge_grid.best_estimator_.predict(X_test)  # 用最优模型预测
# 输出最优α与评估指标，α反映正则化强度
print(f"岭回归（最优α={ridge_grid.best_params_['alpha']}）- MSE：{mean_squared_error(y_test, y_ridge_pred):.4f}，R²：{r2_score(y_test, y_ridge_pred):.4f}")

# 3. Lasso回归训练与评估（网格搜索优化α）
# 网格搜索参数：Lasso对α更敏感，候选值偏小（0.01-10），避免过度剔除特征
lasso_param = {'alpha': [0.01, 0.1, 1, 10]}
# 初始化网格搜索：max_iter=10000（增加迭代次数，确保Lasso模型收敛）
lasso_grid = GridSearchCV(Lasso(random_state=42, max_iter=10000), lasso_param, cv=5)
lasso_grid.fit(X_train, y_train)  # 训练并筛选最优α
y_lasso_pred = lasso_grid.best_estimator_.predict(X_test)  # 最优模型预测
print(f"Lasso回归（最优α={lasso_grid.best_params_['alpha']}）- MSE：{mean_squared_error(y_test, y_lasso_pred):.4f}，R²：{r2_score(y_test, y_lasso_pred):.4f}")

# Lasso特征选择：输出系数非零的特征（系数为0的特征对预测无贡献，可剔除）
lasso_coef = pd.DataFrame({
    'Feature': california.feature_names,  # 特征名称
    'Coeff': lasso_grid.best_estimator_.coef_  # Lasso模型系数
})
print("\nLasso筛选的核心特征（系数非零）：\n", lasso_coef[lasso_coef['Coeff'] != 0])

# 新房屋特征：顺序需与加州数据集特征一致（MedInc→HouseAge→AveRooms→AveBedrms→
# Population→AveOccup→Latitude→Longitude），数值为模拟的合理房屋属性
new_house = np.array([[4.5, 25, 6, 2, 1500, 2.5, 37.7, -122.4]])
# 新数据标准化：仅用transform（复用训练集scaler的均值与标准差，避免数据泄露）
new_house_scaled = scaler.transform(new_house)

# 用Lasso最优模型预测房价：predict返回数组，取第一个元素为预测值
pred_price = lasso_grid.best_estimator_.predict(new_house_scaled)[0]
# 输出预测结果，保留4位小数（符合房价数值精度需求）
print(f"新房屋预测价格：{pred_price:.4f} 万美元")
