import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 加载数据
customer_data = pd.read_csv('./data/customer_segmentation_data.csv')

# 1. 处理缺失值
# 仅平均客单价有少量缺失，用中位数填充（抗极端值影响）
avg_order_median = customer_data['avg_order_value'].median()
customer_data['avg_order_value'].fillna(avg_order_median, inplace=True)
print(f"缺失值处理完成：平均客单价用中位数{avg_order_median:.2f}填充")

# 2. 特征选择
# 聚类核心特征：选择与消费行为、互动活跃度强相关的特征，排除弱区分度特征
# 保留：消费频次、客单价、最近购买时间、APP登录次数、邮件打开率
# 排除：年龄、性别（对电商客户分群区分度较低）
selected_features = [
    'purchase_freq',    # 消费频次
    'avg_order_value',  # 平均客单价
    'recency',          # 最近购买时间
    'app_login_count',  # APP登录次数
    'email_open_rate'   # 邮件打开率
]
X = customer_data[selected_features].copy()

# 3. 特征转换：标准化（K-Means对尺度敏感，需将特征转换为均值0、标准差1）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 转换为DataFrame便于分析
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)

# 输出预处理后的数据信息
print(f"\n选定聚类特征：{selected_features}")
print(f"预处理后数据形状：{X_scaled_df.shape}")
print("\n标准化后特征均值与标准差：")
print(pd.DataFrame({
    '均值': X_scaled_df.mean().round(4),
    '标准差': X_scaled_df.std().round(4)
}))

# 保存标准化器（用于后续新数据处理）
import joblib
joblib.dump(scaler, 'segmentation_scaler.pkl')
print("\n标准化器已保存为'segmentation_scaler.pkl'，供新数据使用")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import joblib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载预处理后的数据（假设已运行 preprocessing 代码）
# X_scaled ：标准化后的特征矩阵
# X ：原始特征（未标准化，用于结果解读）
# selected_features ：选定的聚类特征
# 实际使用时需替换为真实数据加载逻辑
# X_scaled = ... 
# X = ...
# selected_features = ...

# 1. 确定最佳聚类数量 K （肘部法）
# 测试 K=2 到 10 ，计算惯性值（ Inertia ：簇内平方和）
inertia = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('聚类数量K')
plt.ylabel('惯性值（Inertia）')
plt.title('肘部法确定最佳K值')
plt.grid(alpha=0.3)
plt.show()

# 2. 基于肘部图选择 K=4 （惯性值下降趋缓点），训练最终模型
best_k = 4
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 3. 聚类质量评估（无监督指标）
silhouette = silhouette_score(X_scaled, clusters)  # 轮廓系数：越接近 1 越好
calinski = calinski_harabasz_score(X_scaled, clusters)  # 方差比：值越大越好
print(f"\n 聚类评估指标：")
print(f"轮廓系数（ Silhouette Score ）：{silhouette:.4f} （ 0.5 以上为较合理）")
print(f"Calinski-Harabasz 指数：{calinski:.2f}（值越大表示聚类越显著）")

# 4. 分析各群体特征（使用原始特征均值，便于业务解读）
# 将聚类结果加入原始数据
X_with_cluster = X.copy()
X_with_cluster['cluster'] = clusters

# 计算每个群体的特征均值
cluster_analysis = X_with_cluster.groupby('cluster').mean().round(2)
print("\n各群体特征均值分析：")
print(cluster_analysis)

# 5. 可视化群体差异（选取核心特征绘制雷达图）
plt.figure(figsize=(10, 8))
# 标准化特征均值（便于雷达图对比）
normalized_analysis = (cluster_analysis - cluster_analysis.mean()) / cluster_analysis.std()

# 绘制雷达图
categories = normalized_analysis.columns
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar=True)
for i in range(best_k):
    values = normalized_analysis.iloc[i].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'群体{i}')
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(-2, 2)  # 限制范围使差异更明显
plt.title('各用户群体特征标准化雷达图')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.tight_layout()
plt.show()

# 6. 群体命名与核心特征总结
# 基于特征分析为每个群体命名（结合业务理解）
cluster_names = {
    0: '高价值忠诚客户',
    1: '低频潜力客户',
    2: '沉睡流失风险客户',
    3: '高频低价值客户'
}

# 输出各群体核心特征
print("\n用户群体命名与核心特征：")
for cluster_id in range(best_k):
    print(f"\n群体{cluster_id}：{cluster_names[cluster_id]}")
    features = cluster_analysis.loc[cluster_id]
    
    # 提取该群体最显著的 3 个特征（与总体均值对比）
    overall_mean = X.mean()
    diff = (features - overall_mean).abs().sort_values(ascending=False)
    top_features = diff.index[:3]
    
    for feat in top_features:
        trend = "高于" if features[feat] > overall_mean[feat] else "低于"
        print(f"- {feat}：{features[feat]}（{trend}总体均值{overall_mean[feat]:.2f}）")

# 保存模型
joblib.dump(kmeans, 'customer_segmentation_model.pkl')
print("\nK-Means聚类模型已保存为'customer_segmentation_model.pkl'")

import pandas as pd
import numpy as np
import joblib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载模型和预处理工具（假设已训练并保存）
kmeans = joblib.load('customer_segmentation_model.pkl')
scaler = joblib.load('segmentation_scaler.pkl')

# 群体命名映射（与聚类分析一致）
cluster_names = {
    0: '高价值忠诚客户',
    1: '低频潜力客户',
    2: '沉睡流失风险客户',
    3: '高频低价值客户'
}

# 1. 制定分群营销策略
def generate_strategies():
    """基于各群体特征生成差异化营销策略"""
    strategies = {
        '高价值忠诚客户': {
            '核心特征': '消费频次高、客单价高、近期有购买、APP活跃度高',
            '营销目标': '提升忠诚度，预防流失，挖掘交叉销售机会',
            '策略建议': [
                '推出VIP专属权益（如会员日折扣、专属客服）',
                '基于购买历史推荐高端互补产品（如买过手机推荐配件）',
                '邀请参与新品测试，增强品牌认同感',
                '减少高频促销（避免价格敏感化），侧重品质沟通'
            ]
        },
        '低频潜力客户': {
            '核心特征': '客单价中等、近期有购买、邮件打开率高但消费频次低',
            '营销目标': '提高购买频率，培养消费习惯',
            '策略建议': [
                '发送个性化品类推荐（基于历史购买）',
                '推出"满2件减X元"等鼓励多买的活动',
                '设置会员积分阶梯（购买次数越多，积分倍数越高）',
                '通过邮件推送限时优惠券（利用其高打开率）'
            ]
        },
        '沉睡流失风险客户': {
            '核心特征': '近期无购买（>60天）、登录次数少、消费频次低',
            '营销目标': '唤醒活跃度，促进复购',
            '策略建议': [
                '发送"回归礼包"（如无门槛优惠券）',
                '推送近期新品或热门商品，激发兴趣',
                '简化复购流程（如"一键回购"功能）',
                '通过短信触达（APP登录少，邮件打开率低）'
            ]
        },
        '高频低价值客户': {
            '核心特征': '消费频次高、客单价低、APP活跃度高',
            '营销目标': '提升客单价，引导购买高毛利商品',
            '策略建议': [
                '推出"满额升级"活动（如满300元赠高价值小样）',
                '推荐性价比高的中端替代品（如基础款升级款）',
                '设置"满减券"而非"折扣券"（引导提高客单价）',
                '通过APP推送实时促销（利用其高活跃度）'
            ]
        }
    }
    
    # 转换为DataFrame展示
    strategy_df = pd.DataFrame.from_dict(
        {k: [v['核心特征'], v['营销目标'], '\n'.join(v['策略建议'])] 
         for k, v in strategies.items()},
        orient='index',
        columns=['核心特征', '营销目标', '策略建议']
    )
    return strategy_df

# 输出营销策略
strategy_df = generate_strategies()
print("各用户群体营销策略：")
print(strategy_df)

# 2. 新客户群体预测函数
def predict_customer_segment(new_customers, scaler, kmeans, cluster_names):
    """
    预测新客户所属群体
    
    参数:
    - new_customers: 新客户数据（包含selected_features列）
    - scaler: 训练好的标准化器
    - kmeans: 训练好的K-Means模型
    - cluster_names: 群体命名映射
    
    返回:
    - 包含预测结果的DataFrame
    """
    # 提取所需特征
    selected_features = ['purchase_freq', 'avg_order_value', 'recency', 
                         'app_login_count', 'email_open_rate']
    new_data = new_customers[selected_features].copy()
    
    # 处理缺失值（用训练集中位数填充）
    for col in new_data.columns:
        if new_data[col].isnull().any():
            # 实际应用中应保存训练集各特征中位数
            train_median = pd.read_csv('customer_segmentation_data.csv')[col].median()
            new_data[col].fillna(train_median, inplace=True)
    
    # 标准化
    new_scaled = scaler.transform(new_data)
    
    # 预测群体
    new_clusters = kmeans.predict(new_scaled)
    new_data['预测群体ID'] = new_clusters
    new_data['预测群体名称'] = new_data['预测群体ID'].map(cluster_names)
    
    return new_data

# 3. 生成新客户数据并预测
def generate_new_customers(n=5):
    """生成5个新客户数据用于群体预测"""
    np.random.seed(44)
    return pd.DataFrame({
        'purchase_freq': np.random.randint(1, 15, size=n),
        'avg_order_value': np.clip(np.random.normal(300, 200, size=n), 50, 1500).round(2),
        'recency': np.random.choice([10, 30, 90, 180], size=n),
        'app_login_count': np.random.randint(0, 40, size=n),
        'email_open_rate': np.round(np.random.uniform(0.1, 0.8, size=n), 2),
        # 额外信息（非聚类特征，用于展示）
        'age': np.random.randint(20, 60, size=n),
        'gender': np.random.binomial(1, 0.5, size=n)
    })

# 预测新客户群体
new_customers = generate_new_customers(5)
predicted_segments = predict_customer_segment(new_customers, scaler, kmeans, cluster_names)

# 展示预测结果
print("\n新客户群体预测结果：")
print(predicted_segments[['purchase_freq', 'avg_order_value', 'recency', 
                          '预测群体名称']])

# 4. 为新客户推荐策略
print("\n新客户个性化营销策略推荐：")
for i in range(len(predicted_segments)):
    segment_name = predicted_segments.iloc[i]['预测群体名称']
    print(f"\n客户{i+1}（{segment_name}）：")
    print("核心策略：", strategy_df.loc[segment_name]['策略建议'].split('\n')[0])