#1.读取与查看：使用Pandas的read csv读取文件，info()/describe()查看数据信息，代码如下：
import  pandas  as  pd
sales_data = {
"订单ID" :  ["ORD001" , "ORD002" , "ORD003" , "ORD004" , "ORD005"],
"客户ID" :  ["C001" , "C002" , "C001" , "C003" , "C002"],
"产品类别":  ["手机" , "手机" , "耳机" , "平板" , "手表"],
"销售金额" :  [5999,  8999,  799,  3299,  1599],
"销售日期" :  ["2023-10-01" , "2023-10-03" , "2023-10-05" , "2023-10-07" ,"2023-10-09"]
}
df = pd.DataFrame(sales_data)
#  查看数据类型、缺失值情况
df.info()
#  查看数值型特征统计信息（辅助分析）
df.describe()

# 2. 缺失值+异常值处理：中位数填充缺失值（抗极端值），IQR法识别并修正异常值，核心代码思路：
import pandas as pd
#  1.中位数填充销售金额缺失值
df["销售金额"] = df["销售金额"].fillna(df["销售金额"].median())
#  2. IQR 法识别异常值并修正为边界值
Q1 = df["销售金额"].quantile(0.25)
Q3 = df["销售金额"].quantile(0.75)
IQR = Q3  -  Q1
#  计算上下边界
lower_bound = Q1  -  1.5  *  IQR
upper_bound = Q3  +  1.5  *  IQR
#  修正异常值
df.loc[df["销售金额"]  <  lower_bound, "销售金额"] = lower_bound
df.loc[df["销售金额"]  >  upper_bound, "销售金额"] = upper_bound

# 3.按产品类别分组聚合：使groupby+agg实现多指标聚合，reset index重置索引，代码如下：
df_grouped = df.groupby("产品类别").agg(
            年度销售总额=("销售金额" , "sum"),
            平均订单金额 =("销售金额" , "mean"),
            订单数量=("订单ID" , "count")
                ).reset_index()
#  查看聚合结果
print(df_grouped)
# 4. Matplotlib绘制柱状图：设置中文显示，调用bar()绘制，添加标题/标签，代码如下：
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文显示，避免乱码
plt.rcParams["font.family"] = ["SimHei"]  #黑体，适配Windows
# plt.rcParams["font.family"] = ["PingFangSC"]  #  苹方，适配Mac

# 提取绘图数据
x = df_grouped["产品类别"]
y = df_grouped["年度销售总额"]

# 绘制柱状图
plt.bar(x=x, height=y, width=0.6)
# 设置图表标题和坐标轴标签
plt.title("各产品类别年度销售总额")
plt.xlabel("产品类别")
plt.ylabel("销售总额（元）")
# 显示图表
plt.show()

#5.编写客户消费等级统计函数：定义函数接收数据和客户ID ，统计总额后判断等级，代码如下：
def  customer_level(df,  customer_id):
    # 筛选指定客户的所有数据
    customer_data = df[df["客户ID"] == customer_id]
    # 计算年度消费总额
    total_spend = customer_data["销售金额"].sum()
    # 判断并返回消费等级
    if  total_spend  >=  10000:
        return "VIP  客户"
    elif  total_spend  >=  5000:
        return "黄金客户"
    else:
        return "普通客户"

# 函数调用示例
# print(customer_level(df , "C001"))