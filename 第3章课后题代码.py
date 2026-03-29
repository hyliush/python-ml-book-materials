# 1.导入工具
from  sklearn.model_selection  import  train_test_split
from  sklearn.preprocessing  import  StandardScaler,  OneHotEncoder
from  sklearn.compose  import  ColumnTransformer
from  sklearn.linear_model  import  LogisticRegression
from  sklearn.metrics  import  accuracy_score
df = pd.read_csv("./data/customer_purchase_intention.csv")
# 2. 数据准备（假设已加载数据df）
X = df[["浏览时长" , "消费金额" , "客户等级" , "商品类别"]]
y = df["购买意向"]  #  1=  有购买意向，0=  无

# 3. 划分数据集
X_train,  X_test,  y_train,  y_test = train_test_split(X,  y,  test_size=0.3, random_state=42)

# 4. 特征预处理
preprocessor = ColumnTransformer(transformers=[
            ("num",  StandardScaler(),  ["浏览时长" , "消费金额"]),
            ("cat",  OneHotEncoder(drop="first"),  ["客户等级" , "商品类别"])])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
# 5. 模型训练与预测
model = LogisticRegression()
model.fit(X_train_processed,  y_train)
y_pred = model.predict(X_test_processed)

# 6. 模型评估
print(f"模型准确率：{accuracy_score(y_test ,  y_pred):.4f}")