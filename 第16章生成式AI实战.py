import pandas as pd
import openai

df = pd.read_csv("./data/announcements.csv", encoding="utf-8")
print(df.head())
print(df.info())

# 删除正文缺失的记录
df = df.dropna(subset=["content"]).copy()

# 删除重复记录
df = df.drop_duplicates(subset=["id"]).copy()

# 清理多余空格与换行
df["content"] = (
    df["content"]
   .astype(str)
   .str.replace("\n","", regex=False)
   .str.replace("\r","", regex=False)
   .str.replace("\t","", regex=False)
   .str.replace(r"\s+","", regex=True)
   .str.strip()
)

print(df.head())
print("样本量：", len(df))
max_len = 1500
df["content_short"] = df["content"].str[:max_len]

def build_prompt(text):
    prompt = f"""
        你是一名金融文本分析助手。请根据给定的上市公司公告，完成以下任务：

        1. 用一句话概括公告的核心内容，语言简洁，不超过40字。
        2. 判断该公告对公司短期市场预期的影响方向，只能输出“利好”“中性”或“利空”。
        3. 判断事件类型，只能从以下类别中选择一个：
        “业绩”“融资”“并购重组”“重大合同”“监管处罚”“人事变动”“诉讼仲裁”“产品发布”“风险提示”“其他”。
        4. 提取文本中的主要风险点，要求用一句话概括。如果文本中未明显出现风险点，可填写“未明显提及”。

        请严格按照以下 JSON 格式输出，不要输出任何额外解释：

        {{
       "summary":"",
       "sentiment":"",
       "event_type":"",
       "risk_point":""
        }}

        公告文本如下：
        {text}
       """
    return prompt

import json
import time
import  random
from  typing  import  Dict,  Any
 
#  -----------------演示版（手动生成JSON）--------------------------
def call_llm_api_demo(prompt:str) ->  str: 
    '''演示版：基于预设数据随机生成
    JSON格式字符串，无需调用真实大模型
    :param prompt:输入提示词（演示版暂不使用，仅保持函数签名一致）
    :return:JSON格式字符串
    '''
    #  预设的业务数据（基于你提供的信息）
    preset_data = [
        {"summary" : "公司预计年度净利润显著增长" ,  
       "sentiment" : "利好" ,
       "event_type" : "业绩" ,
       "risk_point" : "增长持续性存在不确定性"
        },
        {
       "summary": "公司签署大额长期供货合同" ,
       "sentiment" : "利好" ,
       "event_type" : "重大合同" ,
       "risk_point" : "合同履约进度存在风险"
        },
        {
       "summary" : "公司因信息披露问题收到监管措施" ,
       "sentiment" : "利空" ,
       "event_type" : "监管处罚" ,
       "risk_point" : "后续合规整改压力较大"
        },
        {
       "summary" : "公司披露重大诉讼最新进展" ,
       "sentiment" : "利空" ,
        "event_type" : "诉讼仲裁" ,
       "risk_point" : "案件结果可能影响经营稳定性"
        }
        ]

    try :
        # 随机选择一条预设数据（也可扩展为随机组合字段，示例为整行随机）
        random_result = random.choice(preset_data)
        # 转为标准JSON字符串（ ensure_ascii=False支持中文）
        json_result = json.dumps(random_result, ensure_ascii=False, indent=2)
        return json_result
    except  Exception  as  e:
        # 异常处理：返回错误信息的JSON
        error_info  =  {"error": f"演示版生成失败：{str(e)}"}
        return  json.dumps(error_info,  ensure_ascii=False)

def call_llm_api(prompt:  str)  ->  str:
    '''真实版：调用OpenAI大模型API生成指定格式的JSON字符串
    （可替换为其他大模型：如智谱、百度文心、阿里通义等）
      param  prompt:  输入提示词，要求模型生成包
    含summary/sentiment/event_type/risk_point的JSON
    :return: JSON格式字符串
    '''
    #  步骤1  ：安装依赖（需先执行）
    #  pip  install  openai
    # 基于以下输入生成内容：{prompt}
    try:
        # 调用OpenAI  GPT-3.5/4  API
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo" ,  #  模型版本，可替换为gpt-4
        messages=[
        {"role": "user" , "content": enhanced_prompt}
        ],
        temperature=0.7,  #  随机性，0-1之间
        timeout=30  #  超时时间
        )

        #  提取模型返回的内容并转为JSON（先校验格式）
        llm_output = response.choices[0] .message.content.strip()
        #  确保返回的是合法JSON（去除可能的markdown  包裹）
        if llm_output.startswith("```json") and llm_output.endswith("```"):
            llm_output = llm_output[7:-3] .strip()
        # 验证JSON格式并返回
        json.loads(llm_output)  #  校验合法性，不合法则抛异常
        return llm_output
    except openai.error.OpenAIError  as  e:
        # 大模型API调用异常
        error_info = {"error": f"OpenAI API调用失败：{str(e)}"}
        return json.dumps(error_info,  ensure_ascii=False)
    except json.JSONDecodeError  as  e:
        #  JSON格式异常
        error_info = {"error": f"返回内容非合法JSON：{str(e)}，原始内容:{llm_output}"}
        return  json.dumps(error_info ,  ensure_ascii=False)
    except Exception  as  e:
        #  其他异常
        error_info = {"error": f"未知错误：{str(e)}"}
        return  json.dumps(error_info ,  ensure_ascii=False)
    
def extract_info_from_text(text):
    prompt = build_prompt(text)
    response_text = call_llm_api_demo(prompt)
    
    # 去除可能出现的代码块标记
    response_text = response_text.strip()
    response_text = response_text.replace("```json","").replace("```","").strip()
    
    result = json.loads(response_text)
    return result

results = []

for idx, row in df.head(100).iterrows():
    text = row["content_short"]
    try:
        result = extract_info_from_text(text)
        result["id"] = row["id"]
        result["date"] = row["date"]
        result["company"] = row["company"]
        results.append(result)
        
        print(f"已完成第 {idx + 1} 条样本")
        time.sleep(1)  # 根据实际接口限制调整
        
    except Exception as e:
        print(f"第 {idx + 1} 条样本处理失败：{e}")
        results.append({
           "id": row["id"],
           "date": row["date"],
           "company": row["company"],
           "title": row["title"],
           "summary": None,
           "sentiment": None,
           "event_type": None,
           "risk_point": None
        })

result_df = pd.DataFrame(results)
print(result_df.head())

result_df.to_csv("./data/announcement_analysis_results.csv", index=False, encoding="utf-8-sig")


cross_tab = pd.crosstab(result_df["event_type"], result_df["sentiment"])
print(cross_tab)


import matplotlib.pyplot as plt

sentiment_counts = result_df["sentiment"].value_counts()

plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind="bar")
plt.title("公告情绪分布")
plt.xlabel("情绪标签")
plt.ylabel("频数")
plt.tight_layout()
plt.show()


event_counts = result_df["event_type"].value_counts()

plt.figure(figsize=(10, 5))
event_counts.plot(kind="bar")
plt.title("公告事件类型分布")
plt.xlabel("事件类型")
plt.ylabel("频数")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


company_name ="某公司"
company_df = result_df[result_df["company"] == company_name]
print(company_df[["date","event_type","sentiment","summary","risk_point"]])