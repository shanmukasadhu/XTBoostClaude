# # test_openai_api.py (for openai>=1.0.0)

# import os
# from openai import OpenAI
# from dotenv import load_dotenv

# # 加载 .env
# load_dotenv()

# # 初始化 OpenAI 客户端
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("OPENAI_API_BASE")  # 可选，默认是 https://api.openai.com/v1
# )

# # 获取模型名
# model_name = os.getenv("OPENAI_MODEL_NAME")

# # 发起 ChatCompletion 请求
# response = client.chat.completions.create(
#     model=model_name,
#     messages=[
#         {"role": "user", "content": "Say hi"}
#     ],
#     temperature=0.5,
# )

# print("API 返回结果：", response.choices[0].message.content)


# import os
# from openai import OpenAI

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# response = client.chat.completions.create(
#     model="o3", 
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Say hello in one sentence."}
#     ],
#     temperature=1,  
#     max_completion_tokens=500, 
# )

# print("Raw response:")
# print(response)  

# print("Model response:")
# print(response.choices[0].message.content)




import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="o3",
    input=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one sentence."}
    ],
)

print("Raw response:")
print(response)

for item in response.output:
    if hasattr(item, "content") and item.content:
        for sub in item.content:
            print("Model response:")
            print(sub.text)