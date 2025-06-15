import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")  

model_name = os.getenv("OPENAI_MODEL_NAME")

if not all([openai.api_key, model_name]):
    raise ValueError("OPENAI_API_KEY 或 OPENAI_MODEL_NAME 没有加载成功")

response = openai.ChatCompletion.create(
    model=model_name,
    messages=[{"role": "user", "content": "Say hi"}],
    temperature=0.5
)

print("API 返回结果：", response.choices[0].message["content"])
