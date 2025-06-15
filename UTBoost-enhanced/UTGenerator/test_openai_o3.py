# from openai import OpenAI

# from dotenv import load_dotenv
# load_dotenv()
# client = OpenAI()

# def create_chat_completion(message: str):
#     response = client.chat.completions.create(
#         model="o3-mini",  # 或 "o4-mini"
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": message},
#         ],
#         max_completion_tokens=512  # 注意：不是 max_tokens，而是 max_completion_tokens
#     )
#     return response.choices[0].message.content

import os
import json
import requests
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI()

# 定义工具函数
def list_project_files():
    """列出项目中所有 Python 文件"""
    file_paths = []
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".py"):
                file_paths.append(os.path.join(root, f))
    return file_paths

# 定义工具 schema
tools = [{
    "type": "function",
    "name": "list_project_files",
    "description": (
        "Call this function to get a list of all Python files in the project. "
        "Use this to identify relevant source files for test case insertion. "
        "Only use this when source files are needed for further analysis."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True
}]

# 初始上下文 (developer prompt + user input)
context = [
    {
        "role": "system",
        "content": (
            "You are a software reasoning agent tasked with identifying relevant source code files "
            "in a Python project where test cases should be inserted to verify a bug fix. "
            "You can use tools to explore the file structure if needed. "
            "Only return the file paths relevant to the test insertion in a markdown code block. "
            "Use tools proactively and avoid hallucinating future actions. Call functions immediately when needed."
        )
    },
    {
        "role": "user",
        "content": (
            "I want to insert tests for a recent bug fix. The patch modifies functionality in some files. "
            "Help me identify the relevant source files for test case insertion."
        )
    }
]

# 第一次请求：模型调用工具
response = client.responses.create(
    model="o3",
    input=context,
    tools=tools,
    store=False,
    include=["reasoning.encrypted_content"]
)

context += response.output

tool_call = response.output[1]
args = json.loads(tool_call.arguments)

result = list_project_files()


context.append({
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": json.dumps(result)
})

response_2 = client.responses.create(
    model="o4-mini",
    input=context,
    tools=tools,
    store=False,
    include=["reasoning.encrypted_content"]
)

print(response_2.output_text)



# from UTGenerator.util.api_requests import create_chatgpt_config, request_chatgpt_engine

# config = create_chatgpt_config(
#     message=(
#         "You are an AI assistant. Your task is to output a list of file paths from a Python project.\n"
#         "Return them in a Markdown code block with triple backticks.\n"
#         "Example:\n"
#         "```\n"
#         "src/module/test_xyz.py\n"
#         "src/module/feature.py\n"
#         "```"
#     ),
#     model="gpt-4", 
#     max_tokens=256,
# )

# responses, usage = request_chatgpt_engine(config)
# print("Final Response:", responses)

