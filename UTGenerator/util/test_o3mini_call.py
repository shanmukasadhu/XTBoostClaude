# # File: UTGenerator/util/test_o3mini_call.py

# import os, openai
# import httpx
# import signal
# import time
# from dotenv import load_dotenv


# # Load .env
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
# model_name = os.getenv("OPENAI_MODEL_NAME", "o3-mini")

# print("Model:", model_name)
# print("API Base:", base_url)
# print("API Key Present:", bool(api_key))


# if not api_key:
#     raise RuntimeError("OPENAI_API_KEY is not set!")

# os.environ["OPENAI_API_KEY"] = api_key

# def handler(signum, frame):
#     raise Exception("Timeout!")

# def create_chatgpt_config(message, max_tokens=300, model="o3-mini"):
#     system_message = (
#         "You are a software engineer responsible for generating test cases to verify that a specific bug fix works. "
#         "Given the problem description and patch, output relevant and minimal test cases that trigger and validate the fix. "
#         "Respond with code only, no explanation."
#     )
#     messages = [
#         {"role": "system", "content": system_message},
#         {"role": "user", "content": message}
#     ]
#     config = {
#         "model": model,
#         "messages": messages,
#         "max_completion_tokens": max_tokens,
#     }
#     return config

# def request_chatgpt_engine(config, base_url=None):
#     client = openai.OpenAI(
#         api_key=os.getenv("OPENAI_API_KEY"),
#         base_url=base_url or os.getenv("OPENAI_API_BASE"),
#         timeout=30.0,
#         max_retries=2,
#         http_client=httpx.Client(
#             base_url=base_url or os.getenv("OPENAI_API_BASE"),
#             follow_redirects=True,
#         )
#     )

#     ret = None
#     while ret is None:
#         try:
#             signal.signal(signal.SIGALRM, handler)
#             signal.alarm(30)
#             print("Sending request to model...")
#             ret = client.chat.completions.create(**config)
#             signal.alarm(0)

#             for i, choice in enumerate(ret.choices):
#                 print(f"\n[Result {i}]\n{choice.message.content.strip()}\n")
            
#             if all(choice.message.content.strip() == "" for choice in ret.choices):
#                 print("Empty completion returned.")
#         except Exception as e:
#             print("Error:", e)
#             time.sleep(3)
#             signal.alarm(0)
#     return ret

# if __name__ == "__main__":
#     patch_prompt = "```diff\n- return a\n+ return a + 1\n```\nWrite a pytest function that captures this change."
    
#     config = create_chatgpt_config(
#         patch_prompt,
#         max_tokens=300,
#         model="o3-mini"
#     )
    
#     # Optional: enable streaming if needed
#     # config["stream"] = True

#     resp = request_chatgpt_engine(config)

#     # Print response normally (stream=False)
#     for i, choice in enumerate(resp.choices):
#         print(f"\n[Result {i}]\n{choice.message.content.strip()}\n")
# File: UTGenerator/util/test_o3_basic.py
# File: UTGenerator/util/test_o3mini_call.py

# from openai import OpenAI

# client = OpenAI()

# context = [{"role": "user", "content": "Write a pytest function that tests this change:\n```diff\n- return a\n+ return a + 1\n```"}]

# response = client.responses.create(
#     model="o3",  # not "o3-mini"
#     input=context,
#     store=False,
#     include=["reasoning.encrypted_content"]
# )

# for i, item in enumerate(response.output):
#     print(f"\n[Step {i}]\n", item)


from UTGenerator.util.api_requests import create_chatgpt_config, request_chatgpt_engine

test_prompt = [
    {"role": "user", "content": "Given the following patch:\n\n```diff\n- return a\n+ return a + 1\n```\nWrite a minimal test that fails before this patch and passes after."}
]

config = create_chatgpt_config(
    test_prompt,
    max_tokens=300,
    model="o3", 
)

resp = request_chatgpt_engine(config)
print("Response:", resp.choices[0].message.content)
