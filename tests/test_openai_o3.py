import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def list_project_files():
    file_paths = []
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".py"):
                file_paths.append(os.path.join(root, f))
    return file_paths

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

result = list_project_files()[:10]


context.append({
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": json.dumps(result)
})

print(context)

response_2 = client.responses.create(
    model="o4-mini",
    input=context,
    tools=tools,
    store=False,
    include=["reasoning.encrypted_content"]
)

print(response_2.output_text)

