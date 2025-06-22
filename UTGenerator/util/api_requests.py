import signal
import time
from typing import Dict, Union
import pdb
import openai
import httpx
import tiktoken
import os

from dotenv import load_dotenv


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")

def num_tokens_from_messages(message, model="o3"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    # system_message: str = "You are a software engineer for testing whether an issue is solved, your task is to generate test cases for the given code. Just give me the test cases, no need to explain. Do not delete old test cases or functions",
    model: str = "o3",
) -> Dict:
    model = model or "o3-pro"

    if isinstance(message, list):
        config = {
            "model": model,
            # "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            # "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }

    if model.startswith("gpt-4o") or model.startswith("o3"):
        config["max_completion_tokens"] = max_tokens
        config["temperature"] = 1 
    else:
        config["max_tokens"] = max_tokens
        config["temperature"] = temperature  

    return config



def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")



def request_chatgpt_engine(config, base_url=None):
    import openai
    import httpx
    import signal
    import time
    import os

    def handler(signum, frame):
        raise Exception("Request timed out")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    model = config.get("model", "")
    is_chat_model = not (model.startswith("o3") or model.startswith("gpt-4o"))

    print(f"[DEBUG] model: {model}")
    print(f"[DEBUG] is_chat_model: {is_chat_model}")

    # 建立 client
    if base_url:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(base_url=base_url, follow_redirects=True),
        )
    else:
        client = openai.OpenAI(api_key=api_key)

    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)

            if is_chat_model:
                ret = client.chat.completions.create(**config)
            else:
                # o3 模型
                ret = client.responses.create(
                    model=model,
                    input=config["messages"],
                    temperature=config.get("temperature", 1),
                )

            signal.alarm(0)

        except openai.BadRequestError as e:
            print("[BadRequestError]", e)
            break  # usually unrecoverable
        except openai.RateLimitError as e:
            print("[RateLimitError] Retrying in 5s...")
            time.sleep(5)
        except openai.APIConnectionError as e:
            print("[APIConnectionError] Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            print("[Unknown Exception]", e)
            time.sleep(1)
        finally:
            signal.alarm(0)

    # usage info
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if hasattr(ret, "usage"):
        usage["prompt_tokens"] = getattr(ret.usage, "prompt_tokens", 0)
        usage["completion_tokens"] = getattr(ret.usage, "completion_tokens", 0)

    if hasattr(ret, "choices"):
        return ret.choices, usage
    else:
        return [ret], usage




def print_response_output(response):
    """
    Prints the model response content based on its structure.
    Supports both chat models and response models (like o3).
    """
    print("Model response:")
    try:
        # Case 1: chat model (gpt-3.5, gpt-4, etc.)
        if hasattr(response, "choices") and response.choices:
            print(response.choices[0].message.content.strip())

        # Case 2: o3/o4 response model
        elif hasattr(response, "output"):
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for sub in item.content:
                        if hasattr(sub, "text"):
                            print(sub.text.strip())

        else:
            print("[Warning] Unknown response format. Raw output:")
            print(response)

    except Exception as e:
        print("[Error] Failed to print model response:", e)
        print("Raw response:")
        print(response)



def create_anthropic_config(
    message: str,
    prefill_message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system_message,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system_message,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": prefill_message},
            ],
        }
    return config


def request_anthropic_engine(client, config):
    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = client.messages.create(**config)
            signal.alarm(0)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(10)
    return ret


def read_python_file(file_path):
    """
    This function reads the entire content of a Python file and returns it as a string.
    
    Parameters:
        file_path (str): The path to the Python file.
        
    Returns:
        str: The content of the file.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"The file at {file_path} was not found."
    except Exception as e:
        return f"An error occurred: {e}"



# import signal
# import time
# from typing import Dict, List, Union
# import openai
# import httpx
# import tiktoken
# import os
# from dotenv import load_dotenv

# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# # Token counting utility
# def num_tokens_from_messages(message, model="o3-mini"):
#     try:
#         encoding = tiktoken.encoding_for_model(model)
#     except KeyError:
#         encoding = tiktoken.get_encoding("cl100k_base")
#     if isinstance(message, list):
#         return sum(len(encoding.encode(m["content"])) for m in message)
#     else:
#         return len(encoding.encode(message))


# # def create_chatgpt_config(
# #     message: Union[str, list],
# #     max_tokens: int,
# #     temperature: float = 1,
# #     batch_size: int = 1,
# #     system_message: str = "You are a helpful assistant.",
# #     # system_message: str = "You are a software engineer for testing whether an issue is solved, your task is to generate test cases for the given code. Just give me the test cases, no need to explain. Do not delete old test cases or functions",
# #     model: str = "o3-mini-2025-01-31",
# # ) -> Dict:
# #     if isinstance(message, list):
# #         config = {
# #             "model": model,
# #             "max_tokens": max_tokens,
# #             "temperature": temperature,
# #             "n": batch_size,
# #             "messages": [{"role": "system", "content": system_message}] + message,
# #         }
# #     else:
# #         config = {
# #             "model": model,
# #             "max_tokens": max_tokens,
# #             "temperature": temperature,
# #             "n": batch_size,
# #             "messages": [
# #                 {"role": "system", "content": system_message},
# #                 {"role": "user", "content": message},
# #             ],
# #         }
# #     return config


# # o3-mini

# # ======== ChatGPT Config Builder (unified) ========
# def create_chatgpt_config(
#     message: Union[str, List[Dict[str, str]]],
#     model: str = "o3-mini",
#     system_message: str = "You are a helpful assistant.",
#     temperature: float = 0.7,
#     max_tokens: int = 512,
#     batch_size: int = 1, # Note: n > 1 is not supported in the new request_chatgpt_engine
#     tools: List[Dict] = None,
#     tool_choice: str = "auto",
# ) -> Dict:
#     if not system_message:
#         system_message = (
#             "You are a software engineer responsible for generating minimal test cases. Output only code."
#         )

#     messages = [{"role": "system", "content": system_message}]
#     if isinstance(message, list):
#         messages.extend(message)
#     else:
#         messages.append({"role": "user", "content": message})

#     config = {
#         "model": model,
#         "messages": messages,
#     }

#     # Use max_completion_tokens for o3/o4 models, otherwise use standard parameters
#     if model.startswith("o3") or model.startswith("o4"):
#         config["max_completion_tokens"] = max_tokens
#     else:
#         config["max_tokens"] = max_tokens
#         config["temperature"] = temperature

#     if tools:
#         config["tools"] = tools
#         config["tool_choice"] = tool_choice

#     return config



# def handler(signum, frame):
#     # swallow signum and frame
#     raise Exception("end of time")



# def request_chatgpt_engine(config: Dict, base_url: str = None):
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

#     # Dynamically build the parameters for the API call
#     params = {
#         "model": config["model"],
#         "messages": config["messages"],
#     }
#     if "max_completion_tokens" in config:
#         params["max_completion_tokens"] = config["max_completion_tokens"]
#     if "max_tokens" in config:
#         params["max_tokens"] = config["max_tokens"]
#     if "temperature" in config:
#         params["temperature"] = config["temperature"]

#     # *** THIS IS THE FIX ***
#     # Only add 'tools' and 'tool_choice' if they exist in the config
#     if config.get("tools"):
#         params["tools"] = config["tools"]
#         params["tool_choice"] = config.get("tool_choice", "auto")

#     try:
#         response = client.chat.completions.create(**params)
        
#         # This function will now only handle non-tool-use cases directly.
#         # The logic for handling tool calls has been removed as it was incorrect.
#         # The calling function (like LLMFL.localize) is now responsible for any multi-step logic.
#         print("======== Raw Model Output ========")
#         for i, choice in enumerate(response.choices):
#             print(f"[{i}] {choice.message.content!r}")
        
#         outputs = [c.message.content for c in response.choices]
#         usage = response.usage.to_dict() if response.usage else None
#         return outputs, usage

#     except Exception as e:
#         print(f"Request error: {e}")
#         return [""], None # Return empty string on error to prevent crashes



# def request_anthropic_engine(client, config):
#     ret = None
#     while ret is None:
#         try:
#             signal.signal(signal.SIGALRM, handler)
#             signal.alarm(100)
#             ret = client.messages.create(**config)
#             signal.alarm(0)
#         except Exception as e:
#             print("Unknown error. Waiting...")
#             print(e)
#             signal.alarm(0)
#             time.sleep(10)
#     return ret


# def read_python_file(file_path):
#     """
#     This function reads the entire content of a Python file and returns it as a string.
    
#     Parameters:
#         file_path (str): The path to the Python file.
        
#     Returns:
#         str: The content of the file.
#     """
#     try:
#         with open(file_path, 'r') as file:
#             content = file.read()
#         return content
#     except FileNotFoundError:
#         return f"The file at {file_path} was not found."
#     except Exception as e:
#         return f"An error occurred: {e}"


# def list_project_files(directory="."):
#     file_paths = []
#     for root, _, files in os.walk(directory):
#         for f in files:
#             if f.endswith(".py"):
#                 file_paths.append(os.path.join(root, f))
#     return file_paths
