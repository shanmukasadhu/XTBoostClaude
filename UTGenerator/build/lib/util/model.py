# from abc import ABC, abstractmethod
# from typing import List

# from UTGenerator.util.api_requests import create_chatgpt_config, request_chatgpt_engine


# class DecoderBase(ABC):
#     def __init__(
#         self,
#         name: str,
#         batch_size: int = 1,
#         temperature: float = 0.8,
#         max_new_tokens: int = 1024,
#     ) -> None:
#         print("Initializing a decoder model: {} ...".format(name))
#         self.name = name
#         self.batch_size = batch_size
#         self.temperature = temperature
#         self.max_new_tokens = max_new_tokens

#     @abstractmethod
#     def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
#         pass

#     @abstractmethod
#     def is_direct_completion(self) -> bool:
#         pass

#     def __repr__(self) -> str:
#         return self.name

#     def __str__(self) -> str:
#         return self.name


# class OpenAIChatDecoder(DecoderBase):
#     def __init__(self, name: str, **kwargs) -> None:
#         super().__init__(name, **kwargs)

#     def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
#         if self.temperature == 0:
#             assert num_samples == 1
#         batch_size = min(self.batch_size, num_samples)

#         config = create_chatgpt_config(
#             message=message,
#             max_tokens=self.max_new_tokens,
#             temperature=self.temperature,
#             batch_size=batch_size,
#             model=self.name,
#         )
#         import os
#         base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
#         ret = request_chatgpt_engine(config, base_url=base_url)

#         responses = [choice.message.content for choice in ret.choices]
#         # The nice thing is, when we generate multiple samples from the same input (message),
#         # the input tokens are only charged once according to openai API.
#         # Therefore, we assume the request cost is only counted for the first sample.
#         # More specifically, the `prompt_tokens` is for one input message,
#         # and the `completion_tokens` is the sum of all returned completions.
#         # Therefore, for the second and later samples, the cost is zero.
#         trajs = [
#             {
#                 "response": responses[0],
#                 "usage": {
#                     "completion_tokens": ret.usage.completion_tokens,
#                     "prompt_tokens": ret.usage.prompt_tokens,
#                 },
#             }
#         ]
#         for response in responses[1:]:
#             trajs.append(
#                 {
#                     "response": response,
#                     "usage": {
#                         "completion_tokens": 0,
#                         "prompt_tokens": 0,
#                     },
#                 }
#             )
#         return trajs

#     def is_direct_completion(self) -> bool:
#         return False


# def make_model(
#     model: str,
#     backend: str,
#     batch_size: int = 1,
#     max_tokens: int = 1024,
#     temperature: float = 0.0,
# ):
#     if backend == "openai":
#         return OpenAIChatDecoder(
#             name=model,
#             batch_size=batch_size,
#             max_new_tokens=max_tokens,
#             temperature=temperature,
#         )
#     else:
#         raise NotImplementedError


from abc import ABC, abstractmethod
from typing import List

from UTGenerator.util.api_requests import create_chatgpt_config, request_chatgpt_engine, create_anthropic_config, request_anthropic_engine


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 2048,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)    
    def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )

        responses, usage = request_chatgpt_engine(config)
        responses = responses if responses else []

        trajs = []
        for i, response in enumerate(responses):
            # Extract unified response content
            content = None
            if hasattr(response, "message"):  # chat model
                content = response.message.content
            elif hasattr(response, "output"):  # response model
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for sub in item.content:
                            if hasattr(sub, "text"):
                                content = sub.text
                                break

            trajs.append(
                {
                    "response": content or "[empty]",
                    "usage": {
                        "completion_tokens": getattr(usage, "completion_tokens", 0) if i == 0 else 0,
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if i == 0 else 0,
                    },
                }
            )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class AnthropicChatDecoder(DecoderBase):
    """Anthropic Claude model decoder"""
    
    def __init__(self, name: str, system_message: str = "You are a helpful assistant.", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.system_message = system_message
    
    def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
        """Generate code or text using Anthropic Claude API"""
        if self.temperature == 0:
            assert num_samples == 1
            
        # Anthropic API currently doesn't support multiple samples in one request, so we have to generate sequentially
        trajs = []
        
        for _ in range(num_samples):
            config = create_anthropic_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                system_message=self.system_message,
                model=self.name,
            )
            
            response, usage = request_anthropic_engine(config)
            
            trajs.append({
                "response": response,
                "usage": {
                    "completion_tokens": usage["output_tokens"],
                    "prompt_tokens": usage["input_tokens"],
                }
            })
        
        return trajs
    
    def is_direct_completion(self) -> bool:
        return False


def make_model(
    model: str,
    backend: str,
    batch_size: int = 1,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    system_message: str = "You are a helpful assistant.",
):
    if backend == "openai":
        return OpenAIChatDecoder(
            name=model,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "anthropic":
        return AnthropicChatDecoder(
            name=model,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
            system_message=system_message,
        )
    else:
        raise NotImplementedError(f"Unsupported backend: {backend}")
