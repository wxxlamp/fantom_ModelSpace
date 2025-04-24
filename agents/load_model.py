# from .claude import AsyncClaudeAgent
# from .gemini import AsyncGeminiAgent
# from .gpt import GPT3BaseAgent, AsyncConversationalGPTBaseAgent
# from .light_weight import LightweightHFAgent
from .deepseek import DeepSeekAgent
from .qwen import Qwen
from .qwen_small import QwenSmall


# from .together_ai import AsyncTogetherAIAgent, AsyncLlama3Agent

def load_model(model_name, **kwargs):
    # if model_name.startswith("text-"):
    #     model = GPT3BaseAgent({'engine': model_name, 'temperature': 0, 'top_p': 1.0, 'frequency_penalty': 0.0, 'presence_penalty': 0.0})
    # elif model_name.startswith("gpt-"):
    #     model = AsyncConversationalGPTBaseAgent({'model': model_name, 'temperature': 0, 'top_p': 1.0, 'frequency_penalty': 0.0, 'presence_penalty': 0.0})
    # elif model_name.startswith('gemini-'):
    #     model = AsyncGeminiAgent({'model': model_name, 'temperature': 0, 'max_tokens': 256})
    # elif model_name.startswith('claude-'):
    #     model = AsyncClaudeAgent({'model': model_name, **kwargs})
    # elif model_name in ["meta-llama/Llama-3-70b-chat-hf-tg", "meta-llama/Llama-3-8b-chat-hf-tg"]:
    #     model = AsyncLlama3Agent({'model': model_name, 'temperature': 0, 'max_tokens': 256, **kwargs})
    if model_name.startswith('text-qwen-small'):
        model = QwenSmall(**kwargs)
    elif 'qwen' in model_name:
        model = Qwen(**kwargs)
    elif 'deepseek' in model_name:
        model = DeepSeekAgent(**kwargs)
    else:
        raise NotImplementedError

    return model
