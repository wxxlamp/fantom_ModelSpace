import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .huggingface import HuggingFaceAgent

class DeepSeekAgent(HuggingFaceAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            trust_remote_code=True,  # DeepSeek模型需要此选项
            padding_side='left'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            device_map="auto",
            torch_dtype=torch.float16,  # 使用半精度，节省显存
            trust_remote_code=True,     # DeepSeek模型需要此选项
            **kwargs
        )

        # 设置对话模板参数 - DeepSeek的对话格式
        self.chat_template = {
            "system": "<system>\n{content}\n</system>",
            "user": "<user>\n{content}\n</user>",
            "assistant": "<assistant>\n{content}\n</assistant>"
        }
        self.init_pipeline()

    def preprocess_input(self, text):
        """应用DeepSeek的对话模板"""
        # 对于简单的用户输入，直接使用user模板
        formatted_prompt = self.chat_template["user"].format(content=text)
        return formatted_prompt

    def postprocess_output(self, output):
        """提取生成的回答"""
        # 从输出中提取助手部分的回答
        if "<assistant>" in output:
            # 只提取最后一个助手回答部分
            response = output.split("<assistant>")[-1]
            if "</assistant>" in response:
                response = response.split("</assistant>")[0]
            return response.strip()
        return output.strip()

    def generate(self, prompt, temperature=0.7, max_tokens=512, **kwargs):
        # 应用预处理模板
        formatted_prompt = self.preprocess_input(prompt)

        # 设置生成参数
        generation_args = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.95),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        # 使用pipeline生成
        outputs = self.pipe(
            formatted_prompt,
            **generation_args
        )

        return self.postprocess_output(outputs[0]['generated_text'])
