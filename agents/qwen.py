from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from .base import BaseAgent

class DeepSeekAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__()

        # 获取设备信息
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_memory = 20 * 1024**3  # 20GB安全阈值
        self.batch_size = 4  # 根据显存动态调整

        # 简化参数设置
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 512)
        self.top_p = kwargs.get('top_p', 0.95)

        # 优化模型加载
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            max_memory={0: "20GiB"}  # 显存硬限制
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            trust_remote_code=True
        )

        # 设置对话模板
        self.chat_template = {
            "system": "<system>\n{content}\n</system>",
            "user": "<user>\n{content}\n</user>",
            "assistant": "<assistant>\n{content}\n</assistant>"
        }

    def preprocess_input(self, text):
        """将输入文本转换为适合模型的格式"""
        formatted_prompt = self.chat_template["user"].format(content=text)
        return formatted_prompt

    def postprocess_output(self, output):
        """强化答案提取逻辑"""
        # 提取<assistant>标签后的内容
        if "<assistant>" in output:
            response = output.split("<assistant>")[-1]
            # 去除后续标签和无关内容
            response = response.split("</assistant>")[0].strip()
            # 提取第一个完整句子
            if '.' in response:
                response = response.split('.')[0] + '.'
            return response
        return output.strip()

    def generate(self, prompt, temperature=None, max_tokens=None):
        """生成回复"""
        # 使用默认值或传入的参数
        temp = temperature if temperature is not None else self.temperature
        max_len = max_tokens if max_tokens is not None else self.max_tokens

        # 预处理输入
        formatted_prompt = self.preprocess_input(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # 生成输出
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_len,
                temperature=temp,
                top_p=self.top_p,
                repetition_penalty=1.1,
                do_sample=(temp > 0),
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码输出
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 后处理
        return self.postprocess_output(response_text)

    def _check_memory(self):
        """实时监控显存使用"""
        used = torch.cuda.memory_allocated()
        if used > self.max_memory:
            self.batch_size = max(1, self.batch_size // 2)
            torch.cuda.empty_cache()

    def interact(self, prompt):
        """简化单次调用"""
        return self.batch_interact([prompt])[0]

    def batch_generate(self, prompts, temperature=None, max_tokens=None):
        """批量生成回复"""
        return [self.generate(prompt, temperature, max_tokens) for prompt in prompts]

    def batch_interact(self, texts):
        """优化批量推理"""
        self._check_memory()

        # 自动批处理
        responses = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=(self.temperature > 0),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            json.dumps(outputs)
            batch_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            json.dumps(outputs)
            responses.extend([self.postprocess_output(r) for r in batch_responses])
            json.dumps(responses)
            self._check_memory()  # 每个批次后检查显存
            raise NotImplementedError
        return responses
